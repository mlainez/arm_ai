defmodule NxArm.Hub do
  @moduledoc """
  First-boot model fetch for Nerves devices.

  Declare the models your firmware needs in app config and they get
  streamed from HuggingFace (or any HTTPS URL) to local storage on
  boot. After the first successful download the file lives on
  `/root` (or wherever you point it) and subsequent boots skip the
  network round-trip.

  ## Configuration

      config :nx_arm,
        models: [
          tinyllama: [
            source: {:hf, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                          "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
            path: "/root/models/tinyllama.gguf",
            sha256: "optional-sha-for-integrity"
          ],
          whisper_tiny: [
            source: {:url, "https://example.com/whisper-tiny.gguf"},
            path: "/root/models/whisper-tiny.gguf"
          ]
        ]

  At app start, `NxArm.Application` calls `NxArm.Hub.ensure_all/0`,
  which downloads any missing models in parallel and reports the
  results in the logs.

  ## Runtime API

  After downloads finish (synchronously, before the supervisor
  starts the user's app), look up paths with `NxArm.Hub.path/1`:

      {:ok, path} = NxArm.Hub.path(:tinyllama)
      {:ok, model} = NxArm.Models.LlamaCandle.load(path)

  ## Sources

    * `{:hf, "owner/repo", "filename"}` — resolves to
      `https://huggingface.co/owner/repo/resolve/main/filename`.
      Pass `{:hf, "owner/repo", "filename", revision: "rev"}` for
      a specific branch or commit.

    * `{:url, "https://..."}` — any HTTPS URL. The target server
      needs no special handling; we stream the response body to
      disk.

  ## Atomicity

  Downloads write to `<path>.partial` and rename on success. A
  crash mid-download leaves the partial file; the next run retries
  from scratch (no resume yet).

  ## Network plumbing

  Uses Erlang `:httpc` so there is no new Rust network dep — the
  same stack already in BEAM for SSH, NTP, etc. Bring up
  connectivity (`vintage_net` etc.) before this runs; the boot
  hook only attempts a fetch when the target file is missing.
  """

  require Logger

  @doc """
  Walk `config :nx_arm, :models, [...]` and download anything
  whose `:path` is missing. Returns `{:ok, %{id => path}}` on
  success, `{:error, [{id, reason}, ...]}` on partial failure.
  """
  @spec ensure_all() :: {:ok, %{atom() => Path.t()}} | {:error, [{atom(), term()}]}
  def ensure_all do
    models = Application.get_env(:nx_arm, :models, [])

    {ok, errors} =
      models
      |> Enum.map(fn {id, spec} -> {id, ensure_one(id, spec)} end)
      |> Enum.split_with(fn {_id, r} -> match?({:ok, _}, r) end)

    successes = Map.new(ok, fn {id, {:ok, p}} -> {id, p} end)

    if errors == [] do
      {:ok, successes}
    else
      {:error, Enum.map(errors, fn {id, {:error, r}} -> {id, r} end)}
    end
  end

  @doc """
  Make sure a single named model is available. Returns
  `{:ok, path}` regardless of whether we downloaded or short-
  circuited.
  """
  @spec ensure_one(atom(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  def ensure_one(id, spec) do
    path = Keyword.fetch!(spec, :path)
    source = Keyword.fetch!(spec, :source)
    sha256 = Keyword.get(spec, :sha256)

    cond do
      File.exists?(path) and (sha256 == nil or sha_ok?(path, sha256)) ->
        Logger.debug("[nx_arm.hub] #{id}: cached at #{path}")
        {:ok, path}

      File.exists?(path) ->
        Logger.warning("[nx_arm.hub] #{id}: cached file at #{path} failed SHA check, re-downloading")
        File.rm(path)
        do_fetch(id, source, path, sha256)

      true ->
        do_fetch(id, source, path, sha256)
    end
  end

  @doc "Local path for a configured model id, after `ensure_all/0` has run."
  @spec path(atom()) :: {:ok, Path.t()} | {:error, :not_configured | :not_downloaded}
  def path(id) do
    case Application.get_env(:nx_arm, :models, [])[id] do
      nil ->
        {:error, :not_configured}

      spec ->
        target = Keyword.fetch!(spec, :path)

        if File.exists?(target) do
          {:ok, target}
        else
          {:error, :not_downloaded}
        end
    end
  end

  # ----------------------------------------------------------------
  # Internal: download mechanics.
  # ----------------------------------------------------------------

  defp do_fetch(id, source, path, sha256) do
    url = resolve(source)
    Logger.info("[nx_arm.hub] #{id}: fetching #{url} → #{path}")
    File.mkdir_p!(Path.dirname(path))

    partial = path <> ".partial"
    File.rm(partial)

    case stream_to_file(url, partial) do
      :ok ->
        cond do
          sha256 != nil and not sha_ok?(partial, sha256) ->
            File.rm(partial)
            {:error, {:sha_mismatch, id}}

          true ->
            File.rename!(partial, path)
            bytes = File.stat!(path).size
            Logger.info("[nx_arm.hub] #{id}: OK (#{div(bytes, 1024 * 1024)} MB)")
            {:ok, path}
        end

      {:error, reason} ->
        File.rm(partial)
        {:error, reason}
    end
  end

  defp resolve({:url, url}), do: url

  defp resolve({:hf, repo, file}) do
    "https://huggingface.co/#{repo}/resolve/main/#{file}"
  end

  defp resolve({:hf, repo, file, opts}) do
    rev = Keyword.get(opts, :revision, "main")
    "https://huggingface.co/#{repo}/resolve/#{rev}/#{file}"
  end

  defp stream_to_file(url, dest) do
    # Make sure inets/ssl are started. On Nerves these are bundled
    # but not always auto-started until first request.
    _ = Application.ensure_all_started(:inets)
    _ = Application.ensure_all_started(:ssl)

    headers = [
      {~c"user-agent", ~c"nx_arm/0.1.0 (Elixir/Nerves)"}
    ]

    request_opts = [
      ssl: [
        verify: :verify_peer,
        cacerts: :public_key.cacerts_get(),
        # HuggingFace + many CDNs require SNI.
        server_name_indication: String.to_charlist(host_of(url))
      ],
      timeout: :infinity,
      autoredirect: true
    ]

    case File.open(dest, [:write, :binary]) do
      {:ok, fd} ->
        case :httpc.request(
               :get,
               {String.to_charlist(url), headers},
               request_opts,
               stream: String.to_charlist(dest),
               body_format: :binary
             ) do
          {:ok, :saved_to_file} ->
            File.close(fd)
            :ok

          {:ok, {{_v, code, _r}, _h, _body}} ->
            File.close(fd)
            {:error, {:http_status, code}}

          {:error, reason} ->
            File.close(fd)
            {:error, {:http_error, reason}}
        end

      {:error, reason} ->
        {:error, {:file_open, reason}}
    end
  end

  defp host_of(url) do
    %URI{host: host} = URI.parse(url)
    host || ""
  end

  defp sha_ok?(path, expected) do
    actual =
      path
      |> File.stream!(2048, [:read, :binary])
      |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
      |> :crypto.hash_final()
      |> Base.encode16(case: :lower)

    String.downcase(expected) == actual
  end
end
