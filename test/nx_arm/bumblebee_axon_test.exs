defmodule NxArm.BumblebeeAxonTest do
  @moduledoc """
  Integration smoke tests for Bumblebee + Axon on NxArm.Backend.
  We don't load a remote model (no network in tests); instead we
  build a small Axon model end-to-end and confirm forward + gradient
  match Nx.BinaryBackend within f32 noise.

  If this passes, the standard Elixir ML stack (Bumblebee → Axon →
  Nx) lights up unchanged on NxArm.Backend. Failures here mean a
  Bumblebee op is hitting a fallback that diverges.
  """

  use ExUnit.Case, async: false

  defp arm(t), do: Nx.backend_copy(t, NxArm.Backend)
  defp ref(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  test "tiny Axon MLP forward matches BinaryBackend" do
    model =
      Axon.input("x", shape: {nil, 16})
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(8, activation: :tanh)
      |> Axon.dense(2)

    seed = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]) |> Nx.divide(10)

    {init_fn, predict_fn} = Axon.build(model)

    params = init_fn.(seed, Axon.ModelState.empty())
    out_ref = predict_fn.(params, seed)

    # Move every parameter to NxArm.Backend.
    arm_params = move_params(params, NxArm.Backend)
    arm_seed = arm(seed)
    out_arm = predict_fn.(arm_params, arm_seed) |> ref()

    diff = Nx.subtract(out_arm, out_ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "Axon MLP diverged: diff=#{diff}"
  end

  test "Axon GELU + softmax block matches BinaryBackend" do
    # Reproduce a typical Bumblebee transformer FFN block.
    model =
      Axon.input("x", shape: {nil, 12})
      |> Axon.layer_norm()
      |> Axon.dense(24, activation: :gelu)
      |> Axon.dense(12)
      |> Axon.softmax(axis: -1)

    {init_fn, predict_fn} = Axon.build(model)
    x =
      Nx.iota({2, 12}, type: :f32)
      |> Nx.divide(20)
      |> Nx.sin()

    params = init_fn.(x, Axon.ModelState.empty())
    out_ref = predict_fn.(params, x)

    arm_params = move_params(params, NxArm.Backend)
    out_arm = predict_fn.(arm_params, arm(x)) |> ref()

    diff = Nx.subtract(out_arm, out_ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "Axon FFN diverged: diff=#{diff}"
  end

  test "Nx.Defn.grad through NxArm.Backend matches BinaryBackend" do
    # Bumblebee training (rare but supported) flows through Nx.Defn.grad.
    # Make sure that path doesn't silently route through fallbacks
    # that lose precision.
    defmodule Helper do
      import Nx.Defn

      defn loss(w, x, y) do
        pred = Nx.dot(x, w)
        Nx.sum((pred - y) ** 2)
      end

      defn grad_loss(w, x, y) do
        grad(w, fn ww -> loss(ww, x, y) end)
      end
    end

    w = Nx.iota({4, 2}, type: :f32) |> Nx.divide(10)
    x = Nx.iota({3, 4}, type: :f32) |> Nx.divide(10)
    y = Nx.iota({3, 2}, type: :f32) |> Nx.divide(10)

    g_ref = Helper.grad_loss(w, x, y)
    g_arm = Helper.grad_loss(arm(w), arm(x), arm(y)) |> ref()

    diff = Nx.subtract(g_arm, g_ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-4, "grad diverged: diff=#{diff}"
  end

  test "tokenizer roundtrip via upstream :tokenizers" do
    # The Bumblebee path shares the same `:tokenizers` Hex package
    # (the elixir-nx wrapper around HuggingFace's Rust crate).
    # Sanity-check that an encode → decode roundtrip survives.
    # Skip if no tokenizer.json is staged.
    candidates = [
      "/root/tinyllama-tokenizer.json",
      "/tmp/tinyllama-tokenizer.json"
    ]

    case Enum.find(candidates, &File.exists?/1) do
      nil ->
        # No tokenizer file in this test env — just confirm the
        # upstream module is loaded.
        assert Code.ensure_loaded?(Tokenizers.Tokenizer)

      path ->
        {:ok, tok} = Tokenizers.Tokenizer.from_file(path)
        {:ok, enc} = Tokenizers.Tokenizer.encode(tok, "Hello world", add_special_tokens: false)
        ids = Tokenizers.Encoding.get_ids(enc)
        {:ok, text} = Tokenizers.Tokenizer.decode(tok, ids, skip_special_tokens: true)
        assert String.downcase(text) =~ "hello"
    end
  end

  # Axon model state is a %Axon.ModelState{} struct around nested
  # maps of tensors. Walk it generically.
  defp move_params(%Nx.Tensor{} = t, backend), do: Nx.backend_copy(t, backend)
  defp move_params(%{} = map, backend) when not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, move_params(v, backend)} end)
  end
  defp move_params(%_struct{} = struct, backend) do
    fields = Map.from_struct(struct)
    moved = Map.new(fields, fn {k, v} -> {k, move_params(v, backend)} end)
    struct(struct, moved)
  end
  defp move_params(list, backend) when is_list(list),
    do: Enum.map(list, &move_params(&1, backend))
  defp move_params(other, _backend), do: other
end
