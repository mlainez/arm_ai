defmodule NxArm.StorageResizer do
  @moduledoc """
  First-boot expansion of the F2FS application data partition.

  On Nerves images that ship a small (256 MB) F2FS into a much
  bigger raw partition, `df` reports the small size and the rest of
  the disk is wasted. The fwup config grows the *partition* via
  `expand = true` but the F2FS inside stays at its mkfs size.

  This module compares filesystem size vs partition size at app
  start. If the FS is materially smaller, it:

    1. Unmounts the target.
    2. Runs `resize.f2fs` to grow the FS to the partition size.
    3. Remounts the target.

  All while the BEAM keeps running (it lives on `/srv/erlang` on
  squashfs, not on the target partition). Anything that wrote to
  the target before this ran would have done so on the 256 MB FS;
  the resize preserves data.

  The resize is idempotent — once the FS fills the partition,
  subsequent boots short-circuit.

  ## Configuration

      config :nx_arm, :storage_resizer,
        partition: "/dev/mmcblk0p62p3",
        mount_point: "/root",
        mount_opts: "nodev",
        # Skip resize when partition <= fs * threshold (default 1.1).
        # Avoids no-op resizes for negligible differences.
        threshold: 1.1

  Empty / `nil` config disables the resizer.
  """

  require Logger

  @doc """
  Run the resize check + grow once. Safe to call at any boot point
  before significant writes to the target partition. Returns one of:

    * `:already_grown` — FS already fills the partition
    * `:disabled` — no config / no partition specified
    * `{:ok, %{before: bytes, after: bytes}}` — grew the FS
    * `{:error, reason}` — failed (logged but not raised)
  """
  @spec run() ::
          :already_grown
          | :disabled
          | {:ok, %{before: non_neg_integer(), after: non_neg_integer()}}
          | {:error, term()}
  def run do
    case Application.get_env(:nx_arm, :storage_resizer) do
      nil ->
        :disabled

      config ->
        partition = Keyword.get(config, :partition)
        mount_point = Keyword.get(config, :mount_point, "/root")
        mount_opts = Keyword.get(config, :mount_opts, "nodev")
        threshold = Keyword.get(config, :threshold, 1.1)

        if partition == nil do
          :disabled
        else
          do_resize(partition, mount_point, mount_opts, threshold)
        end
    end
  end

  defp do_resize(partition, mount_point, mount_opts, threshold) do
    with {:ok, fs_bytes} <- fs_bytes(mount_point),
         {:ok, part_bytes} <- partition_bytes(partition),
         true <- part_bytes / max(fs_bytes, 1) >= threshold || :already_grown do
      Logger.info(
        "[nx_arm] resizing #{partition}: FS=#{mb(fs_bytes)} MB → partition=#{mb(part_bytes)} MB"
      )

      with :ok <- unmount(mount_point),
           :ok <- resize_f2fs(partition),
           :ok <- mount_f2fs(partition, mount_point, mount_opts),
           {:ok, after_bytes} <- fs_bytes(mount_point) do
        Logger.info("[nx_arm] resize ok: FS now #{mb(after_bytes)} MB")
        {:ok, %{before: fs_bytes, after: after_bytes}}
      else
        {:error, reason} = err ->
          Logger.error("[nx_arm] resize failed at step: #{inspect(reason)}; attempting remount")
          _ = mount_f2fs(partition, mount_point, mount_opts)
          err

        other ->
          Logger.error("[nx_arm] resize failed: #{inspect(other)}")
          _ = mount_f2fs(partition, mount_point, mount_opts)
          {:error, other}
      end
    else
      :already_grown ->
        :already_grown

      {:error, _} = err ->
        Logger.warning("[nx_arm] storage resizer skipping: #{inspect(err)}")
        err
    end
  end

  defp fs_bytes(mount_point) do
    case File.stat(mount_point) do
      {:ok, _} ->
        # Use POSIX -P + -k flags; works on busybox and full coreutils.
        case System.cmd("df", ["-P", "-k", mount_point], stderr_to_stdout: true) do
          {out, 0} ->
            with [_header, data | _] <- out |> String.trim() |> String.split("\n"),
                 [_fs, size_kb | _] <- data |> String.split(~r/\s+/, trim: true),
                 {kb, ""} <- Integer.parse(size_kb) do
              {:ok, kb * 1024}
            else
              other -> {:error, {:df_parse, other}}
            end

          {out, code} ->
            {:error, {:df_failed, code, out}}
        end

      _ ->
        {:error, :mount_point_missing}
    end
  end

  defp partition_bytes(partition) do
    # sysfs path works for plain block devices; for kpartx /
    # device-mapper subpartitions on Nerves boards (e.g. FP3 where
    # the userdata partition is split via dm into p62p1..p62p3) we
    # have to fall back to `blockdev --getsize64`.
    sysfs_size = "/sys/class/block/" <> Path.basename(partition) <> "/size"

    case File.read(sysfs_size) do
      {:ok, blocks_str} ->
        blocks = blocks_str |> String.trim() |> String.to_integer()
        {:ok, blocks * 512}

      {:error, _} ->
        case System.cmd("blockdev", ["--getsize64", partition], stderr_to_stdout: true) do
          {out, 0} ->
            case Integer.parse(String.trim(out)) do
              {n, _} -> {:ok, n}
              :error -> {:error, {:blockdev_parse, out}}
            end

          {out, code} ->
            {:error, {:blockdev_failed, code, out}}
        end
    end
  end

  defp unmount(mount_point) do
    case System.cmd("umount", [mount_point], stderr_to_stdout: true) do
      {_, 0} -> :ok
      {out, code} -> {:error, {:umount_failed, code, out}}
    end
  end

  defp resize_f2fs(partition) do
    case System.cmd("resize.f2fs", [partition], stderr_to_stdout: true) do
      {_, 0} -> :ok
      {out, code} -> {:error, {:resize_failed, code, out}}
    end
  end

  defp mount_f2fs(partition, mount_point, opts) do
    case System.cmd("mount", ["-t", "f2fs", "-o", opts, partition, mount_point],
           stderr_to_stdout: true
         ) do
      {_, 0} -> :ok
      {out, code} -> {:error, {:mount_failed, code, out}}
    end
  end

  defp mb(bytes), do: div(bytes, 1024 * 1024)
end
