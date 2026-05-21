defmodule NxArm.WrappersErrorTest do
  use ExUnit.Case, async: true

  # Contract tests for the rest of the wrapper modules that the
  # models_api_test suite doesn't cover: missing-file / bad-input
  # paths must surface as `{:error, _}` or a controlled raise,
  # never an opaque crash.

  describe "NxArm.Audio" do
    test "decode_file on a missing path returns {:error, _}" do
      assert {:error, msg} = NxArm.Audio.decode_file("/tmp/__no_such_audio.wav")
      assert is_binary(msg)
    end

    test "to_mono on already-mono passes through" do
      mono = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      out = NxArm.Audio.to_mono(mono, 1)
      assert Nx.shape(out) == {4}
    end

    test "resample with from == to is a no-op (shape preserved)" do
      s = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32, backend: NxArm.Backend)
      out = NxArm.Audio.resample(s, 16_000, 16_000)
      assert Nx.shape(out) == Nx.shape(s)
    end
  end

  describe "upstream Safetensors / Tokenizers contracts (no nx_arm wrappers)" do
    test "Safetensors.read! raises on a missing file" do
      assert_raise File.Error, fn ->
        Safetensors.read!("/tmp/__no_such.safetensors")
      end
    end

    test "Tokenizers.Tokenizer.from_file returns {:error, _} on missing file" do
      assert {:error, _} = Tokenizers.Tokenizer.from_file("/tmp/__no_such_tokenizer.json")
    end
  end

  describe "NxArm.Performance" do
    test "with_performance runs and returns the fun's value when no perf cores" do
      # On a host with no /sys/devices/system/cpu/* tree (or empty
      # perf_cores), this should just invoke the function and return
      # its value without raising.
      assert :hello = NxArm.Performance.with_performance(fn -> :hello end)
    end

    test "current_governors returns {} for the empty list" do
      assert NxArm.Performance.current_governors([]) == %{}
    end
  end
end
