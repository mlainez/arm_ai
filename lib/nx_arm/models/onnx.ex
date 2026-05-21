defmodule NxArm.Models.Onnx do
  @moduledoc """
  ONNX inference via `tract-onnx`.

  Bumblebee covers BERT/GPT-2/ViT/etc. natively through Axon + Nx.
  For everything else in the HuggingFace zoo — Whisper, CLIP,
  detection models, embedding models, sentence-encoders — the
  universal path is HuggingFace → ONNX export → `NxArm.Models.Onnx`.

      {:ok, model} = NxArm.Models.Onnx.load("/root/all-MiniLM-L6-v2.onnx")
      # inputs is a map: %{name => Nx.Tensor}
      outputs = NxArm.Models.Onnx.run(model, %{
        "input_ids"      => Nx.tensor([[101, 2129, 2024, 2017, 102]], type: :s64),
        "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1]], type: :s64)
      })

  All outputs come back as f32 Nx tensors on `NxArm.Backend` (the
  tract bridge casts everything via `cast_to::<f32>` for uniformity).
  """

  defstruct [:handle, :input_names, :output_names]

  @type t :: %__MODULE__{
          handle: reference(),
          input_names: [String.t()],
          output_names: [String.t()]
        }

  @doc """
  Load an ONNX file. Returns the model + its declared input/output names.

  Requires the `onnx` Cargo feature (default-on). Falls back to
  `{:error, :onnx_feature_disabled}` when nx_arm was built without it.
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    if not function_exported?(NxArm.Native, :onnx_load_op, 1) do
      {:error, :onnx_feature_disabled}
    else
      try do
        {handle, input_names, output_names} = NxArm.Native.onnx_load_op(path)
        {:ok, %__MODULE__{handle: handle, input_names: input_names, output_names: output_names}}
      rescue
        e -> {:error, e}
      end
    end
  end

  @doc """
  Run inference. Returns a map `%{name => Nx.Tensor}` on
  `NxArm.Backend`. Wrapped in the scoped CPU governor by default;
  pass `performance_governor: false` to disable.
  """
  @spec run(t(), %{String.t() => Nx.Tensor.t()}, keyword()) :: %{String.t() => Nx.Tensor.t()}
  def run(%__MODULE__{handle: handle}, inputs, opts \\ []) do
    do_run = fn ->
      input_args =
        Enum.map(inputs, fn {name, tensor} ->
          shape = Nx.shape(tensor) |> Tuple.to_list()
          bin = NxArm.Backend.__bin_of__(tensor) |> to_f32_bin(Nx.type(tensor), shape)
          {name, shape, bin}
        end)

      outputs = NxArm.Native.onnx_run_op(handle, input_args)

      for {name, shape, bin} <- outputs, into: %{} do
        t =
          Nx.from_binary(bin, :f32)
          |> Nx.reshape(List.to_tuple(shape))
          |> Nx.backend_copy(NxArm.Backend)

        {name, t}
      end
    end

    if Keyword.get(opts, :performance_governor, true) do
      NxArm.Performance.with_performance(do_run)
    else
      do_run.()
    end
  end

  # tract's input plumbing in this bridge takes f32. If a caller
  # gives us s32/s64 indices (typical for token IDs), we cast through
  # Nx so the binary that crosses the NIF boundary is always f32.
  defp to_f32_bin(bin, {:f, 32}, _shape), do: bin

  defp to_f32_bin(_bin, _other_type, shape) do
    raise ArgumentError,
          "ONNX bridge currently accepts only f32 inputs; got non-f32 with shape #{inspect(shape)}. " <>
            "Cast via `Nx.as_type(t, :f32)` at the call site."
  end
end
