defmodule NxArm.GGUF do
  @moduledoc """
  Minimal reader for the llama.cpp GGUF v3 model format. Parses the
  header, metadata key/value pairs, and tensor catalogue; returns
  raw byte slices for each tensor on demand.

  Quantized tensors (Q4_0, Q8_0, …) are returned as their packed
  byte representations together with shape + dtype, so callers can
  feed them straight into the matching NxArm NIFs (e.g.
  `int4_matmul_f32_op` for Q4_0).
  """

  @magic "GGUF"

  # GGUF metadata value types (gguf.md).
  @vt_u8 0
  @vt_i8 1
  @vt_u16 2
  @vt_i16 3
  @vt_u32 4
  @vt_i32 5
  @vt_f32 6
  @vt_bool 7
  @vt_string 8
  @vt_array 9
  @vt_u64 10
  @vt_i64 11
  @vt_f64 12

  # GGUF tensor dtypes (subset of ggml types we expose by name).
  @tensor_types %{
    0 => :f32,
    1 => :f16,
    2 => :q4_0,
    3 => :q4_1,
    6 => :q5_0,
    7 => :q5_1,
    8 => :q8_0,
    9 => :q8_1,
    10 => :q2_k,
    11 => :q3_k,
    12 => :q4_k,
    13 => :q5_k,
    14 => :q6_k,
    15 => :q8_k,
    24 => :i8,
    25 => :i16,
    26 => :i32,
    27 => :i64,
    28 => :f64,
    29 => :bf16
  }

  defstruct [
    :version,
    :metadata,
    :tensors,
    :data_offset,
    :file_contents,
    :mmap_handle
  ]

  @type tensor_info :: %{
          name: String.t(),
          dtype: atom(),
          shape: [non_neg_integer()],
          offset: non_neg_integer(),
          byte_size: non_neg_integer()
        }

  @type t :: %__MODULE__{
          version: non_neg_integer(),
          metadata: map(),
          tensors: %{String.t() => tensor_info()},
          data_offset: non_neg_integer(),
          file_contents: binary() | nil,
          mmap_handle: reference() | nil
        }

  @doc """
  Read a GGUF file. Loads the full file into memory (use `read_mmap/1`
  on Linux once `E1` lands for very large models).
  """
  @spec read(Path.t()) :: {:ok, t()} | {:error, term()}
  def read(path) do
    case File.read(path) do
      {:ok, bin} -> parse(bin)
      {:error, _} = err -> err
    end
  end

  @doc """
  Memory-mapped GGUF reader. Parses the header + tensor catalogue
  from the file, then keeps an mmap handle so per-tensor reads page
  in from disk lazily. The right choice for multi-GB models on
  RAM-constrained devices.
  """
  @spec read_mmap(Path.t()) :: {:ok, t()} | {:error, term()}
  def read_mmap(path) do
    try do
      {handle, file_size} = NxArm.Native.mmap_open_op(path)
      # Pull just enough head bytes to cover header + metadata +
      # tensor catalogue; the catalogue grows with tensor count, not
      # weight bytes, so 4 MiB is plenty for real models. Cap at
      # actual file size for small files / tests.
      head_size = min(4_194_304, file_size)
      head = NxArm.Native.mmap_slice_op(handle, 0, head_size)

      case parse(head) do
        {:ok, %__MODULE__{} = parsed} ->
          {:ok, %{parsed | mmap_handle: handle, file_contents: nil}}

        err ->
          err
      end
    rescue
      e -> {:error, e}
    catch
      :error, reason -> {:error, reason}
    end
  end

  @doc "Parse already-loaded GGUF bytes."
  @spec parse(binary()) :: {:ok, t()} | {:error, term()}
  def parse(<<@magic, version::little-32, tensor_count::little-64, meta_count::little-64,
              rest::binary>> = bin) do
    with {:ok, metadata, rest1} <- read_metadata(rest, meta_count, %{}),
         {:ok, tensors, rest2} <- read_tensor_infos(rest1, tensor_count, []) do
      alignment = Map.get(metadata, "general.alignment", 32)
      consumed = byte_size(bin) - byte_size(rest2)
      pad = rem(alignment - rem(consumed, alignment), alignment)
      data_offset = consumed + pad

      tensors_with_size =
        tensors
        |> Enum.map(&Map.put(&1, :byte_size, tensor_byte_size(&1)))
        |> Map.new(fn t -> {t.name, t} end)

      {:ok,
       %__MODULE__{
         version: version,
         metadata: metadata,
         tensors: tensors_with_size,
         data_offset: data_offset,
         file_contents: bin
       }}
    end
  end

  def parse(_), do: {:error, :not_a_gguf_file}

  @doc """
  Return the raw byte slice for a tensor. For F32/F16/BF16/I*/F64 this
  is just the packed values; for quantized formats (Q4_0 etc.) it is
  the GGML block layout (see gguf.md for the format of each block).
  """
  @spec tensor_bytes(t(), String.t()) :: {:ok, binary()} | {:error, :not_found}
  def tensor_bytes(%__MODULE__{tensors: ts, data_offset: base} = gguf, name) do
    case Map.fetch(ts, name) do
      {:ok, %{offset: off, byte_size: sz}} ->
        bytes =
          cond do
            gguf.mmap_handle != nil ->
              NxArm.Native.mmap_slice_op(gguf.mmap_handle, base + off, sz)

            gguf.file_contents != nil ->
              binary_part(gguf.file_contents, base + off, sz)

            true ->
              <<>>
          end

        {:ok, bytes}

      :error ->
        {:error, :not_found}
    end
  end

  @doc """
  For a Q4_0 tensor, split the GGML block bytes into the `{packed,
  scales}` pair our `int4_matmul_f32_op` expects.

  GGML Q4_0 block layout (one block = 32 weights):
    * `d`: f16 scale (2 bytes)
    * `qs`: 16 packed bytes (32 nibbles)

  We unpack to:
    * `scales`: `[N, K/32]` f32 LE
    * `packed`: `[N, K/2]` u8

  The shape is the tensor's ggml shape, with K being the last
  dimension and N the product of the leading dimensions.
  """
  @spec q4_0_unpack(t(), String.t()) ::
          {:ok, %{packed: binary(), scales: binary(), n: pos_integer(), k: pos_integer()}}
          | {:error, term()}
  def q4_0_unpack(gguf, name) do
    with {:ok, info} <- Map.fetch(gguf.tensors, name) |> ok_or(:not_found),
         :q4_0 <- info.dtype,
         {:ok, raw} <- tensor_bytes(gguf, name) do
      # ggml row-major: dim 0 is the contraction axis (K), the rest
      # are output dimensions.
      [k | rest_dims] = info.shape
      n = Enum.reduce(rest_dims, 1, &(&1 * &2))

      if rem(k, 32) != 0 do
        {:error, {:bad_q4_0_shape, info.shape}}
      else
        n_groups = div(k, 32)
        {packed, scales} = split_q4_0(raw, n, n_groups)
        {:ok, %{packed: packed, scales: scales, n: n, k: k}}
      end
    else
      dtype when is_atom(dtype) -> {:error, {:not_q4_0, dtype}}
      err -> err
    end
  end

  # --------------------------------------------------------------
  # Internal: metadata parsing.
  # --------------------------------------------------------------

  defp read_metadata(rest, 0, acc), do: {:ok, acc, rest}

  defp read_metadata(rest, n, acc) when n > 0 do
    with {:ok, key, rest1} <- read_string(rest),
         <<vtype::little-32, rest2::binary>> <- rest1,
         {:ok, val, rest3} <- read_value(vtype, rest2) do
      read_metadata(rest3, n - 1, Map.put(acc, key, val))
    else
      err -> {:error, {:bad_metadata, err}}
    end
  end

  defp read_value(@vt_u8, <<v::little-8, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_i8, <<v::little-signed-8, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_u16, <<v::little-16, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_i16, <<v::little-signed-16, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_u32, <<v::little-32, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_i32, <<v::little-signed-32, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_f32, <<v::float-32-little, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_bool, <<v::little-8, rest::binary>>), do: {:ok, v != 0, rest}
  defp read_value(@vt_u64, <<v::little-64, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_i64, <<v::little-signed-64, rest::binary>>), do: {:ok, v, rest}
  defp read_value(@vt_f64, <<v::float-64-little, rest::binary>>), do: {:ok, v, rest}

  defp read_value(@vt_string, bin) do
    case read_string(bin) do
      {:ok, s, rest} -> {:ok, s, rest}
      err -> err
    end
  end

  defp read_value(@vt_array, <<inner_vt::little-32, n::little-64, rest::binary>>) do
    Enum.reduce_while(1..n//1, {:ok, [], rest}, fn _, {:ok, acc, r} ->
      case read_value(inner_vt, r) do
        {:ok, v, r2} -> {:cont, {:ok, [v | acc], r2}}
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, vs, r} -> {:ok, Enum.reverse(vs), r}
      err -> err
    end
  end

  defp read_value(@vt_array, <<inner_vt::little-32, 0::little-64, rest::binary>>) do
    _ = inner_vt
    {:ok, [], rest}
  end

  defp read_value(t, _), do: {:error, {:bad_value_type, t}}

  defp read_string(<<len::little-64, str::binary-size(len), rest::binary>>) do
    {:ok, str, rest}
  end

  defp read_string(_), do: {:error, :short_string}

  # --------------------------------------------------------------
  # Internal: tensor info parsing.
  # --------------------------------------------------------------

  defp read_tensor_infos(rest, 0, acc), do: {:ok, Enum.reverse(acc), rest}

  defp read_tensor_infos(rest, n, acc) when n > 0 do
    with {:ok, name, rest1} <- read_string(rest),
         <<n_dims::little-32, rest2::binary>> <- rest1,
         {:ok, dims, rest3} <- read_dims(rest2, n_dims, []),
         <<ttype::little-32, offset::little-64, rest4::binary>> <- rest3 do
      dtype = Map.get(@tensor_types, ttype, {:unknown, ttype})

      info = %{
        name: name,
        dtype: dtype,
        shape: Enum.reverse(dims),
        offset: offset
      }

      read_tensor_infos(rest4, n - 1, [info | acc])
    else
      err -> {:error, {:bad_tensor_info, err}}
    end
  end

  defp read_dims(rest, 0, acc), do: {:ok, acc, rest}

  defp read_dims(<<d::little-64, rest::binary>>, n, acc) when n > 0 do
    read_dims(rest, n - 1, [d | acc])
  end

  # --------------------------------------------------------------
  # Internal: dtype size helpers.
  # --------------------------------------------------------------

  defp tensor_byte_size(%{dtype: dtype, shape: shape}) do
    n_elems = Enum.reduce(shape, 1, &(&1 * &2))

    case dtype do
      :f32 -> n_elems * 4
      :f64 -> n_elems * 8
      :f16 -> n_elems * 2
      :bf16 -> n_elems * 2
      :i8 -> n_elems
      :i16 -> n_elems * 2
      :i32 -> n_elems * 4
      :i64 -> n_elems * 8
      # Q4_0: 32 weights / block, block = 2 bytes scale + 16 bytes packed = 18 bytes.
      :q4_0 -> div(n_elems, 32) * 18
      # Q4_1: 32 weights / block, block = 2 bytes scale + 2 bytes min + 16 bytes packed = 20 bytes.
      :q4_1 -> div(n_elems, 32) * 20
      # Q5_0: 32 weights / block, block = 22 bytes (2 + 4 + 16).
      :q5_0 -> div(n_elems, 32) * 22
      # Q5_1: 32 weights / block, block = 24 bytes (2 + 2 + 4 + 16).
      :q5_1 -> div(n_elems, 32) * 24
      # Q8_0: 32 weights / block, block = 2 + 32 = 34 bytes.
      :q8_0 -> div(n_elems, 32) * 34
      # Q8_1: 32 weights / block, block = 2 + 2 + 32 = 36 bytes.
      :q8_1 -> div(n_elems, 32) * 36
      # K-quants: 256 weights per super-block.
      :q2_k -> div(n_elems, 256) * 84
      :q3_k -> div(n_elems, 256) * 110
      :q4_k -> div(n_elems, 256) * 144
      :q5_k -> div(n_elems, 256) * 176
      :q6_k -> div(n_elems, 256) * 210
      :q8_k -> div(n_elems, 256) * 292
      {:unknown, _} -> 0
    end
  end

  # --------------------------------------------------------------
  # Internal: Q4_0 block unpacking.
  # --------------------------------------------------------------

  defp split_q4_0(raw, n_rows, n_groups) do
    # Each block is 18 bytes: <<scale_f16::2, qs::16>>.
    # We accumulate scales as f32 and packed bytes verbatim.
    do_split_rows(raw, n_rows, n_groups, [], [])
  end

  defp do_split_rows(_rest, 0, _ng, packed_acc, scales_acc) do
    {IO.iodata_to_binary(Enum.reverse(packed_acc)),
     IO.iodata_to_binary(Enum.reverse(scales_acc))}
  end

  defp do_split_rows(rest, rows_left, n_groups, packed_acc, scales_acc) do
    {row_packed, row_scales, rest1} = do_split_groups(rest, n_groups, [], [])

    do_split_rows(
      rest1,
      rows_left - 1,
      n_groups,
      [row_packed | packed_acc],
      [row_scales | scales_acc]
    )
  end

  defp do_split_groups(rest, 0, packed_acc, scales_acc) do
    {Enum.reverse(packed_acc), Enum.reverse(scales_acc), rest}
  end

  defp do_split_groups(<<scale_f16::little-16, packed::binary-size(16), rest::binary>>, g, p_acc, s_acc) do
    scale_f32 = f16_to_f32(scale_f16)
    do_split_groups(rest, g - 1, [packed | p_acc], [<<scale_f32::float-32-little>> | s_acc])
  end

  # IEEE 754 binary16 → binary32. Standard layout: 1 sign / 5 exp / 10
  # mantissa. Subnormals become normalised f32, NaN/Inf propagated.
  defp f16_to_f32(h) do
    s = Bitwise.bsr(h, 15) |> Bitwise.band(0x1)
    e = Bitwise.bsr(h, 10) |> Bitwise.band(0x1F)
    m = Bitwise.band(h, 0x3FF)

    cond do
      e == 0 and m == 0 ->
        if s == 1, do: -0.0, else: 0.0

      e == 0 ->
        # Subnormal: 2^-14 · (m / 1024) · (-1)^s.
        sign = if s == 1, do: -1.0, else: 1.0
        sign * :math.pow(2, -14) * m / 1024.0

      e == 31 ->
        cond do
          m == 0 and s == 0 -> :infinity
          m == 0 and s == 1 -> :negative_infinity
          true -> :nan
        end

      true ->
        sign = if s == 1, do: -1.0, else: 1.0
        sign * :math.pow(2, e - 15) * (1.0 + m / 1024.0)
    end
  end

  defp ok_or({:ok, _} = ok, _), do: ok
  defp ok_or(:error, reason), do: {:error, reason}

  @doc """
  Unpack a Q8_0 tensor into `{i8_weights_bin, scales_bin}` where the
  i8 weights are `[N, K]` row-major (signed bytes) and scales are
  `[N, K/32]` f32 LE — one scale per 32-weight group.

  GGML Q8_0 block layout (one block = 32 weights):
    * `d`: f16 scale (2 bytes)
    * `qs`: 32 int8 weights (32 bytes)
  Block size = 34 bytes.
  """
  @spec q8_0_unpack(t(), String.t()) ::
          {:ok, %{weights: binary(), scales: binary(), n: pos_integer(), k: pos_integer()}}
          | {:error, term()}
  def q8_0_unpack(gguf, name) do
    with {:ok, info} <- Map.fetch(gguf.tensors, name) |> ok_or(:not_found),
         :q8_0 <- info.dtype,
         {:ok, raw} <- tensor_bytes(gguf, name) do
      [k | rest_dims] = info.shape
      n = Enum.reduce(rest_dims, 1, &(&1 * &2))

      if rem(k, 32) != 0 do
        {:error, {:bad_q8_0_shape, info.shape}}
      else
        n_groups = div(k, 32)
        {weights, scales} = split_q8_0(raw, n, n_groups)
        {:ok, %{weights: weights, scales: scales, n: n, k: k}}
      end
    else
      dtype when is_atom(dtype) -> {:error, {:not_q8_0, dtype}}
      err -> err
    end
  end

  defp split_q8_0(raw, n_rows, n_groups) do
    do_split_q8_rows(raw, n_rows, n_groups, [], [])
  end

  defp do_split_q8_rows(_rest, 0, _ng, w_acc, s_acc) do
    {IO.iodata_to_binary(Enum.reverse(w_acc)),
     IO.iodata_to_binary(Enum.reverse(s_acc))}
  end

  defp do_split_q8_rows(rest, rows_left, n_groups, w_acc, s_acc) do
    {row_w, row_s, rest1} = do_split_q8_groups(rest, n_groups, [], [])

    do_split_q8_rows(
      rest1,
      rows_left - 1,
      n_groups,
      [row_w | w_acc],
      [row_s | s_acc]
    )
  end

  defp do_split_q8_groups(rest, 0, w_acc, s_acc) do
    {Enum.reverse(w_acc), Enum.reverse(s_acc), rest}
  end

  defp do_split_q8_groups(<<scale_f16::little-16, weights::binary-size(32), rest::binary>>, g, w_acc, s_acc) do
    scale_f32 = f16_to_f32(scale_f16)
    do_split_q8_groups(rest, g - 1, [weights | w_acc], [<<scale_f32::float-32-little>> | s_acc])
  end

  @doc """
  Dequantize one Q4_0 row to f32. Used for embedding lookups when
  the token_embd is Q4_0 (TinyLlama 1.1B Chat ships this way).
  """
  @spec q4_0_dequant_row(%{packed: binary(), scales: binary(), n: pos_integer(), k: pos_integer()}, non_neg_integer()) :: binary()
  def q4_0_dequant_row(%{packed: p, scales: s, k: k}, row) do
    n_groups = div(k, 32)
    bytes_per_row = div(k, 2)
    row_p = binary_part(p, row * bytes_per_row, bytes_per_row)
    row_s = binary_part(s, row * n_groups * 4, n_groups * 4)

    for g <- 0..(n_groups - 1), into: <<>> do
      group_p = binary_part(row_p, g * 16, 16)
      <<scale::float-32-little>> = binary_part(row_s, g * 4, 4)

      for <<byte::8 <- group_p>>, into: <<>> do
        lo = Bitwise.band(byte, 0x0F) - 8
        hi = Bitwise.band(Bitwise.bsr(byte, 4), 0x0F) - 8
        <<lo * scale::float-32-little, hi * scale::float-32-little>>
      end
    end
  end

  @doc """
  Dequantize one Q8_0 row to f32. Useful for embedding lookups
  where we need just a single row.
  """
  @spec q8_0_dequant_row(%{weights: binary(), scales: binary(), n: pos_integer(), k: pos_integer()}, non_neg_integer()) :: binary()
  def q8_0_dequant_row(%{weights: w, scales: s, k: k}, row) do
    n_groups = div(k, 32)
    row_w = binary_part(w, row * k, k)
    row_s = binary_part(s, row * n_groups * 4, n_groups * 4)

    for g <- 0..(n_groups - 1), into: <<>> do
      group_w = binary_part(row_w, g * 32, 32)
      <<scale::float-32-little>> = binary_part(row_s, g * 4, 4)

      for <<v::signed-8 <- group_w>>, into: <<>> do
        <<v * scale::float-32-little>>
      end
    end
  end
end
