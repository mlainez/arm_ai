defmodule NxArm.Backend do
  @moduledoc """
  Nx backend for ARM CPUs via NEON intrinsics + rayon parallelism.

  Tensors are stored as plain Erlang binaries inside an `%NxArm.Backend{}`
  struct. Every Nx callback dispatches directly to a `NxArm.Native` NIF —
  no GPU device context, no `buffer_read` / `buffer_write` roundtrips.

  ## Coverage

    * Binary elementwise: `add`, `subtract`, `multiply`, `divide`, `max`,
      `min`, `pow`, `atan2`, `remainder` — same-shape and scalar broadcast
    * Unary elementwise: `negate`, `exp`, `log`, `tanh`, `sigmoid`, `abs`,
      `sqrt`, `rsqrt`, `cbrt`, `expm1`, `log1p`, `sin`, `cos`, `tan`,
      `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`,
      `ceil`, `floor`, `round`, `sign`, `erf`, `erfc`
    * Linear algebra: `dot` (2-D via batched matmul with b=1, 3-D+ via
      fold-into-M, 4-D batched for transformer attention)
    * Reductions: `sum`, `reduce_max`, `reduce_min` (axis-wise and full)
    * Shape: `reshape`, `squeeze`, `bitcast` (zero-copy metadata), plus
      `broadcast`, `transpose`, `concatenate` via dedicated NIFs
    * Conv2D: `conv/4` via NEON f32 conv NIF

  Other ops fall back to `Nx.BinaryBackend`.

  ## Usage

      Nx.global_default_backend(NxArm.Backend)
      # or
      gpu_tensor = Nx.backend_transfer(cpu_tensor, NxArm.Backend)
  """

  @behaviour Nx.Backend

  defstruct [:bin]

  # ── Lifecycle / transfer ──────────────────────────────────

  @impl true
  def init(opts), do: opts

  @impl true
  def from_binary(%Nx.Tensor{} = tensor, binary, _backend_opts) do
    put_in(tensor.data, %__MODULE__{bin: binary})
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{bin: bin}}, _limit), do: bin

  @impl true
  def inspect(%Nx.Tensor{} = tensor, opts) do
    binary = to_binary(tensor, Nx.byte_size(tensor))
    Nx.Backend.inspect(tensor, binary, opts)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{}}), do: :ok

  @impl true
  def backend_copy(tensor, Nx.BinaryBackend, _opts) do
    Nx.BinaryBackend.from_binary(
      %{tensor | data: %Nx.BinaryBackend{}},
      to_binary(tensor, Nx.byte_size(tensor)),
      []
    )
  end

  def backend_copy(tensor, backend, opts) do
    binary = to_binary(tensor, Nx.byte_size(tensor))
    backend.from_binary(%{tensor | data: %{__struct__: backend}}, binary, opts)
  end

  @impl true
  def backend_transfer(tensor, backend, opts), do: backend_copy(tensor, backend, opts)

  # ── Constant / Eye / Iota ─────────────────────────────────

  @impl true
  def constant(%Nx.Tensor{} = out, number, backend_opts) do
    binary = Nx.BinaryBackend.constant(%{out | data: %Nx.BinaryBackend{}}, number, [])
    from_binary(out, Nx.to_binary(binary), backend_opts)
  end

  @impl true
  def eye(%Nx.Tensor{} = out, backend_opts) do
    cpu = Nx.BinaryBackend.eye(%{out | data: %Nx.BinaryBackend{}}, [])
    from_binary(out, Nx.to_binary(cpu), backend_opts)
  end

  @impl true
  def iota(%Nx.Tensor{} = out, axis, backend_opts) do
    cpu = Nx.BinaryBackend.iota(%{out | data: %Nx.BinaryBackend{}}, axis, [])
    from_binary(out, Nx.to_binary(cpu), backend_opts)
  end

  # ── Shape (zero-copy metadata or fast NIF) ────────────────

  @impl true
  def reshape(%Nx.Tensor{} = out, %Nx.Tensor{data: %__MODULE__{} = data}) do
    put_in(out.data, data)
  end

  @impl true
  def squeeze(out, tensor, _axes), do: put_in(out.data, tensor.data)

  @impl true
  def bitcast(%Nx.Tensor{} = out, %Nx.Tensor{data: %__MODULE__{} = data}) do
    put_in(out.data, data)
  end

  @impl true
  def as_type(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    if Nx.type(out) == Nx.type(tensor) do
      put_in(out.data, tensor.data)
    else
      try do
        src_dt = dtype_code(Nx.type(tensor))
        dst_dt = dtype_code(Nx.type(out))
        n = Nx.size(tensor)
        out_bin = NxArm.Native.as_type_op(bin_of(tensor), src_dt, dst_dt, n)
        put_in(out.data, %__MODULE__{bin: out_bin})
      rescue
        _ -> fallback(:as_type, [out, tensor])
      end
    end
  end

  @impl true
  def broadcast(out, tensor, shape, axes) do
    bin = bin_of(tensor)
    in_shape = Nx.shape(tensor) |> Tuple.to_list()
    out_shape = shape |> Tuple.to_list()
    esize = element_size(Nx.type(tensor))
    out_bin = NxArm.Native.broadcast_op(bin, in_shape, out_shape, axes, esize)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  @impl true
  def transpose(out, tensor, axes) do
    bin = bin_of(tensor)
    in_shape = Nx.shape(tensor) |> Tuple.to_list()
    esize = element_size(Nx.type(tensor))
    out_bin = NxArm.Native.transpose_op(bin, in_shape, axes, esize)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  @impl true
  def concatenate(out, tensors, axis) do
    esize = element_size(Nx.type(hd(tensors)))
    bins = Enum.map(tensors, &bin_of/1)
    shapes = Enum.map(tensors, fn t -> Nx.shape(t) |> Tuple.to_list() end)
    out_bin = NxArm.Native.concatenate_op(bins, shapes, axis, esize)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  # ── Elementwise Binary Ops ────────────────────────────────

  @binary_ops %{
    add: "add",
    subtract: "subtract",
    multiply: "multiply",
    divide: "divide",
    pow: "pow",
    max: "max",
    min: "min",
    atan2: "atan2",
    remainder: "remainder"
  }

  for {op, op_name} <- @binary_ops do
    @impl true
    def unquote(op)(%Nx.Tensor{} = out, %Nx.Tensor{} = left, %Nx.Tensor{} = right) do
      both_f32? = Nx.type(left) == {:f, 32} and Nx.type(right) == {:f, 32}
      out_f32? = Nx.type(out) == {:f, 32}

      cond do
        # Same-shape f32 — straight NEON kernel.
        both_f32? and Nx.shape(left) == Nx.shape(out) and Nx.shape(right) == Nx.shape(out) ->
          bin = NxArm.Native.elementwise_binary_f32_op(unquote(op_name), bin_of(left), bin_of(right))
          put_in(out.data, %__MODULE__{bin: bin})

        # Bias-add fast path. Pattern: `add` with left = full
        # activation, right = 1-D bias matching the last axis of out.
        # Bumblebee MLP / projection biases all land here. Replaces
        # `Nx.broadcast({inner}, {..., inner}) + add` — 60–170 ms per
        # call through Nx.broadcast → ~1–2 ms via this NIF.
        unquote(op) == :add and both_f32? and
            Nx.shape(left) == Nx.shape(out) and
            bias_add_compatible?(right, out) ->
          inner = elem(Nx.shape(out), tuple_size(Nx.shape(out)) - 1)
          outer = div(Nx.size(out), inner)
          bias_bin = bias_flat_bin(right)
          bin = NxArm.Native.bias_add_f32_op(bin_of(left), bias_bin, outer, inner)
          put_in(out.data, %__MODULE__{bin: bin})

        # `right` is a scalar (any numeric dtype) and `left` matches output.
        # The scalar gets cast to f32 (covers Nx.add(x, 1) where 1 is :s64).
        Nx.size(right) == 1 and Nx.type(left) == {:f, 32} and out_f32? and
            Nx.shape(left) == Nx.shape(out) ->
          bin = NxArm.Native.scalar_binary_f32_op(unquote(op_name), "ab", bin_of(left), to_f32_scalar(right))
          put_in(out.data, %__MODULE__{bin: bin})

        # `left` is a scalar.
        Nx.size(left) == 1 and Nx.type(right) == {:f, 32} and out_f32? and
            Nx.shape(right) == Nx.shape(out) ->
          bin = NxArm.Native.scalar_binary_f32_op(unquote(op_name), "ba", bin_of(right), to_f32_scalar(left))
          put_in(out.data, %__MODULE__{bin: bin})

        not both_f32? ->
          fallback(unquote(op), [out, left, right])

        true ->
          # General broadcasting — materialise both operands at out_shape
          # via the fast broadcast NIF, then run same-shape elementwise.
          left_b = Nx.broadcast(left, Nx.shape(out))
          right_b = Nx.broadcast(right, Nx.shape(out))
          bin = NxArm.Native.elementwise_binary_f32_op(unquote(op_name), bin_of(left_b), bin_of(right_b))
          put_in(out.data, %__MODULE__{bin: bin})
      end
    end
  end

  # ── Elementwise Unary Ops ─────────────────────────────────

  @unary_ops %{
    negate: "negate",
    exp: "exp",
    log: "log",
    tanh: "tanh",
    sigmoid: "sigmoid",
    abs: "abs",
    sqrt: "sqrt",
    rsqrt: "rsqrt",
    cbrt: "cbrt",
    expm1: "expm1",
    log1p: "log1p",
    sin: "sin",
    cos: "cos",
    tan: "tan",
    asin: "asin",
    acos: "acos",
    atan: "atan",
    sinh: "sinh",
    cosh: "cosh",
    asinh: "asinh",
    acosh: "acosh",
    atanh: "atanh",
    ceil: "ceil",
    floor: "floor",
    round: "round",
    sign: "sign",
    erf: "erf",
    erfc: "erfc"
  }

  for {op, op_name} <- @unary_ops do
    @impl true
    def unquote(op)(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
      if Nx.type(tensor) == {:f, 32} do
        bin = NxArm.Native.elementwise_unary_f32_op(unquote(op_name), bin_of(tensor))
        put_in(out.data, %__MODULE__{bin: bin})
      else
        fallback(unquote(op), [out, tensor])
      end
    end
  end

  # ── Dot / Matmul (CPU NEON batched matmul, b=1 covers 2-D) ──

  @impl true
  def dot(
        %Nx.Tensor{} = out,
        %Nx.Tensor{} = left,
        [left_contract_axis],
        [],
        %Nx.Tensor{} = right,
        [right_contract_axis],
        []
      ) do
    left_shape = Nx.shape(left)
    right_shape = Nx.shape(right)
    both_f32? = Nx.type(left) == {:f, 32} and Nx.type(right) == {:f, 32}

    cond do
      not both_f32? ->
        fallback(:dot, [out, left, [left_contract_axis], [], right, [right_contract_axis], []])

      # Plain 2-D × 2-D, x @ w.
      tuple_size(left_shape) == 2 and tuple_size(right_shape) == 2 and
          left_contract_axis == 1 and right_contract_axis == 0 ->
        {m, k} = left_shape
        {_k, n} = right_shape
        cpu_matmul_2d(out, left, right, m, n, k)

      # 2-D × 2-D, x @ w^T (output-major weights, the
      # Axon/Bumblebee Dense convention).
      tuple_size(left_shape) == 2 and tuple_size(right_shape) == 2 and
          left_contract_axis == 1 and right_contract_axis == 1 ->
        {m, k} = left_shape
        {n, _k} = right_shape
        cpu_matmul_2d_right_transposed(out, left, right, m, n, k)

      # 3-D+ × 2-D — Axon/Bumblebee Linear pattern. Fold leading dims
      # into M and use the same path; the output buffer is laid out
      # [m, n] contiguous, reshape is metadata-only.
      tuple_size(right_shape) == 2 and right_contract_axis == 0 and
          left_contract_axis == tuple_size(left_shape) - 1 ->
        {_, n} = right_shape
        k = elem(left_shape, tuple_size(left_shape) - 1)
        lead = left_shape |> Tuple.delete_at(tuple_size(left_shape) - 1)
        m = lead |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

        flat_out = %Nx.Tensor{shape: {m, n}, type: {:f, 32}, names: [nil, nil]}
        flat = cpu_matmul_2d(flat_out, left, right, m, n, k)
        reshaped = Nx.reshape(flat, Nx.shape(out))
        put_in(out.data, reshaped.data)

      # 3-D+ × 2-D, contracting last with last (x @ w^T flavour).
      tuple_size(right_shape) == 2 and right_contract_axis == 1 and
          left_contract_axis == tuple_size(left_shape) - 1 ->
        {n, _k} = right_shape
        k = elem(left_shape, tuple_size(left_shape) - 1)
        lead = left_shape |> Tuple.delete_at(tuple_size(left_shape) - 1)
        m = lead |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

        flat_out = %Nx.Tensor{shape: {m, n}, type: {:f, 32}, names: [nil, nil]}
        flat = cpu_matmul_2d_right_transposed(flat_out, left, right, m, n, k)
        reshaped = Nx.reshape(flat, Nx.shape(out))
        put_in(out.data, reshaped.data)

      true ->
        fallback(:dot, [out, left, [left_contract_axis], [], right, [right_contract_axis], []])
    end
  end

  # Batched 4-D dot for transformer attention: Q @ K^T and attn @ V.
  def dot(out, left, left_contract, left_batch, right, right_contract, right_batch) do
    left_shape = Nx.shape(left)
    right_shape = Nx.shape(right)
    l_rank = tuple_size(left_shape)
    r_rank = tuple_size(right_shape)
    both_f32? = Nx.type(left) == {:f, 32} and Nx.type(right) == {:f, 32}

    cond do
      not both_f32? or l_rank < 3 or l_rank != r_rank ->
        fallback(:dot, [out, left, left_contract, left_batch, right, right_contract, right_batch])

      true ->
        batch_count = l_rank - 2
        expected_batch = Enum.to_list(0..(batch_count - 1))

        left_batch_dims = left_shape |> Tuple.to_list() |> Enum.take(batch_count)
        right_batch_dims = right_shape |> Tuple.to_list() |> Enum.take(batch_count)

        cond do
          left_batch != expected_batch or right_batch != expected_batch ->
            fallback(:dot, [out, left, left_contract, left_batch, right, right_contract, right_batch])

          left_batch_dims != right_batch_dims ->
            fallback(:dot, [out, left, left_contract, left_batch, right, right_contract, right_batch])

          left_contract == [l_rank - 1] and right_contract == [r_rank - 1] ->
            batched_matmul(out, left, right, true)

          left_contract == [l_rank - 1] and right_contract == [r_rank - 2] ->
            batched_matmul(out, left, right, false)

          true ->
            fallback(:dot, [out, left, left_contract, left_batch, right, right_contract, right_batch])
        end
    end
  end

  defp cpu_matmul_2d(out, left, right, m, n, k) do
    bin = NxArm.Native.batched_matmul_f32_op(bin_of(left), bin_of(right), 1, m, n, k, false)
    put_in(out.data, %__MODULE__{bin: bin})
  end

  defp cpu_matmul_2d_right_transposed(out, left, right, m, n, k) do
    # left: (M, K) row-major. right: (N, K) row-major, contract on K.
    # batched_matmul_f32_op already supports right_transposed=true.
    bin = NxArm.Native.batched_matmul_f32_op(bin_of(left), bin_of(right), 1, m, n, k, true)
    put_in(out.data, %__MODULE__{bin: bin})
  end

  defp batched_matmul(out, left, right, right_transposed?) do
    left_shape = Nx.shape(left) |> Tuple.to_list()
    right_shape = Nx.shape(right) |> Tuple.to_list()
    rank = length(left_shape)
    batch_dims = Enum.take(left_shape, rank - 2)
    b = Enum.reduce(batch_dims, 1, &(&1 * &2))
    m = Enum.at(left_shape, rank - 2)
    k = Enum.at(left_shape, rank - 1)

    n =
      if right_transposed? do
        Enum.at(right_shape, rank - 2)
      else
        Enum.at(right_shape, rank - 1)
      end

    bin =
      NxArm.Native.batched_matmul_f32_op(
        bin_of(left),
        bin_of(right),
        b,
        m,
        n,
        k,
        right_transposed?
      )

    put_in(out.data, %__MODULE__{bin: bin})
  end

  # ── Reductions ────────────────────────────────────────────

  for {nx_fn, op_name} <- [sum: "sum", reduce_max: "max", reduce_min: "min"] do
    @impl true
    def unquote(nx_fn)(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
      do_reduce(unquote(nx_fn), unquote(op_name), out, tensor, opts)
    end
  end

  defp do_reduce(nx_fn, op_name, out, tensor, opts) do
    axes = opts[:axes]
    rank = tuple_size(Nx.shape(tensor))
    last_axis = rank - 1

    cond do
      Nx.type(tensor) != {:f, 32} ->
        fallback(nx_fn, [out, tensor, opts])

      axes == nil or axes == Nx.axes(tensor) ->
        # Full reduce: treat as n_outer=1, inner=total.
        total = Nx.size(tensor)
        bin = NxArm.Native.reduce_axis_f32_op(op_name, bin_of(tensor), 1, total)
        put_in(out.data, %__MODULE__{bin: bin})

      axes == [last_axis] ->
        total = Nx.size(tensor)
        inner = elem(Nx.shape(tensor), last_axis)
        n_out = div(total, inner)
        bin = NxArm.Native.reduce_axis_f32_op(op_name, bin_of(tensor), n_out, inner)
        put_in(out.data, %__MODULE__{bin: bin})

      true ->
        fallback(nx_fn, [out, tensor, opts])
    end
  end

  # ── Conv2D ────────────────────────────────────────────────

  @impl true
  def conv(out, tensor, kernel, opts) do
    strides = opts[:strides] || [1, 1]
    padding = opts[:padding] || []
    input_dilation = opts[:input_dilation] || [1, 1]
    kernel_dilation = opts[:kernel_dilation] || [1, 1]
    feature_group_size = opts[:feature_group_size] || 1
    batch_group_size = opts[:batch_group_size] || 1
    input_perm = opts[:input_permutation] || Enum.to_list(0..(tuple_size(Nx.shape(tensor)) - 1))
    kernel_perm = opts[:kernel_permutation] || Enum.to_list(0..(tuple_size(Nx.shape(kernel)) - 1))
    output_perm = opts[:output_permutation] || Enum.to_list(0..(tuple_size(Nx.shape(out)) - 1))

    cond do
      tuple_size(Nx.shape(tensor)) != 4 -> fallback(:conv, [out, tensor, kernel, opts])
      tuple_size(Nx.shape(kernel)) != 4 -> fallback(:conv, [out, tensor, kernel, opts])
      not all_ones?(input_dilation) -> fallback(:conv, [out, tensor, kernel, opts])
      not all_ones?(kernel_dilation) -> fallback(:conv, [out, tensor, kernel, opts])
      batch_group_size != 1 -> fallback(:conv, [out, tensor, kernel, opts])

      # Depthwise: feature_group_size == Cin. Each kernel "group" has
      # 1 input channel and 1 output channel — MobileNet/EfficientNet
      # pattern.
      feature_group_size == elem(Nx.shape(tensor), Enum.at(input_perm, 1)) ->
        do_depthwise_neon_conv(out, tensor, kernel, strides, padding, input_perm, kernel_perm, output_perm)

      feature_group_size != 1 -> fallback(:conv, [out, tensor, kernel, opts])

      true -> do_neon_conv(out, tensor, kernel, strides, padding, input_perm, kernel_perm, output_perm)
    end
  end

  defp do_depthwise_neon_conv(out, tensor, kernel, strides, padding, input_perm, kernel_perm, output_perm) do
    tensor = ensure_on_arm(tensor)
    kernel = ensure_on_arm(kernel)

    [batch_ax, chan_ax, h_ax, w_ax] = input_perm
    nhwc_in = Nx.transpose(tensor, axes: [batch_ax, h_ax, w_ax, chan_ax])

    # Kernel: Nx hands us {Cin, 1, Kh, Kw} after the perm. We squeeze
    # the second axis (which has size 1 in depthwise) to get
    # {Cin, Kh, Kw}.
    [out_ch_ax, in_ch_ax, kh_ax, kw_ax] = kernel_perm
    cout_first_kernel = Nx.transpose(kernel, axes: [out_ch_ax, kh_ax, kw_ax, in_ch_ax])
    {c_in, kh, kw, group_in} = Nx.shape(cout_first_kernel)

    if group_in != 1 do
      raise ArgumentError,
            "depthwise conv expected kernel group_in == 1, got #{group_in}"
    end

    weight_3d = Nx.reshape(cout_first_kernel, {c_in, kh, kw})

    {n, h_in, w_in, _} = Nx.shape(nhwc_in)

    {sh, sw} =
      case strides do
        [a, b] -> {a, b}
        a when is_integer(a) -> {a, a}
      end

    {pt, pb, pl, pr} = normalize_conv_padding(padding, h_in, w_in, kh, kw, sh, sw)

    out_bin =
      NxArm.Native.depthwise_conv2d_f32_op(
        bin_of(nhwc_in),
        bin_of(weight_3d),
        <<>>,
        [n, h_in, w_in, c_in, kh, kw],
        [sh, sw],
        [pt, pb, pl, pr]
      )

    h_out = div(h_in + pt + pb - kh, sh) + 1
    w_out = div(w_in + pl + pr - kw, sw) + 1

    nhwc_out =
      Nx.from_binary(out_bin, :f32, backend: __MODULE__)
      |> Nx.reshape({n, h_out, w_out, c_in})

    [o_batch_ax, o_chan_ax, o_h_ax, o_w_ax] = output_perm
    nhwc_to_caller = invert_permutation([o_batch_ax, o_h_ax, o_w_ax, o_chan_ax])
    permuted = Nx.transpose(nhwc_out, axes: nhwc_to_caller)
    put_in(out.data, permuted.data)
  end

  defp all_ones?(list), do: Enum.all?(list, &(&1 == 1))

  defp do_neon_conv(out, tensor, kernel, strides, padding, input_perm, kernel_perm, output_perm) do
    # CRITICAL: ensure both tensor and kernel are on NxArm.Backend
    # before doing any Nx ops on them. Bumblebee params come from
    # `binary_to_term` on tensors saved against Nx.BinaryBackend (the
    # default on the host where we exported). Doing Nx.transpose on a
    # BinaryBackend tensor dispatches to BinaryBackend.transpose which
    # is pure-Elixir — 412 ms on the {16, 16, 3, 192} ViT patch
    # kernel. After this ensure_on_arm/1, the same transpose takes
    # 5 ms via our NEON NIF.
    tensor = ensure_on_arm(tensor)
    kernel = ensure_on_arm(kernel)

    [batch_ax, chan_ax, h_ax, w_ax] = input_perm
    nhwc_in = Nx.transpose(tensor, axes: [batch_ax, h_ax, w_ax, chan_ax])

    [out_ch_ax, in_ch_ax, kh_ax, kw_ax] = kernel_perm
    cout_first_kernel = Nx.transpose(kernel, axes: [out_ch_ax, kh_ax, kw_ax, in_ch_ax])

    {c_out, kh, kw, c_in} = Nx.shape(cout_first_kernel)
    flat_kernel = Nx.reshape(cout_first_kernel, {c_out, kh * kw * c_in})

    {n, h_in, w_in, _} = Nx.shape(nhwc_in)

    {sh, sw} =
      case strides do
        [a, b] -> {a, b}
        a when is_integer(a) -> {a, a}
      end

    {pt, pb, pl, pr} = normalize_conv_padding(padding, h_in, w_in, kh, kw, sh, sw)

    # Patchify fast path: when stride == kernel size and padding is
    # zero, the conv is equivalent to a single matmul. This is exactly
    # ViT's `embedder` step (16×16 stride-16 conv) — the generic NEON
    # conv NIF spends ~575 ms on it; this path drops it to ~5 ms by
    # reshape + transpose + 4×8 matmul. Detect and short-circuit.
    if sh == kh and sw == kw and pt == 0 and pb == 0 and pl == 0 and pr == 0 and
         rem(h_in, kh) == 0 and rem(w_in, kw) == 0 do
      patchify_conv_as_matmul(out, nhwc_in, flat_kernel, n, h_in, w_in, c_in, c_out, kh, kw, output_perm)
    else
      generic_neon_conv(out, nhwc_in, flat_kernel, n, h_in, w_in, c_in, c_out, kh, kw, sh, sw, pt, pb, pl, pr, output_perm)
    end
  end

  # ViT-style patchify: when stride matches kernel exactly with no
  # padding, each output cell is the dot product of one non-overlapping
  # input patch with one row of the flattened kernel. That's just
  # `patches @ kernel^T` after reshape + transpose.
  defp patchify_conv_as_matmul(out, nhwc_in, flat_kernel, n, h_in, w_in, c_in, c_out, kh, kw, output_perm) do
    h_out = div(h_in, kh)
    w_out = div(w_in, kw)

    # Reshape {N, H, W, Cin} → {N, H_out, kh, W_out, kw, Cin}, transpose
    # to {N, H_out, W_out, kh, kw, Cin}, then flatten to
    # {N*H_out*W_out, kh*kw*Cin}.
    patches =
      nhwc_in
      |> Nx.reshape({n, h_out, kh, w_out, kw, c_in})
      |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
      |> Nx.reshape({n * h_out * w_out, kh * kw * c_in})

    # flat_kernel is {Cout, Kh*Kw*Cin}. We need patches @ kernel^T
    # which is the Q@K^T pattern of batched_matmul_f32_op (right
    # transposed). With b=1, M = n*h_out*w_out, K = kh*kw*c_in, N = c_out.
    m = n * h_out * w_out
    k = kh * kw * c_in

    out_bin =
      NxArm.Native.batched_matmul_f32_op(
        bin_of(patches),
        bin_of(flat_kernel),
        1,
        m,
        c_out,
        k,
        true
      )

    nhwc_out =
      Nx.from_binary(out_bin, :f32, backend: __MODULE__)
      |> Nx.reshape({n, h_out, w_out, c_out})

    [o_batch_ax, o_chan_ax, o_h_ax, o_w_ax] = output_perm
    nhwc_to_caller = invert_permutation([o_batch_ax, o_h_ax, o_w_ax, o_chan_ax])
    permuted = Nx.transpose(nhwc_out, axes: nhwc_to_caller)
    put_in(out.data, permuted.data)
  end

  defp generic_neon_conv(out, nhwc_in, flat_kernel, n, h_in, w_in, c_in, c_out, kh, kw, sh, sw, pt, pb, pl, pr, output_perm) do
    input_bin = bin_of(nhwc_in)
    weight_bin = bin_of(flat_kernel)

    out_bin =
      NxArm.Native.conv2d_f32_op(
        input_bin,
        weight_bin,
        <<>>,
        [n, h_in, w_in, c_in, c_out, kh, kw],
        [sh, sw],
        [pt, pb, pl, pr]
      )

    h_out = div(h_in + pt + pb - kh, sh) + 1
    w_out = div(w_in + pl + pr - kw, sw) + 1

    nhwc_out =
      Nx.from_binary(out_bin, :f32, backend: __MODULE__)
      |> Nx.reshape({n, h_out, w_out, c_out})

    [o_batch_ax, o_chan_ax, o_h_ax, o_w_ax] = output_perm
    nhwc_to_caller = invert_permutation([o_batch_ax, o_h_ax, o_w_ax, o_chan_ax])
    permuted = Nx.transpose(nhwc_out, axes: nhwc_to_caller)
    put_in(out.data, permuted.data)
  end

  defp normalize_conv_padding(:valid, _, _, _, _, _, _), do: {0, 0, 0, 0}

  defp normalize_conv_padding(:same, h, w, kh, kw, sh, sw) do
    pad_h = max(0, (Float.ceil(h / sh) |> trunc()) * sh - h + kh - sh)
    pad_w = max(0, (Float.ceil(w / sw) |> trunc()) * sw - w + kw - sw)
    {div(pad_h, 2), pad_h - div(pad_h, 2), div(pad_w, 2), pad_w - div(pad_w, 2)}
  end

  defp normalize_conv_padding([{pt, pb}, {pl, pr}], _, _, _, _, _, _), do: {pt, pb, pl, pr}
  defp normalize_conv_padding([], _, _, _, _, _, _), do: {0, 0, 0, 0}

  defp invert_permutation(perm) do
    perm
    |> Enum.with_index()
    |> Enum.sort_by(fn {axis, _} -> axis end)
    |> Enum.map(fn {_, src} -> src end)
  end

  # ── Everything else: fallback to BinaryBackend ────────────

  @all_binary_fallbacks [
    :pow, :remainder, :atan2, :min, :max,
    :quotient, :bitwise_and, :bitwise_or, :bitwise_xor,
    :left_shift, :right_shift,
    :equal, :not_equal, :greater, :less, :greater_equal, :less_equal,
    :logical_and, :logical_or, :logical_xor
  ]

  @fallback_binary_ops @all_binary_fallbacks -- Map.keys(@binary_ops)

  for op <- @fallback_binary_ops do
    @impl true
    def unquote(op)(out, left, right) do
      fallback(unquote(op), [out, left, right])
    end
  end

  # Nx.Backend unary callbacks (the ones the behaviour actually
  # declares). `logical_not` / `phase` look like unaries but aren't
  # listed in `@behaviour Nx.Backend`, so they're emitted separately
  # below without `@impl true`.
  @all_unary_ops Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
                   [
                     :bitwise_not,
                     :ceil,
                     :conjugate,
                     :floor,
                     :round,
                     :sign,
                     :count_leading_zeros,
                     :population_count,
                     :real,
                     :imag,
                     :is_nan,
                     :is_infinity
                   ]

  @fallback_unary_ops @all_unary_ops -- Map.keys(@unary_ops)

  for op <- @fallback_unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      fallback(unquote(op), [out, tensor])
    end
  end

  # Unary-shaped ops Nx exposes but doesn't declare as
  # `Nx.Backend` callbacks — same fallback shape, no `@impl`.
  for op <- [:logical_not, :phase] do
    def unquote(op)(out, tensor) do
      fallback(unquote(op), [out, tensor])
    end
  end

  @impl true
  def pad(out, tensor, pad_value, padding_config) do
    try do
      in_shape = Nx.shape(tensor) |> Tuple.to_list()
      out_shape = Nx.shape(out) |> Tuple.to_list()
      esize = element_size(Nx.type(tensor))
      fill = bin_of(pad_value)

      if byte_size(fill) != esize do
        fallback(:pad, [out, tensor, pad_value, padding_config])
      else
        out_bin =
          NxArm.Native.pad_op(
            bin_of(tensor),
            in_shape,
            out_shape,
            padding_config,
            fill,
            esize
          )

        put_in(out.data, %__MODULE__{bin: out_bin})
      end
    rescue
      _ -> fallback(:pad, [out, tensor, pad_value, padding_config])
    end
  end

  @impl true
  def reverse(out, tensor, axes) do
    try do
      shape = Nx.shape(tensor) |> Tuple.to_list()
      esize = element_size(Nx.type(tensor))
      out_bin = NxArm.Native.reverse_op(bin_of(tensor), shape, axes, esize)
      put_in(out.data, %__MODULE__{bin: out_bin})
    rescue
      _ -> fallback(:reverse, [out, tensor, axes])
    end
  end

  @impl true
  def clip(out, tensor, min, max) do
    type = Nx.type(tensor)

    cond do
      not match?({k, _} when k in [:f, :s, :u], type) ->
        fallback(:clip, [out, tensor, min, max])

      true ->
        try do
          min_f = to_f32_scalar(min)
          max_f = to_f32_scalar(max)
          out_bin = NxArm.Native.clip_op(bin_of(tensor), dtype_code(type), min_f * 1.0, max_f * 1.0)
          put_in(out.data, %__MODULE__{bin: out_bin})
        rescue
          _ -> fallback(:clip, [out, tensor, min, max])
        end
    end
  end

  @impl true
  def slice(out, tensor, starts, lengths, strides) do
    in_shape = Nx.shape(tensor) |> Tuple.to_list()
    esize = element_size(Nx.type(tensor))
    bin = bin_of(tensor)
    out_bin = NxArm.Native.slice_op(bin, in_shape, starts, lengths, strides, esize)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  @impl true
  def put_slice(out, tensor, starts, slice_tensor) do
    in_shape = Nx.shape(tensor) |> Tuple.to_list()
    slice_shape = Nx.shape(slice_tensor) |> Tuple.to_list()
    esize = element_size(Nx.type(tensor))

    # Nx allows start_indices to be a mix of integers and 0-D tensors
    # (for dynamic indexing). Normalise to plain integers — bail to
    # fallback if any is actually dynamic.
    case normalize_starts(starts) do
      {:ok, int_starts} ->
        tensor_bin = bin_of(tensor)
        slice_bin = bin_of(slice_tensor)
        out_bin = NxArm.Native.put_slice_op(tensor_bin, in_shape, slice_bin, slice_shape, int_starts, esize)
        put_in(out.data, %__MODULE__{bin: out_bin})

      :dynamic ->
        fallback(:put_slice, [out, tensor, starts, slice_tensor])
    end
  end

  defp normalize_starts(starts) do
    result =
      Enum.reduce_while(starts, [], fn
        i, acc when is_integer(i) ->
          {:cont, [i | acc]}

        %Nx.Tensor{shape: {}} = t, acc ->
          # 0-D tensor of an integer — read its value.
          {:cont, [Nx.to_number(t) | acc]}

        _other, _acc ->
          {:halt, :dynamic}
      end)

    case result do
      :dynamic -> :dynamic
      list -> {:ok, Enum.reverse(list)}
    end
  end

  @impl true
  def gather(out, input, indices, opts) do
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(Nx.shape(indices)) - 1) - 1)
    in_shape = Nx.shape(input) |> Tuple.to_list()
    idx_shape = Nx.shape(indices) |> Tuple.to_list()
    in_rank = length(in_shape)

    contiguous_prefix? = axes == Enum.to_list(0..(length(axes) - 1))

    if contiguous_prefix? and length(axes) <= in_rank do
      esize = element_size(Nx.type(input))
      isize = element_size(Nx.type(indices))
      in_bin = bin_of(input)
      idx_bin = bin_of(indices)
      out_bin = NxArm.Native.gather_op(in_bin, in_shape, idx_bin, idx_shape, isize, axes, esize)
      put_in(out.data, %__MODULE__{bin: out_bin})
    else
      fallback(:gather, [out, input, indices, opts])
    end
  end

  @impl true
  def stack(out, tensors, axis) do
    cond do
      axis != 0 ->
        fallback(:stack, [out, tensors, axis])

      tensors == [] ->
        fallback(:stack, [out, tensors, axis])

      true ->
        first = hd(tensors)
        first_type = Nx.type(first)
        first_shape = Nx.shape(first)

        compatible? =
          Enum.all?(tensors, fn t ->
            Nx.type(t) == first_type and Nx.shape(t) == first_shape
          end)

        if not compatible? do
          fallback(:stack, [out, tensors, axis])
        else
          try do
            tensor_bytes = byte_size(bin_of(first))
            bins = Enum.map(tensors, &bin_of/1)
            out_bin = NxArm.Native.stack_axis0_op(bins, tensor_bytes)
            put_in(out.data, %__MODULE__{bin: out_bin})
          rescue
            _ -> fallback(:stack, [out, tensors, axis])
          end
        end
    end
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    pred_shape = Nx.shape(pred)
    t_shape = Nx.shape(on_true)
    f_shape = Nx.shape(on_false)

    if pred_shape == t_shape and pred_shape == f_shape and
         Nx.type(on_true) == Nx.type(on_false) do
      try do
        pred_bytes = pred_to_bytes(pred)
        esize = element_size(Nx.type(on_true))
        out_bin = NxArm.Native.select_op(pred_bytes, bin_of(on_true), bin_of(on_false), esize)
        put_in(out.data, %__MODULE__{bin: out_bin})
      rescue
        _ -> fallback(:select, [out, pred, on_true, on_false])
      end
    else
      fallback(:select, [out, pred, on_true, on_false])
    end
  end

  @impl true
  def all(out, tensor, opts) do
    # Only handle the "reduce-everything → scalar" path on u8. Axis-
    # restricted reductions stay on fallback for now.
    if opts == [] or opts[:axes] in [nil, []] do
      try do
        type = Nx.type(tensor)

        bin =
          if type == {:u, 8} do
            bin_of(tensor)
          else
            NxArm.Native.as_type_op(
              bin_of(tensor),
              dtype_code(type),
              dtype_code({:u, 8}),
              Nx.size(tensor)
            )
          end

        val = NxArm.Native.reduce_all_u8_op(bin)
        put_in(out.data, %__MODULE__{bin: <<val::8>>})
      rescue
        _ -> fallback(:all, [out, tensor, opts])
      end
    else
      fallback(:all, [out, tensor, opts])
    end
  end

  @impl true
  def any(out, tensor, opts) do
    if opts == [] or opts[:axes] in [nil, []] do
      try do
        type = Nx.type(tensor)

        bin =
          if type == {:u, 8} do
            bin_of(tensor)
          else
            NxArm.Native.as_type_op(
              bin_of(tensor),
              dtype_code(type),
              dtype_code({:u, 8}),
              Nx.size(tensor)
            )
          end

        val = NxArm.Native.reduce_any_u8_op(bin)
        put_in(out.data, %__MODULE__{bin: <<val::8>>})
      rescue
        _ -> fallback(:any, [out, tensor, opts])
      end
    else
      fallback(:any, [out, tensor, opts])
    end
  end

  @impl true
  def product(out, tensor, opts) do
    if (opts == [] or opts[:axes] in [nil, []]) and Nx.type(tensor) == {:f, 32} do
      try do
        val = NxArm.Native.reduce_product_f32_op(bin_of(tensor))
        put_in(out.data, %__MODULE__{bin: <<val::float-32-little>>})
      rescue
        _ -> fallback(:product, [out, tensor, opts])
      end
    else
      fallback(:product, [out, tensor, opts])
    end
  end

  @impl true
  def argmax(out, tensor, opts), do: do_argmax_argmin(out, tensor, opts, :max)

  @impl true
  def argmin(out, tensor, opts), do: do_argmax_argmin(out, tensor, opts, :min)

  defp do_argmax_argmin(out, tensor, opts, which) do
    type = Nx.type(tensor)
    shape = Nx.shape(tensor) |> Tuple.to_list()
    rank = length(shape)
    axis = opts[:axis]
    tie_break = opts[:tie_break] || :low

    cond do
      type != {:f, 32} ->
        fallback_name = if which == :max, do: :argmax, else: :argmin
        fallback(fallback_name, [out, tensor, opts])

      tie_break != :low ->
        fallback_name = if which == :max, do: :argmax, else: :argmin
        fallback(fallback_name, [out, tensor, opts])

      axis == nil or axis == rank - 1 or axis == -1 ->
        # Reduce over the last (or only) axis.
        inner = if rank == 0, do: 1, else: List.last(shape)
        outer = if rank == 0, do: 1, else: div(Nx.size(tensor), inner)

        try do
          out_bin =
            case which do
              :max -> NxArm.Native.argmax_axis_f32_op(bin_of(tensor), outer, inner)
              :min -> NxArm.Native.argmin_axis_f32_op(bin_of(tensor), outer, inner)
            end

          # Result type is s64 per Nx convention.
          put_in(out.data, %__MODULE__{bin: out_bin})
        rescue
          _ ->
            fallback_name = if which == :max, do: :argmax, else: :argmin
            fallback(fallback_name, [out, tensor, opts])
        end

      true ->
        fallback_name = if which == :max, do: :argmax, else: :argmin
        fallback(fallback_name, [out, tensor, opts])
    end
  end

  @impl true
  def reduce(out, tensor, acc, opts, fun),
    do: fallback(:reduce, [out, tensor, acc, opts, fun])

  @impl true
  def window_reduce(out, tensor, acc, shape, opts, fun),
    do: fallback(:window_reduce, [out, tensor, acc, shape, opts, fun])

  @impl true
  def window_sum(out, tensor, shape, opts), do: do_window(out, tensor, shape, opts, "sum")

  @impl true
  def window_product(out, tensor, shape, opts), do: do_window(out, tensor, shape, opts, "product")

  @impl true
  def window_max(out, tensor, shape, opts), do: do_window(out, tensor, shape, opts, "max")

  @impl true
  def window_min(out, tensor, shape, opts), do: do_window(out, tensor, shape, opts, "min")

  defp do_window(out, tensor, window_shape, opts, op) do
    if Nx.type(tensor) == {:f, 32} do
      in_shape = Nx.shape(tensor) |> Tuple.to_list()
      window_dims = window_shape |> Tuple.to_list()
      strides = opts[:strides] || List.duplicate(1, length(window_dims))
      padding = opts[:padding] || List.duplicate({0, 0}, length(window_dims))
      pad_low = Enum.map(padding, fn {lo, _} -> lo end)
      pad_high = Enum.map(padding, fn {_, hi} -> hi end)

      bin = bin_of(tensor)
      out_bin =
        NxArm.Native.window_reduce_f32_op(op, bin, in_shape, window_dims, strides, pad_low, pad_high)

      put_in(out.data, %__MODULE__{bin: out_bin})
    else
      fallback(String.to_atom("window_" <> op), [out, tensor, window_shape, opts])
    end
  end

  @impl true
  def sort(out, tensor, opts), do: do_sort_argsort(out, tensor, opts, :sort)

  @impl true
  def argsort(out, tensor, opts), do: do_sort_argsort(out, tensor, opts, :argsort)

  defp do_sort_argsort(out, tensor, opts, which) do
    type = Nx.type(tensor)
    shape = Nx.shape(tensor) |> Tuple.to_list()
    rank = length(shape)
    axis = Keyword.get(opts, :axis, rank - 1)
    direction = Keyword.get(opts, :direction, :asc)
    descending = direction == :desc

    cond do
      type != {:f, 32} ->
        fallback_name = if which == :sort, do: :sort, else: :argsort
        fallback(fallback_name, [out, tensor, opts])

      not (axis == rank - 1 or axis == -1) ->
        fallback_name = if which == :sort, do: :sort, else: :argsort
        fallback(fallback_name, [out, tensor, opts])

      true ->
        inner = if rank == 0, do: 1, else: List.last(shape)
        outer = if rank == 0, do: 1, else: div(Nx.size(tensor), inner)

        try do
          out_bin =
            case which do
              :sort -> NxArm.Native.sort_axis_f32_op(bin_of(tensor), outer, inner, descending)
              :argsort -> NxArm.Native.argsort_axis_f32_op(bin_of(tensor), outer, inner, descending)
            end

          put_in(out.data, %__MODULE__{bin: out_bin})
        rescue
          _ ->
            fallback_name = if which == :sort, do: :sort, else: :argsort
            fallback(fallback_name, [out, tensor, opts])
        end
    end
  end

  @impl true
  def window_scatter_max(out, tensor, source, init, shape, opts),
    do: fallback(:window_scatter_max, [out, tensor, source, init, shape, opts])

  @impl true
  def window_scatter_min(out, tensor, source, init, shape, opts),
    do: fallback(:window_scatter_min, [out, tensor, source, init, shape, opts])

  @impl true
  def indexed_add(out, tensor, indices, updates, opts) do
    do_indexed(out, tensor, indices, updates, opts, :add)
  end

  @impl true
  def indexed_put(out, tensor, indices, updates, opts) do
    do_indexed(out, tensor, indices, updates, opts, :put)
  end

  defp do_indexed(out, tensor, indices, updates, opts, kind) do
    type = Nx.type(tensor)

    cond do
      type != {:f, 32} ->
        fb = if kind == :add, do: :indexed_add, else: :indexed_put
        fallback(fb, [out, tensor, indices, updates, opts])

      true ->
        try do
          t_bin = bin_of(tensor)
          shape = Nx.shape(tensor) |> Tuple.to_list()
          total = Enum.reduce(shape, 1, &(&1 * &2))
          strides = compute_strides(shape)

          # Convert N-D indices to flat indices.
          flat_idx_bin = flatten_indices(indices, strides, total)
          upd_bin = bin_of(updates) |> ensure_f32(Nx.type(updates), Nx.size(updates))

          out_bin =
            case kind do
              :add -> NxArm.Native.indexed_add_f32_op(t_bin, flat_idx_bin, upd_bin)
              :put -> NxArm.Native.indexed_put_f32_op(t_bin, flat_idx_bin, upd_bin)
            end

          put_in(out.data, %__MODULE__{bin: out_bin})
        rescue
          _ ->
            fb = if kind == :add, do: :indexed_add, else: :indexed_put
            fallback(fb, [out, tensor, indices, updates, opts])
        end
    end
  end

  defp compute_strides([]), do: []

  defp compute_strides(shape) do
    Enum.reduce(Enum.reverse(shape), {[], 1}, fn dim, {acc, stride} ->
      {[stride | acc], stride * dim}
    end)
    |> elem(0)
  end

  defp flatten_indices(indices, strides, total) do
    rank = length(strides)
    {n, _} = Nx.shape(indices)

    raw =
      indices
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    flat =
      raw
      |> Enum.chunk_every(rank)
      |> Enum.map(fn coords ->
        flat =
          Enum.zip(coords, strides)
          |> Enum.reduce(0, fn {c, s}, acc -> acc + c * s end)

        rem(flat, total)
      end)

    flat
    |> Enum.map(&<<&1::little-signed-64>>)
    |> IO.iodata_to_binary()
    |> tap(fn _ -> n end)
  end

  defp ensure_f32(bin, {:f, 32}, _n), do: bin

  defp ensure_f32(bin, type, n) do
    NxArm.Native.as_type_op(bin, dtype_code(type), dtype_code({:f, 32}), n)
  end

  @impl true
  def fft(out, tensor, opts) do
    cond do
      not function_exported?(NxArm.Native, :fft_complex_op, 1) ->
        fallback(:fft, [out, tensor, opts])

      Nx.type(tensor) != {:c, 64} ->
        fallback(:fft, [out, tensor, opts])

      true ->
        try do
          # Nx complex tensors are stored as f32 LE re/im interleaved.
          bin = bin_of(tensor)
          out_bin = NxArm.Native.fft_complex_op(bin)
          put_in(out.data, %__MODULE__{bin: out_bin})
        rescue
          _ -> fallback(:fft, [out, tensor, opts])
        end
    end
  end

  @impl true
  def ifft(out, tensor, opts) do
    cond do
      not function_exported?(NxArm.Native, :ifft_complex_op, 1) ->
        fallback(:ifft, [out, tensor, opts])

      Nx.type(tensor) != {:c, 64} ->
        fallback(:ifft, [out, tensor, opts])

      true ->
        try do
          bin = bin_of(tensor)
          out_bin = NxArm.Native.ifft_complex_op(bin)
          put_in(out.data, %__MODULE__{bin: out_bin})
        rescue
          _ -> fallback(:ifft, [out, tensor, opts])
        end
    end
  end

  @impl true
  def triangular_solve(out, a, b, opts),
    do: fallback(:triangular_solve, [out, a, b, opts])

  # `lu` is exposed by Nx but isn't a declared Nx.Backend callback —
  # we still need to satisfy the dispatch table, so no `@impl true`.
  def lu(out, tensor, opts), do: fallback(:lu, [out, tensor, opts])

  @impl true
  def to_batched(out, tensor, opts), do: fallback(:to_batched, [out, tensor, opts])

  @impl true
  def from_pointer(_out, _pointer, _backend_opts, _offset, _byte_size) do
    raise "NxArm does not support from_pointer"
  end

  @impl true
  def to_pointer(_tensor, _opts) do
    raise "NxArm does not support to_pointer"
  end

  @impl true
  def block(struct, _output, args, fun) do
    # Mirrors `Nx.BinaryBackend.block/4`: apply the user-supplied `fun`
    # to the captured struct + args. Used by Nx for grouping operations
    # under a Defn-style block construct.
    apply(fun, [struct | args])
  end

  # ── Custom fused ops (dispatched by NxArm.Compiler rewrite pass) ──

  @doc """
  Fused softmax. Reached only via the NxArm.Compiler pattern-fusion
  pass — direct user code should call `NxArm.softmax/2` instead.
  """
  def nxarm_softmax(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, axis) do
    shape = Nx.shape(tensor) |> Tuple.to_list()
    rank = length(shape)
    axis = if axis < 0, do: axis + rank, else: axis

    if axis != rank - 1 do
      raise ArgumentError,
            "nxarm_softmax currently only supports the last axis (got #{axis} for rank #{rank})"
    end

    inner = elem(Nx.shape(tensor), rank - 1)
    n_outer = div(Nx.size(tensor), inner)
    bin = bin_of(tensor)
    out_bin = NxArm.Native.softmax_f32_op(bin, n_outer, inner)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  @doc "Fused GELU. Reached via NxArm.Compiler pattern fusion."
  def nxarm_gelu(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    bin = bin_of(tensor)
    out_bin = NxArm.Native.gelu_f32_op(bin)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  @doc "Fused LayerNorm along the last axis. Reached via NxArm.Compiler pattern fusion."
  def nxarm_layernorm(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = gamma, %Nx.Tensor{} = beta, epsilon) do
    rank = tuple_size(Nx.shape(tensor))
    inner = elem(Nx.shape(tensor), rank - 1)
    n_outer = div(Nx.size(tensor), inner)
    bin = bin_of(tensor)
    gamma_bin = bin_of(gamma)
    beta_bin = bin_of(beta)
    out_bin = NxArm.Native.layernorm_f32_op(bin, gamma_bin, beta_bin, n_outer, inner, epsilon)
    put_in(out.data, %__MODULE__{bin: out_bin})
  end

  # ── Internal helpers ──────────────────────────────────────

  defp bin_of(%Nx.Tensor{data: %__MODULE__{bin: bin}}) when not is_nil(bin), do: bin
  defp bin_of(%Nx.Tensor{} = t), do: Nx.to_binary(t)

  @doc false
  # Public for use by NxArm fused-op helpers (e.g. NxArm.softmax).
  def __bin_of__(t), do: bin_of(t)

  # If `t` already lives on NxArm.Backend, return it untouched. Otherwise
  # materialise its bytes and create a fresh NxArm tensor of the same
  # shape/type. Used at the entry of ops (like conv) that go through
  # several Nx.* helpers — those dispatch on the tensor's backend, and
  # we don't want them to land on the slow pure-Elixir BinaryBackend.
  defp ensure_on_arm(%Nx.Tensor{data: %__MODULE__{}} = t), do: t
  defp ensure_on_arm(%Nx.Tensor{} = t), do: Nx.backend_copy(t, __MODULE__)

  # A tensor is "bias-add compatible" with `out` when it has the same
  # total size as the last axis of out — covers {K}, {1, K}, {1, 1, K},
  # and other right-aligned cases that broadcast across out's leading
  # axes.
  defp bias_add_compatible?(%Nx.Tensor{} = right, %Nx.Tensor{} = out) do
    out_shape = Nx.shape(out)
    rank = tuple_size(out_shape)
    last_axis_size = elem(out_shape, rank - 1)

    Nx.type(right) == {:f, 32} and Nx.size(right) == last_axis_size
  end

  # Pull the bytes of the bias as a flat length-`inner` f32 binary,
  # regardless of the multi-dim shape (e.g. {1, 1, 192} or {192}).
  defp bias_flat_bin(%Nx.Tensor{} = bias), do: bin_of(bias)

  defp element_size({_kind, bits}) when rem(bits, 8) == 0, do: div(bits, 8)
  defp element_size({:bf, 16}), do: 2

  defp dtype_code({:f, 32}), do: 0
  defp dtype_code({:f, 64}), do: 1
  defp dtype_code({:s, 8}),  do: 2
  defp dtype_code({:s, 16}), do: 3
  defp dtype_code({:s, 32}), do: 4
  defp dtype_code({:s, 64}), do: 5
  defp dtype_code({:u, 8}),  do: 6
  defp dtype_code({:u, 16}), do: 7
  defp dtype_code({:u, 32}), do: 8
  defp dtype_code({:u, 64}), do: 9
  defp dtype_code({:bf, 16}), do: 10
  defp dtype_code({:f, 16}), do: 11

  defp pred_to_bytes(%Nx.Tensor{} = pred) do
    case Nx.type(pred) do
      {:u, 8} -> bin_of(pred)
      {:s, 8} -> bin_of(pred)
      _ ->
        # Promote any predicate to {:u, 8} bytes via NIF as_type.
        n = Nx.size(pred)
        NxArm.Native.as_type_op(bin_of(pred), dtype_code(Nx.type(pred)), 6, n)
    end
  end

  defp to_f32_scalar(%Nx.Tensor{} = t) do
    bin = bin_of(t)

    case Nx.type(t) do
      {:f, 32} -> <<v::float-little-32>> = bin; v
      {:f, 64} -> <<v::float-little-64>> = bin; v
      {:bf, 16} ->
        <<bf::little-16>> = bin
        <<v::float-little-32>> = <<bf::little-16, 0::little-16>>
        v
      {:s, 64} -> <<v::little-signed-64>> = bin; v * 1.0
      {:s, 32} -> <<v::little-signed-32>> = bin; v * 1.0
      {:s, 16} -> <<v::little-signed-16>> = bin; v * 1.0
      {:s, 8} -> <<v::signed-8>> = bin; v * 1.0
      {:u, 64} -> <<v::little-unsigned-64>> = bin; v * 1.0
      {:u, 32} -> <<v::little-unsigned-32>> = bin; v * 1.0
      {:u, 16} -> <<v::little-unsigned-16>> = bin; v * 1.0
      {:u, 8} -> <<v::unsigned-8>> = bin; v * 1.0
      other -> raise "NxArm: to_f32_scalar can't convert dtype #{inspect(other)}"
    end
  end

  defp fallback(callback, args) do
    cpu_args = Enum.map(args, &to_cpu_arg/1)

    if System.get_env("NXARM_TRACE_FALLBACK") == "1" do
      shapes =
        cpu_args
        |> Enum.filter(&match?(%Nx.Tensor{}, &1))
        |> Enum.map(&Nx.shape/1)

      IO.puts("[NxArm fallback] #{callback} shapes=#{inspect(shapes)}")
    end

    case apply(Nx.BinaryBackend, callback, cpu_args) do
      %Nx.Tensor{} = t -> to_nx_arm(t)
      other -> other
    end
  end

  defp to_cpu_arg(%Nx.Tensor{data: %__MODULE__{bin: bin}} = t) when not is_nil(bin) do
    Nx.BinaryBackend.from_binary(%{t | data: %Nx.BinaryBackend{}}, bin, [])
  end

  defp to_cpu_arg(%Nx.Tensor{data: %__MODULE__{}} = t) do
    %{t | data: %Nx.BinaryBackend{}}
  end

  defp to_cpu_arg(list) when is_list(list), do: Enum.map(list, &to_cpu_arg/1)
  defp to_cpu_arg(other), do: other

  defp to_nx_arm(%Nx.Tensor{} = t), do: put_in(t.data, %__MODULE__{bin: Nx.to_binary(t)})
end
