defmodule NxArm.Compiler do
  @moduledoc """
  Custom `Nx.Defn.Compiler` for `NxArm.Backend`.

  ## Phase 1 — identity (current)

  This compiler walks the `Nx.Defn.Expr` graph and dispatches each
  op directly to `NxArm.Backend`. Behaviourally equivalent to
  `Nx.Defn.Evaluator` configured against our backend, plus:

    * `__to_backend__/1` returns `NxArm.Backend` so any creation /
      constant op lands on us by default;
    * The graph is walked through a rewriter pass that's a no-op
      today — the hook is in place for the fusion phase.

  ## Phase 2 — pattern fusion (next)

  The rewriter pass will pattern-detect softmax, GELU, and LayerNorm
  subgraphs and replace them with single custom-op Expr nodes
  (`:nxarm_softmax`, `:nxarm_gelu`, `:nxarm_layernorm`) that dispatch
  to fused NIFs.

  ## Usage

      Nx.Defn.default_options(compiler: NxArm.Compiler)
      # ...or per-call:
      Nx.Defn.jit(&forward/1, compiler: NxArm.Compiler)

  Bumblebee's `Axon.build/2` accepts `compiler:`:

      {_init, predict_fn} = Axon.build(model, mode: :inference, compiler: NxArm.Compiler)
  """

  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.{Composite, Expr, Tree}

  @creation_ops [:eye, :iota, :from_binary]
  @list_ops [:concatenate, :stack]
  @indices_ops [:slice, :put_slice]

  @impl true
  def __partitions_options__(opts) do
    List.duplicate(opts, Keyword.get(opts, :max_concurrency, 1))
  end

  @impl true
  def __to_backend__(_opts), do: {NxArm.Backend, []}

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, opts) do
    hooks = Keyword.get(opts, :hooks, %{})
    gc? = Keyword.get(opts, :garbage_collect, false)

    # 1. Build the raw Expr tree (keep args intact for rewriting).
    {expr, output} = build_raw_expr(fun, vars)

    # 2. Phase-2 rewrite pass: pattern-fuse softmax / (next: GELU,
    #    LayerNorm) into custom op nodes.
    expr = rewrite(expr)

    # 3. Build the refcount cache on the rewritten tree (this is what
    #    strips args).
    {expr, cache} = init_compute_cache(expr, %{hooks: hooks, parent_ids: nil, current_ids: nil})

    fn [params] ->
      state = %{params: params, gc: gc?, hooks: hooks}
      [expr |> composite_eval(state, [cache]) |> apply_output(output)]
    end
  end

  defp build_raw_expr(fun, vars) do
    {expr, output} =
      vars
      |> fun.()
      |> Composite.traverse([], &{Nx.devectorize(&1), [Nx.to_template(&1) | &2]})

    {expr, Enum.reverse(output)}
  end

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts) do
    raise "sharding is not supported by NxArm.Compiler"
  end

  # ── Rewrite pass ──────────────────────────────────────────
  #
  # Walks the Expr graph bottom-up. At each node, after children have
  # been rewritten, tries to match a fusion pattern. If matched,
  # replaces the subtree with a single custom-op node that dispatches
  # to a fused NIF (NxArm.Backend.nxarm_softmax/3, etc).

  defp rewrite(expr) do
    {result, _cache} = Nx.Defn.Composite.traverse(expr, %{}, &rewrite_node/2)
    result
  end

  defp rewrite_node(%Nx.Tensor{data: %Expr{op: op}} = tensor, cache)
       when op in [:tensor, :constant, :parameter] do
    {tensor, cache}
  end

  defp rewrite_node(%Nx.Tensor{data: %Expr{id: id}} = tensor, cache) do
    case cache do
      %{^id => already} ->
        {already, cache}

      %{} ->
        # Bottom-up: rewrite arg children first. For control-flow ops
        # (:cond, :while, :fun, :block, :token, :metadata, :slice,
        # :put_slice, :runtime_call) we fall back to identity for now
        # — those don't show up in typical forward passes (Bumblebee
        # ViT etc.) and adding fusion across them needs more care.
        {new_args, cache} = Nx.Defn.Tree.apply_args(tensor, cache, &rewrite_node/2)
        tensor_with_new = put_in(tensor.data.args, new_args)
        rewritten = try_patterns(tensor_with_new)
        {rewritten, Map.put(cache, id, rewritten)}
    end
  end

  # ── Pattern matchers ─────────────────────────────────────

  defp try_patterns(tensor) do
    try_dead_broadcast(tensor) ||
      try_dropout_elim(tensor) ||
      try_constant_fold(tensor) ||
      try_softmax_divide(tensor) ||
      try_softmax_multiply(tensor) ||
      try_gelu(tensor) ||
      try_layernorm(tensor) ||
      tensor
  end

  # B7: Constant folding. When both operands of a binary op are
  # `:constant` Expr nodes (and the op is exactly representable
  # without round-off concerns), evaluate at compile time and emit a
  # new constant. Nx's expr.ex already folds the common cases at
  # graph-build time, so this is a defensive net that catches what
  # leaks through our rewriter (e.g., a constant produced by an
  # earlier fusion pass).
  defp try_constant_fold(%Nx.Tensor{data: %Expr{op: op, args: [a, b]}} = tensor)
       when op in [:add, :subtract, :multiply, :divide, :max, :min] do
    with {:ok, av} <- extract_scalar_const(a),
         {:ok, bv} <- extract_scalar_const(b) do
      result =
        case op do
          :add -> av + bv
          :subtract -> av - bv
          :multiply -> av * bv
          :divide -> av / bv
          :max -> max(av, bv)
          :min -> min(av, bv)
        end

      if System.get_env("NXARM_TRACE_FUSION") == "1" do
        IO.puts("[NxArm fusion] constant fold #{op}(#{av}, #{bv}) = #{result}")
      end

      %{tensor | data: %Expr{id: make_ref(), op: :constant, args: [result], context: tensor.data.context}}
    else
      _ -> nil
    end
  end

  defp try_constant_fold(_), do: nil

  defp extract_scalar_const(%Nx.Tensor{data: %Expr{op: :constant, args: [n]}}) when is_number(n),
    do: {:ok, n * 1.0}

  defp extract_scalar_const(_), do: :error

  # B4: Dead broadcast elimination. `Nx.broadcast(x, shape)` where
  # the input already matches `shape` is an identity — but Nx itself
  # has substantial wrapper overhead per call (axes computation,
  # `apply_vectorized`, etc.). Replacing the Expr node with its input
  # tensor skips that overhead entirely.
  defp try_dead_broadcast(%Nx.Tensor{
         data: %Expr{op: :broadcast, args: [inner, target_shape, _axes]}
       }) do
    if Nx.shape(inner) == target_shape do
      if System.get_env("NXARM_TRACE_FUSION") == "1" do
        IO.puts("[NxArm fusion] dead broadcast eliminated (shape=#{inspect(target_shape)})")
      end

      inner
    else
      nil
    end
  end

  defp try_dead_broadcast(_), do: nil

  # B3: Dropout elimination. In inference mode, dropout is identity.
  # Axon's `:dropout` op (when seen at our level) is wrapped in a
  # `:metadata` Expr node by Nx.Defn; for safety we recognise the
  # explicit form too.
  defp try_dropout_elim(%Nx.Tensor{data: %Expr{op: :metadata, args: [inner, %{dropout: true}]}}) do
    if System.get_env("NXARM_TRACE_FUSION") == "1" do
      IO.puts("[NxArm fusion] dropout eliminated")
    end

    inner
  end

  defp try_dropout_elim(_), do: nil

  # softmax composed as `e / s` (my manual form, sanity check).
  #
  # Tree shape:
  #   divide(
  #     exp_node = exp(subtract(x, broadcast(reduce_max(x, …)))),
  #     broadcast(sum(exp_node_same_id, …), …)
  #   )
  defp try_softmax_divide(%Nx.Tensor{data: %Expr{op: :divide, args: [num, denom]}} = tensor) do
    with %Nx.Tensor{data: %Expr{id: exp_id, op: :exp, args: [shifted]}} <- num,
         {:ok, sum_t} <- unwrap_to_sum(denom),
         %Nx.Tensor{data: %Expr{op: :sum, args: [inner_exp, sum_opts]}} <- sum_t,
         %Nx.Tensor{data: %Expr{id: ^exp_id}} <- inner_exp,
         %Nx.Tensor{data: %Expr{op: :subtract, args: [x, _broadcast_of_max]}} <- shifted,
         {:ok, axis} <- last_axis_match(sum_opts, tensor) do
      build_softmax(tensor, x, axis)
    else
      _ -> nil
    end
  end

  # Accept either a bare sum or a broadcast-wrapped sum.
  defp unwrap_to_sum(%Nx.Tensor{data: %Expr{op: :sum}} = sum_t), do: {:ok, sum_t}

  defp unwrap_to_sum(%Nx.Tensor{data: %Expr{op: :broadcast, args: [inner | _]}}),
    do: unwrap_to_sum(inner)

  defp unwrap_to_sum(_), do: :error

  defp try_softmax_divide(_), do: nil

  # softmax composed as `reciprocal(s) * e` (Axon.Activations.softmax form).
  #
  # `reciprocal(z) = divide(1.0, z)` lowers to `divide(constant_1, z)` in defn.
  #
  # Tree shape:
  #   multiply(
  #     broadcast(divide(constant_1, sum(exp_node, …)), …),
  #     exp_node_same_id
  #   )
  # (operands may be swapped — Nx.multiply commutes constant operands)
  defp try_softmax_multiply(%Nx.Tensor{data: %Expr{op: :multiply, args: [left, right]}} = tensor) do
    try_softmax_multiply_ordered(tensor, left, right) ||
      try_softmax_multiply_ordered(tensor, right, left)
  end

  defp try_softmax_multiply(_), do: nil

  defp try_softmax_multiply_ordered(tensor, recip_side, exp_side) do
    with %Nx.Tensor{data: %Expr{id: exp_id, op: :exp, args: [shifted]}} <- exp_side,
         %Nx.Tensor{data: %Expr{op: :subtract, args: [x, _broadcast_of_max]}} <- shifted,
         {:ok, sum_t} <- unwrap_broadcast_or_div(recip_side),
         %Nx.Tensor{data: %Expr{op: :sum, args: [inner_exp, sum_opts]}} <- sum_t,
         %Nx.Tensor{data: %Expr{id: ^exp_id}} <- inner_exp,
         {:ok, axis} <- last_axis_match(sum_opts, tensor) do
      build_softmax(tensor, x, axis)
    else
      _ -> nil
    end
  end

  # The reciprocal-side may be either:
  #   - divide(constant_1, sum_t)             (no broadcast yet, scalar)
  #   - broadcast(divide(constant_1, sum_t))  (broadcast wraps the divide)
  defp unwrap_broadcast_or_div(%Nx.Tensor{data: %Expr{op: :broadcast, args: [inner | _]}}),
    do: unwrap_broadcast_or_div(inner)

  defp unwrap_broadcast_or_div(%Nx.Tensor{data: %Expr{op: :divide, args: [one, denom]}}) do
    case one do
      %Nx.Tensor{data: %Expr{op: :constant, args: [n]}} when n == 1 or n == 1.0 ->
        {:ok, denom}

      _ ->
        :error
    end
  end

  defp unwrap_broadcast_or_div(_), do: :error

  # Confirm `sum`/`reduce_max` axes correspond to the last axis of the
  # input tensor, the only configuration our fused softmax NIF handles.
  defp last_axis_match(opts, tensor) when is_list(opts) do
    axes = Keyword.get(opts, :axes, nil)
    rank = tuple_size(Nx.shape(tensor))

    case axes do
      [axis] when axis == rank - 1 -> {:ok, rank - 1}
      [axis] when axis == -1 -> {:ok, rank - 1}
      _ -> :error
    end
  end

  defp last_axis_match(_, _), do: :error

  defp build_softmax(out_tensor, x, axis) do
    if System.get_env("NXARM_TRACE_FUSION") == "1" do
      IO.puts("[NxArm fusion] softmax shape=#{inspect(Nx.shape(out_tensor))} axis=#{axis}")
    end

    new_data = %Expr{
      id: make_ref(),
      op: :nxarm_softmax,
      args: [x, axis],
      context: out_tensor.data.context
    }

    %{out_tensor | data: new_data}
  end

  # ── GELU pattern: divide(multiply(add(erf(divide(x, √2)), 1), x), 2) ──
  # Nx commutes constants to the front in add/multiply, so we have to
  # accept either operand order on the add and multiply.
  defp try_gelu(%Nx.Tensor{data: %Expr{op: :divide, args: [num, denom]}} = tensor) do
    with true <- constant_close?(denom, 2.0),
         %Nx.Tensor{data: %Expr{op: :multiply, args: [a, b]}} <- num,
         {erf_add_t, x_t} <- pick_erf_add_and_x(a, b),
         {:ok, erf_t} <- pick_erf_from_add(erf_add_t),
         %Nx.Tensor{data: %Expr{op: :erf, args: [div_t]}} <- erf_t,
         %Nx.Tensor{data: %Expr{op: :divide, args: [x_in, sqrt2_t]}} <- div_t,
         true <- constant_close?(sqrt2_t, :math.sqrt(2.0)),
         true <- same_id?(x_in, x_t) do
      build_gelu(tensor, x_t)
    else
      _ -> nil
    end
  end

  # `add(erf, 1)` and `add(1, erf)` both legal — Nx commutes constants.
  defp pick_erf_from_add(%Nx.Tensor{data: %Expr{op: :add, args: [a, b]}}) do
    cond do
      match?(%Nx.Tensor{data: %Expr{op: :erf}}, a) and constant_close?(b, 1.0) -> {:ok, a}
      match?(%Nx.Tensor{data: %Expr{op: :erf}}, b) and constant_close?(a, 1.0) -> {:ok, b}
      true -> :error
    end
  end

  defp pick_erf_from_add(_), do: :error

  defp try_gelu(_), do: nil

  # In `multiply(a, b)` where one is (add(erf(...), 1)) and the other
  # is `x`, return them in canonical order. Nx may commute factors.
  defp pick_erf_add_and_x(a, b) do
    cond do
      match?(%Nx.Tensor{data: %Expr{op: :add}}, a) -> {a, b}
      match?(%Nx.Tensor{data: %Expr{op: :add}}, b) -> {b, a}
      true -> {nil, nil}
    end
  end

  defp constant_close?(%Nx.Tensor{data: %Expr{op: :constant, args: [n]}}, target) when is_number(n) do
    abs(n - target) < 1.0e-4
  end

  defp constant_close?(_, _), do: false

  defp same_id?(%Nx.Tensor{data: %Expr{id: a}}, %Nx.Tensor{data: %Expr{id: b}}), do: a == b
  defp same_id?(_, _), do: false

  defp build_gelu(out_tensor, x) do
    if System.get_env("NXARM_TRACE_FUSION") == "1" do
      IO.puts("[NxArm fusion] gelu shape=#{inspect(Nx.shape(out_tensor))}")
    end

    new_data = %Expr{
      id: make_ref(),
      op: :nxarm_gelu,
      args: [x],
      context: out_tensor.data.context
    }

    %{out_tensor | data: new_data}
  end

  # ── LayerNorm pattern (Axon's `scale * (input - mean) + bias`) ──
  #
  # Tree (input/output shape `{..., hidden}`):
  #   add(
  #     multiply(
  #       multiply(gamma_param_or_reshape,
  #                rsqrt(add(variance_t, eps_const))),
  #       subtract(x_t, mean_t)
  #     ),
  #     beta_param_or_reshape
  #   )
  #
  # `variance_t` is itself a chain `mean((x - mean(x))^2)` — we don't
  # match it exhaustively; instead we require that the subtract's left
  # is `x_t` and the rsqrt input has the right shape (1 in the last
  # axis). Good enough to catch Bumblebee/Axon's LayerNorm without
  # false positives.
  defp try_layernorm(%Nx.Tensor{data: %Expr{op: :add, args: [a, b]}} = tensor) do
    # Top-level `add(inner_multiply, beta)` — but Nx commutes constants,
    # and `beta` is a parameter so it may end up first too if the
    # multiply contains a constant. Try both orders.
    try_layernorm_ordered(tensor, a, b) || try_layernorm_ordered(tensor, b, a)
  end

  defp try_layernorm(_), do: nil

  defp try_layernorm_ordered(tensor, inner, beta_t) do
    with %Nx.Tensor{data: %Expr{op: :multiply, args: [m1, m2]}} <- inner,
         {scale_t, normed_t} <- pick_scale_and_normed(m1, m2),
         %Nx.Tensor{data: %Expr{op: :multiply, args: [s1, s2]}} <- scale_t,
         {gamma_t, rsqrt_t} <- pick_gamma_and_rsqrt(s1, s2),
         %Nx.Tensor{data: %Expr{op: :rsqrt, args: [var_plus_eps_t]}} <- rsqrt_t,
         %Nx.Tensor{data: %Expr{op: :add, args: [aa, bb]}} <- var_plus_eps_t,
         {:ok, eps} <- pick_eps_constant(aa, bb),
         %Nx.Tensor{data: %Expr{op: :subtract, args: [x_t, _mean_t]}} <- normed_t,
         {:ok, gamma_1d} <- unwrap_to_param_or_constant(gamma_t),
         {:ok, beta_1d} <- unwrap_to_param_or_constant(beta_t) do
      build_layernorm(tensor, x_t, gamma_1d, beta_1d, eps)
    else
      _ -> nil
    end
  end

  # In `multiply(scale, normed)` find which one is the scale
  # (`multiply(gamma, rsqrt)`) and which is the normed (`subtract`).
  defp pick_scale_and_normed(a, b) do
    cond do
      match?(%Nx.Tensor{data: %Expr{op: :subtract}}, a) and
        match?(%Nx.Tensor{data: %Expr{op: :multiply}}, b) ->
        {b, a}

      match?(%Nx.Tensor{data: %Expr{op: :subtract}}, b) and
        match?(%Nx.Tensor{data: %Expr{op: :multiply}}, a) ->
        {a, b}

      true ->
        nil
    end
  end

  defp pick_gamma_and_rsqrt(a, b) do
    cond do
      match?(%Nx.Tensor{data: %Expr{op: :rsqrt}}, b) -> {a, b}
      match?(%Nx.Tensor{data: %Expr{op: :rsqrt}}, a) -> {b, a}
      true -> nil
    end
  end

  defp pick_eps_constant(a, b) do
    case extract_constant(a) do
      {:ok, n} when n > 0.0 and n < 0.1 -> {:ok, n}
      _ -> extract_constant(b) |> validate_small_positive()
    end
  end

  defp validate_small_positive({:ok, n}) when n > 0.0 and n < 0.1, do: {:ok, n}
  defp validate_small_positive(_), do: :error

  defp try_layernorm(_), do: nil

  defp extract_constant(%Nx.Tensor{data: %Expr{op: :constant, args: [n]}}) when is_number(n),
    do: {:ok, n * 1.0}

  defp extract_constant(_), do: :error

  # gamma/beta in Axon are reshape(parameter, [..., 1, ..., hidden]).
  # We accept anything that has a parameter at the root and the last
  # axis matches the LayerNorm hidden dimension. For the fused NIF we
  # need them as flat length-`inner` vectors, so we also pass the
  # tensor as-is and the backend callback reshapes if needed.
  defp unwrap_to_param_or_constant(%Nx.Tensor{data: %Expr{op: :reshape}} = t),
    do: {:ok, t}

  defp unwrap_to_param_or_constant(%Nx.Tensor{data: %Expr{op: :parameter}} = t),
    do: {:ok, t}

  defp unwrap_to_param_or_constant(%Nx.Tensor{data: %Expr{op: :broadcast, args: [inner | _]}}),
    do: unwrap_to_param_or_constant(inner)

  defp unwrap_to_param_or_constant(_), do: :error

  defp build_layernorm(out_tensor, x, gamma, beta, eps) do
    if System.get_env("NXARM_TRACE_FUSION") == "1" do
      IO.puts("[NxArm fusion] layernorm shape=#{inspect(Nx.shape(out_tensor))} eps=#{eps}")
    end

    new_data = %Expr{
      id: make_ref(),
      op: :nxarm_layernorm,
      args: [x, gamma, beta, eps],
      context: out_tensor.data.context
    }

    %{out_tensor | data: new_data}
  end

  # ── Precompile (build refcount cache; mirrors Nx.Defn.Evaluator) ──

  defp apply_output({result, _cache}, output) do
    {result, []} =
      Composite.traverse(result, output, fn result, [out | acc] ->
        {%{out | data: result.data}, acc}
      end)

    result
  end

  defp init_compute_cache(expr, state) do
    state = %{state | parent_ids: %{}, current_ids: Tree.scope_ids(expr, %{})}
    composite_compute_cache(expr, state, %{})
  end

  defp composite_compute_cache(expr, state, cache) do
    Composite.traverse(expr, cache, &compute_cache(&1, state, &2))
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: op}} = tensor, _state, cache)
       when op in [:constant, :tensor] do
    {tensor, cache}
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, cache) do
    composite_compute_cache(expr, state, cache)
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, cache) do
    cache =
      case state.parent_ids do
        %{^id => _} ->
          Map.put_new(cache, id, tensor)

        %{} ->
          case cache do
            %{^id => {:args, counter, args_or_placeholder}} ->
              %{cache | id => {:args, counter + 1, args_or_placeholder}}

            %{} ->
              cache = Map.put(cache, id, {:args, 1, nil})
              {args, cache} = compute_cache(op, tensor, state, cache)
              Map.update!(cache, id, fn {:args, counter, _} -> {:args, counter, args} end)
          end
      end

    {put_in(tensor.data.args, nil), cache}
  end

  defp compute_cache(:fun, %{data: %Expr{args: args}}, state, cache) do
    [args, expr, _mfa] = args
    {expr, expr_cache} = init_compute_cache(expr, state)
    {[length(args), expr, expr_cache], cache}
  end

  defp compute_cache(:while, %{data: %Expr{args: args}}, state, cache) do
    [initial, _arg, pred, block] = args
    {initial, cache} = composite_compute_cache(initial, state, cache)
    {{pred, block}, while_cache} = init_compute_cache({pred, block}, state)
    {[initial, pred, block, while_cache], cache}
  end

  defp compute_cache(:block, %{data: %Expr{args: args}}, state, cache) do
    [struct, in_args, expr, callback] = args
    {call_prefix, call_suffix} = Enum.split_while(in_args, &(not is_list(&1)))
    {call_prefix, cache} = Enum.map_reduce(call_prefix, cache, &compute_cache(&1, state, &2))
    in_args = call_prefix ++ call_suffix
    {[struct, in_args, expr, callback], cache}
  end

  defp compute_cache(:cond, %{data: %Expr{args: [clauses, last]}}, state, cache) do
    %{parent_ids: parent_ids, current_ids: current_ids} = state

    clause_caches =
      Enum.map([last | clauses], fn clause ->
        state = %{
          state
          | parent_ids: current_ids,
            current_ids: Tree.scope_ids(clause, current_ids)
        }

        composite_compute_cache(clause, state, %{})
      end)

    {[last_cache | clauses_cache], {all_ids, cache}} =
      Enum.map_reduce(clause_caches, {%{}, cache}, fn {clause, clause_cache}, seen_ids_cache ->
        {clause_cache, seen_ids_cache} =
          Enum.flat_map_reduce(clause_cache, seen_ids_cache, fn
            {id, %_{} = tensor}, {seen_ids, cache} ->
              case seen_ids do
                %{^id => _} ->
                  {[], {seen_ids, cache}}

                %{} when is_map_key(parent_ids, id) ->
                  {[], {seen_ids, Map.put_new(cache, id, tensor)}}

                %{} ->
                  {_, cache} = composite_compute_cache(tensor, state, cache)
                  {[], {Map.put(seen_ids, id, true), cache}}
              end

            {id, counter}, seen_ids_cache ->
              {[{id, counter}], seen_ids_cache}
          end)

        {{clause, Map.new(clause_cache)}, seen_ids_cache}
      end)

    {[clauses_cache, last_cache, Map.keys(all_ids)], cache}
  end

  defp compute_cache(:token, %{data: %Expr{args: [token]}}, state, cache) do
    hooks = state.hooks

    {exprs_hooks, cache} =
      Enum.flat_map_reduce(token.hooks, cache, fn
        %{callback: callback, expr: expr, name: name}, cache ->
          hook_fun = hooks[name] || callback

          cond do
            hook_fun ->
              {expr, cache} = composite_compute_cache(expr, state, cache)
              {[{expr, hook_fun}], cache}

            Tree.has_hooks?(expr, hooks) ->
              {expr, cache} = composite_compute_cache(expr, state, cache)
              {[{expr, nil}], cache}

            true ->
              {[], cache}
          end
      end)

    {[exprs_hooks], cache}
  end

  defp compute_cache(_op, tensor, state, cache) do
    Tree.apply_args(tensor, cache, &compute_cache(&1, state, &2))
  end

  # ── Evaluation ────────────────────────────────────────────

  defp composite_eval(expr, state, caches) do
    Composite.traverse(expr, caches, &eval(&1, state, &2))
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _state, caches) do
    {t, caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :constant, args: [constant]}} = ans, _state, caches) do
    {backend, backend_options} = Nx.default_backend()
    {backend.constant(ans, constant, backend_options), caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, state, [cache | caches]) do
    case cache do
      %{^id => {:args, count, args}} ->
        {res, [cache | caches]} = eval_apply(op, args, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {:result, count, res}} ->
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{} ->
        eval_parent(caches, id, op, ans, state, [cache])
    end
  end

  defp decrement_cache(cache, id, 1, _res), do: Map.delete(cache, id)
  defp decrement_cache(cache, id, counter, res), do: %{cache | id => {:result, counter - 1, res}}

  defp eval_parent([cache | caches], id, op, ans, state, acc) do
    case cache do
      %{^id => {:result, _count, res}} ->
        {res, Enum.reverse(acc, [cache | caches])}

      %{^id => {:args, count, args}} ->
        {res, [cache | caches]} = eval_apply(op, args, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        {res, Enum.reverse(acc, [Map.put(cache, id, {:result, count, res}) | caches])}

      %{} ->
        eval_parent(caches, id, op, ans, state, [cache | acc])
    end
  end

  defp eval_parent([], id, op, _ans, _state, _acc) do
    raise "NxArm.Compiler cache miss for OP=#{op} ID=#{inspect(id)}"
  end

  defp decrement_parents([cache | caches], id) do
    case cache do
      %{^id => {:result, count, value}} -> [decrement_cache(cache, id, count, value) | caches]
      %{^id => {:args, count, args}} -> [%{cache | id => {:args, count - 1, args}} | caches]
      %{} -> [cache | decrement_parents(caches, id)]
    end
  end

  defp eval_apply(:parameter, [i], _ans, state, caches) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {Nx.devectorize(tensor), caches}
    end
  end

  defp eval_apply(:elem, [tuple, i], _ans, state, caches) do
    {tuple, caches} = composite_eval(tuple, state, caches)
    {elem(tuple, i), caches}
  end

  defp eval_apply(:attach_token, [token, expr], _ans, state, caches) do
    {_, caches} = eval(token, state, caches)
    eval(expr, state, caches)
  end

  defp eval_apply(:fun, [length, expr, expr_cache], _ans, state, caches) do
    fun =
      case length do
        1 ->
          fn arg1 ->
            params = [fn -> Nx.to_tensor(arg1) end]
            {result, _} = composite_eval(expr, %{state | params: params}, [expr_cache])
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [fn -> Nx.to_tensor(arg1) end, fn -> Nx.to_tensor(arg2) end]
            {result, _} = composite_eval(expr, %{state | params: params}, [expr_cache])
            result
          end
      end

    {fun, caches}
  end

  defp eval_apply(:cond, [clauses_cache, last_cache, parent_ids], _ans, state, caches) do
    {chosen, chosen_cache} = cond_clause(clauses_cache, last_cache, state, caches)
    {res, [_ | caches]} = composite_eval(chosen, state, chosen_cache)
    caches = Enum.reduce(parent_ids, caches, &decrement_parents(&2, &1))
    {res, caches}
  end

  defp eval_apply(:while, [initial, pred, block, while_cache], _ans, state, caches) do
    {initial, caches} = composite_eval(initial, state, caches)
    {while(initial, pred, block, state, [while_cache]), caches}
  end

  defp eval_apply(:token, [exprs_hooks], _ans, state, caches) do
    caches =
      List.foldr(exprs_hooks, caches, fn {expr, hook_fun}, caches ->
        {res, caches} = composite_eval(expr, state, caches)
        hook_fun && hook_fun.(res)
        caches
      end)

    {{}, caches}
  end

  defp eval_apply(:block, [struct, in_args, expr, callback], ans, state, caches) do
    {in_args, caches} = Enum.map_reduce(in_args, caches, &eval(&1, state, &2))
    {param_prefix, _} = Enum.split_while(in_args, &(not is_list(&1)))
    backend = Nx.Shared.list_impl!(param_prefix)

    out =
      case ans do
        %{type: {:tuple, _}} -> expr
        _ -> ans
      end

    {backend.block(struct, out, in_args, callback), caches}
  end

  defp eval_apply(:runtime_call, [expr, fun, out_template, opts], _ans, state, caches) do
    {tensor_value, caches} = composite_eval(expr, state, caches)
    result = fun.(tensor_value, opts)

    if not Nx.compatible?(out_template, result) do
      raise "expected the runtime_call function to match the given output template"
    end

    case out_template do
      %Nx.Tensor{} -> {result, caches}
      _ -> {[result] |> Composite.flatten_list() |> List.to_tuple(), caches}
    end
  end

  defp eval_apply(op, args, ans, state, caches) do
    ans = put_in(ans.data.args, args)
    {args, caches} = Tree.apply_args(ans, caches, &eval(&1, state, &2))

    {mod, args} =
      cond do
        op in @creation_ops ->
          {backend, backend_options} = Nx.default_backend()
          {backend, [ans | args] ++ [backend_options]}

        op in @list_ops ->
          {Nx.Shared.list_impl!(hd(args)), [ans | args]}

        op in @indices_ops ->
          [tensor, indices | _] = args
          {Nx.Shared.list_impl!([tensor | indices]), [ans | args]}

        match?({:tuple, _}, ans.type) ->
          {Nx.Shared.list_impl!(args), args}

        true ->
          {Nx.Shared.list_impl!(args), [ans | args]}
      end

    if System.get_env("NXARM_PROFILE_OPS") == "1" do
      {us, result} = :timer.tc(fn -> apply(mod, op, args) end)

      key =
        case System.get_env("NXARM_PROFILE_BY_SHAPE") do
          "1" ->
            shapes =
              args
              |> Enum.filter(&match?(%Nx.Tensor{}, &1))
              |> Enum.map(&Nx.shape/1)

            {op, shapes}

          _ ->
            op
        end

      bump_counter(key, us)
      {result, caches}
    else
      {apply(mod, op, args), caches}
    end
  end

  @profile_table :__nxarm_op_profile__

  defp bump_counter(op, us) do
    table =
      case :ets.whereis(@profile_table) do
        :undefined ->
          :ets.new(@profile_table, [:set, :public, :named_table, write_concurrency: true])

        ref ->
          ref
      end

    :ets.update_counter(table, op, [{2, us}, {3, 1}], {op, 0, 0})
  end

  @doc """
  Pretty-print the op-time aggregation collected when
  `NXARM_PROFILE_OPS=1` was set. Resets the table.
  """
  def dump_profile() do
    case :ets.whereis(@profile_table) do
      :undefined ->
        IO.puts("no profile data — set NXARM_PROFILE_OPS=1 before running")

      _ref ->
        rows = :ets.tab2list(@profile_table)

        IO.puts("op                    calls    total_ms   avg_us")
        IO.puts("--------------------- -------- ---------- --------")

        rows
        |> Enum.sort_by(fn {_op, us, _n} -> -us end)
        |> Enum.each(fn {op_key, us, n} ->
          label = inspect(op_key, limit: :infinity, printable_limit: :infinity)
          IO.puts(
            "#{String.pad_leading(to_string(n), 6)}  " <>
              "#{String.pad_leading(:erlang.float_to_binary(us / 1000, decimals: 1), 10)}ms  " <>
              "#{String.pad_leading(:erlang.float_to_binary(us / n, decimals: 1), 10)}us  " <>
              "#{label}"
          )
        end)

        :ets.delete_all_objects(@profile_table)
    end
  end

  # ── While / cond plumbing ─────────────────────────────────

  defp while(initial, pred, block, state, while_cache) do
    {pred_value, _} = composite_eval(pred, %{state | params: composite_to_params(initial)}, while_cache)

    if Nx.to_number(pred_value) != 0 do
      {next, _} = composite_eval(block, %{state | params: composite_to_params(initial)}, while_cache)
      while(next, pred, block, state, while_cache)
    else
      initial
    end
  end

  defp composite_to_params(composite) do
    composite
    |> Composite.flatten_list()
    |> Enum.map(fn t -> fn -> t end end)
  end

  defp cond_clause([{pred, expr, clause_cache} | clauses], last_cache, state, caches) do
    {pred, _} = composite_eval(pred, state, caches)

    if Nx.to_number(pred) != 0 do
      {expr, [clause_cache | caches]}
    else
      cond_clause(clauses, last_cache, state, caches)
    end
  end

  defp cond_clause([], {last_expr, last_cache}, _state, caches) do
    {last_expr, [last_cache | caches]}
  end
end
