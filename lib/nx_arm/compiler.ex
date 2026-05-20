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

    {expr, output, cache} = precompile(fun, vars, hooks)

    # Rewrite hook — no-op in Phase 1.
    expr = rewrite(expr)

    fn [params] ->
      state = %{params: params, gc: gc?, hooks: hooks}
      [expr |> composite_eval(state, [cache]) |> apply_output(output)]
    end
  end

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts) do
    raise "sharding is not supported by NxArm.Compiler"
  end

  # ── Rewrite hook ──────────────────────────────────────────
  # Phase 2 will pattern-match softmax/GELU/LayerNorm subgraphs here.

  defp rewrite(expr), do: expr

  # ── Precompile (build refcount cache; mirrors Nx.Defn.Evaluator) ──

  defp apply_output({result, _cache}, output) do
    {result, []} =
      Composite.traverse(result, output, fn result, [out | acc] ->
        {%{out | data: result.data}, acc}
      end)

    result
  end

  defp precompile(fun, vars, hooks) do
    {expr, output} =
      vars
      |> fun.()
      |> Composite.traverse([], &{Nx.devectorize(&1), [Nx.to_template(&1) | &2]})

    state = %{hooks: hooks, parent_ids: nil, current_ids: nil}
    {expr, cache} = init_compute_cache(expr, state)
    {expr, Enum.reverse(output), cache}
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

    {apply(mod, op, args), caches}
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
