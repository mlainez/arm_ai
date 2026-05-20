defmodule NxArmCase do
  @moduledoc """
  Test helpers for comparing `NxArm.Backend` results against a
  reference `Nx.BinaryBackend` computation.

  ## Example

      use NxArmCase

      test "add matches BinaryBackend" do
        a = Nx.tensor([[1.0, 2.0]])
        b = Nx.tensor([[3.0, 4.0]])
        assert_arm_matches_ref(fn input -> Nx.add(input, b) end, a)
      end
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import NxArmCase
    end
  end

  @doc """
  Runs `fun.(input)` first on `Nx.BinaryBackend` (reference) and then
  on `NxArm.Backend`, asserting the two outputs match within `tol`.
  """
  def assert_arm_matches_ref(fun, input, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-5)
    cpu_input = Nx.backend_copy(input, Nx.BinaryBackend)
    arm_input = Nx.backend_copy(input, NxArm.Backend)

    ref = fun.(cpu_input) |> Nx.backend_copy(Nx.BinaryBackend)
    got = fun.(arm_input) |> Nx.backend_copy(Nx.BinaryBackend)

    assert Nx.shape(ref) == Nx.shape(got),
           "shapes differ: ref #{inspect(Nx.shape(ref))}, got #{inspect(Nx.shape(got))}"

    assert Nx.type(ref) == Nx.type(got),
           "types differ: ref #{inspect(Nx.type(ref))}, got #{inspect(Nx.type(got))}"

    diff =
      Nx.subtract(ref, got)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff <= tol,
           "tensors differ by #{diff}, tolerance #{tol}\nref:\n#{inspect(ref, limit: 30)}\ngot:\n#{inspect(got, limit: 30)}"
  end

  @doc """
  Same as `assert_arm_matches_ref/3` but for multi-argument functions.
  Each arg is transferred to both backends; the function is called
  twice and outputs compared.
  """
  def assert_arm_matches_ref_n(fun, args, opts \\ []) do
    tol = Keyword.get(opts, :tol, 1.0e-5)
    cpu_args = Enum.map(args, &Nx.backend_copy(&1, Nx.BinaryBackend))
    arm_args = Enum.map(args, &Nx.backend_copy(&1, NxArm.Backend))

    ref = apply(fun, cpu_args) |> Nx.backend_copy(Nx.BinaryBackend)
    got = apply(fun, arm_args) |> Nx.backend_copy(Nx.BinaryBackend)

    assert Nx.shape(ref) == Nx.shape(got)
    assert Nx.type(ref) == Nx.type(got)

    diff =
      Nx.subtract(ref, got)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff <= tol,
           "tensors differ by #{diff}, tolerance #{tol}"
  end
end
