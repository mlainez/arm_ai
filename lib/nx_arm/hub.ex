defmodule NxArm.Hub do
  @moduledoc """
  Thin shim around `NervesModelHub` for back-compat. The model
  downloader was extracted from nx_arm into its own package; new
  code should call `NervesModelHub` directly.

  Reads config under `:nx_arm, :models` (the old namespace) so
  existing target.exs files keep working without changes.
  """

  @doc "See `NervesModelHub.ensure_all/1`."
  def ensure_all, do: NervesModelHub.ensure_all(app: :nx_arm)

  @doc "See `NervesModelHub.ensure_one/2`."
  defdelegate ensure_one(id, spec), to: NervesModelHub

  @doc "See `NervesModelHub.path/2`."
  def path(id), do: NervesModelHub.path(id, app: :nx_arm)
end
