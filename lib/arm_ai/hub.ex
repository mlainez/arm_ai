defmodule ArmAI.Hub do
  @moduledoc """
  Thin shim around `ModelHub` for back-compat. The model
  downloader was extracted from nx_arm into its own package; new
  code should call `ModelHub` directly.

  Reads config under `:nx_arm, :models` (the old namespace) so
  existing target.exs files keep working without changes.
  """

  @doc "See `ModelHub.ensure_all/1`."
  def ensure_all, do: ModelHub.ensure_all(app: :nx_arm)

  @doc "See `ModelHub.ensure_one/2`."
  defdelegate ensure_one(id, spec), to: ModelHub

  @doc "See `ModelHub.path/2`."
  def path(id), do: ModelHub.path(id, app: :nx_arm)
end
