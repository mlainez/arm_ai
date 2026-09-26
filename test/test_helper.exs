# Tests tagged :models run real models. Point NERVES_AI_MODELS at a
# directory holding the files named in each test to enable them.
exclude = if System.get_env("NERVES_AI_MODELS"), do: [], else: [:models]
ExUnit.start(exclude: exclude)
