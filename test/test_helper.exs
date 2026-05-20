ExUnit.start()

# NxArm tests run on whatever host the developer is on; we keep the
# global default backend as BinaryBackend so reference computations
# stay reproducible. Tests opt into NxArm.Backend with
# `Nx.backend_transfer(t, NxArm.Backend)`.
Nx.global_default_backend(Nx.BinaryBackend)
