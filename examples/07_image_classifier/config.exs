import Config

# Bumblebee handles model fetch + caching itself — no ArmAI.Hub entry
# needed. Just make sure the device has internet on first boot.
config :nx_arm, features: ["full"]

# Cache Bumblebee model files on the writable data partition.
config :bumblebee, :cache_dir, "/root/.bumblebee"
