import Config

# Smallest config that supports embedding-based retrieval.
# Embeddings primitives are in the `core` set, so for the *retrieval*
# half (the part this example demonstrates) you actually need no
# features at all. If you also want to encode text at runtime, add
# `onnx` + `tokenizers`:
config :nx_arm, features: ["sentence-rag"]

# For a real sentence encoder + tokenizer:
config :nx_arm,
  models: [
    minilm: [
      source: {:hf, "sentence-transformers/all-MiniLM-L6-v2",
                    "onnx/model.onnx"},
      path: "/root/models/all-MiniLM-L6-v2.onnx"
    ],
    minilm_tokenizer: [
      source: {:hf, "sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json"},
      path: "/root/models/all-MiniLM-L6-v2-tokenizer.json"
    ]
  ]
