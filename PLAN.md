# nx_arm versatile-edge-backend plan

10 features in priority order. Each unlocks a model class or removes a
blocker. Tagged with effort and what it unlocks. Tests required for
each item before moving on.

**Status (2026-05-20):** All 10 items shipped. 69 tests passing on host.

## 1. `gather` NEON NIF ✅
- **Unlocks**: all text encoders / embedding models (sentence-transformers, BERT-family, MiniLM). Token-embedding lookup is `gather(emb_table, token_ids)`; vocab is 30K–256K rows × hidden_dim; called every forward.
- **Effort**: 1–2 days. Generic n-D gather is complex; the common case (axis 0, 1-D indices, contiguous rows) is a memcpy loop.
- **Test**: correctness vs `Nx.BinaryBackend.gather/4` over a battery of shapes + axes.

## 2. ✅ `slice` / `put_slice` NEON NIF
- **Unlocks**: KV caches for autoregressive LLMs, sliding windows, masked-LM batches.
- **Effort**: 1–2 days.
- **Test**: correctness vs BinaryBackend + edge cases (strides, negative, dynamic starts).

## 3. ✅ Depthwise conv NIF
- **Unlocks**: MobileNet, EfficientNet, every "mobile" CNN. Currently falls through to generic conv with wrong inner loop order (5–10× slower than needed).
- **Effort**: 2–3 days. NEON-vectorised over the spatial dimensions per channel.
- **Test**: correctness vs Nx.conv with `feature_group_size == channels`; benchmark vs current path.

## 4. ✅ `max_pool` / `avg_pool` NIF
- **Unlocks**: ResNet, classic CNNs, anywhere pooling is used.
- **Effort**: 1 day each. Straightforward NEON windowed reductions.
- **Test**: correctness vs BinaryBackend pooling.

## 5. ✅ int8 matmul + mixed-precision compiler support
- **Unlocks**: LLMs (TinyLlama, Phi-1.5, Qwen-0.5B) at usable speeds + fitting in 4GB RAM. The single biggest "what edge AI is in 2026" lever.
- **Effort**: 2–3 weeks. NEON `vmlal_s8` + `vpadalq_s16` accumulator pattern (no SDOT on Cortex-A73), `is_aarch64_feature_detected!("dotprod")` fast-path for ARMv8.2+. Compiler-side: track int8 weight × f32 activation values, route to the right matmul, handle scale + zero-point.
- **Test**: numerical correctness vs f32 reference within tolerance + accuracy validation on a calibrated model.

## 6. ✅ RMSNorm + RoPE fused NIFs
- **Unlocks**: Llama/Mistral-architecture LLMs. Most modern LMs use RMSNorm (Llama, Mistral, Phi-2+, Qwen). RoPE is the position encoding for the same family.
- **Effort**: ~1 day each.
- **Test**: numerical correctness vs decomposed forms.

## 7. ✅ Sampling helpers + streaming generation API
- **Unlocks**: real chat/generation UX. Top-k, top-p, temperature, repetition penalty. `Stream.iterate/2` over token outputs.
- **Effort**: 2–3 days. Mostly Elixir; argmax + sort + slice underneath.
- **Test**: deterministic sampling at temperature=0; distribution sanity at temp > 0.

## 8. ✅ SafeTensors loader
- **Unlocks**: load any HF-published model directly without the host-side `term_to_binary` dance.
- **Effort**: 2–3 days. SafeTensors is a documented format (JSON header + raw binary), pure Elixir port is feasible.
- **Test**: load a known SafeTensors file, validate tensor data byte-for-byte against `torch.load` reference.

## 9. ✅ Image featurizer pipeline on Nerves
- **Unlocks**: real image input — camera → tensor — without bespoke per-app glue. JPEG decode → resize → normalize.
- **Effort**: 3–5 days. May need `Image`/`Vix`/`stb_image` cross-compile validation on Nerves.
- **Test**: input JPEG → tensor matches Bumblebee's `Image` reference output within tolerance.

## 10. ✅ NMS + anchor-box decoding
- **Unlocks**: YOLO and other detection models.
- **Effort**: 2–3 days. NMS is sorted top-K + IoU pairwise rejection. Anchor decoding is arithmetic.
- **Test**: NMS on hand-crafted boxes; full YOLO pipeline correctness check vs reference.

---

## Test discipline

Each item ships with:
1. Unit tests in `test/nx_arm/<op>_test.exs` covering correctness vs `Nx.BinaryBackend`.
2. Property-based tests where applicable (random shapes/values).
3. Benchmark in `bench/` showing perf vs the previous path.
4. The forward pass of at least one model that uses the feature, run end-to-end on the FP3+.

Tests run on host (x86_64) AND aarch64 cross-compile target. Failures on host block merge.
