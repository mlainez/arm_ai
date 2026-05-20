# Buffer pool design (Phase 3, deferred)

## Status

Documented; not implemented as of 2026-05-21. The earlier in-place
NIF refactor (commit 3fb085e) and the bias-add NIF (commit 49250f9)
already removed the highest-cost per-op allocations. A full arena
would yield ~5–10% more on warm forwards, but it's a substantial
multi-week refactor touching every NIF and the backend struct.

## Sketch

Per-defn-invocation arena: at `Nx.Defn.compile` time, walk the
Expr graph; for each Expr node compute the output byte size
(known from shape + dtype). Build an "alloc plan":

  * Run liveness analysis: for each id, find the last consumer.
  * Greedy bin-packing of node lifetimes into byte ranges of a
    single large arena allocation.
  * At invocation, allocate one big `Vec<u8>` (or mmap region);
    pass byte offsets to each NIF.

## Why it's not done

1. Every NIF needs a "write into this offset" signature variant.
   We've done that for elementwise/unary/scalar; the heavier NIFs
   (matmul, batched_matmul, broadcast, transpose, concatenate)
   still alloc internally.
2. The Erlang term <-> Rust buffer handoff goes through
   `OwnedBinary`, which BEAM controls. Sharing a memory region
   across NIF calls means going outside the `OwnedBinary` model —
   either with `ResourceArc<MutexGuard<Vec<u8>>>` (clunky) or with
   a custom memory-mapped region.
3. Most of the per-op overhead we measured wasn't the alloc — it
   was Nx-level wrapper calls (`Nx.broadcast` cost 50–170 ms via
   shape inference, totally unrelated to NIF allocs). The bias-add
   NIF + `ensure_on_arm` fixes addressed the real bottlenecks.

## When to revisit

If a future profile shows ≥10% of total forward time in
allocator-attributable work (Vec drops, OwnedBinary news, BEAM
GC churn), pick up this design.
