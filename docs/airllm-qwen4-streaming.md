# AirLLM-style Qwen4 routed-expert streaming

Peregrine already has a demand-streaming lane for GLM-family routed experts. This change applies the same memory-first idea to the single-stream `qwen4_exp` path: the router chooses the expert union for the current forward, and the runtime loads only those experts from the existing SafeTensors index.

## What changes

With streaming disabled, Qwen4 keeps each layer's routed experts resident, as before. With streaming enabled, Qwen4:

1. Computes the normal softmax top-k routing and unique expert union.
2. Loads one selected expert's gate, up, and down projections from the checkpoint.
3. Computes that expert for the rows that selected it and scatters its weighted outputs in the same deterministic union order.
4. Drops the temporary expert weights before loading the next selected expert.
5. Adds the shared expert through the unchanged shared-gate path.

This bounds the *extra routed-expert weight residency* to roughly one expert at a time rather than the full expert bank. It does not evict all layer weights: attention and shared-expert weights, router data, residual mixers, and other non-routed parameters still follow the existing resident path. Actual peak RSS also includes activations, KV/recurrent state, allocator overhead, and I/O buffers.

## Enable it

Set the runtime environment variable before loading a model:

```bash
COLI_STREAM=1 cargo run --release -p peregrine-engine -- --help
```

For code that controls model loading directly, use the existing explicit loader:

```rust
let mut model = peregrine_model::Model::load_streaming(
    std::path::Path::new("/path/to/model"),
    true,
)?;
```

`COLI_STREAM=0` keeps routed experts resident. The regular `Model::load` path continues to make its existing memory-budget-based decision unless overridden by the environment.

## Tradeoff and current limits

The initial implementation prioritizes bounded memory, not maximum throughput. Selected experts are loaded on demand and computed sequentially in this Qwen4-specific path; it does not yet reuse Peregrine's generic routed-expert warm cache, predictive prefetcher, or concurrent I/O/CPU/GPU scheduler. A workload that repeatedly reads cold experts from a slow disk can therefore generate fewer tokens per second than resident inference. SSDs and the operating-system page cache can mitigate the cost, but results depend on the model layout and hardware.

This change does not add batched/concurrent Qwen4 serving, GPU execution, MTP, or token-tree support. Those are separate architectural paths and remain unsupported for `qwen4_exp`.

## Correctness check

The Qwen4 tiny end-to-end test now runs the same prompt with resident and demand-streamed loading and asserts that greedy output token IDs match:

```bash
cargo test -p peregrine-model qwen4_tiny_model_loads_and_decodes_deterministically
cargo test -p peregrine-model qwen4
cargo test --workspace
```

The implementation intentionally preserves the top-k router, expert-union construction, and output scatter order. The new test is a regression gate, not a benchmark; throughput and peak RSS should be measured on the actual checkpoint before choosing the streaming mode for production.
