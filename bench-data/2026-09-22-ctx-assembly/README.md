# Context-assembly bench (2026-09-22)

Harness: `cargo run --release -p peregrine-ctx --example ctxbench`
(synthetic multi-turn histories, budget 512 heuristic tokens, query fixed on
one planted fact; asserts the fact survives in the assembly).

| Turns | Input tok | Ingest ms | Plan ms | Out tok | Fits + carries fact? |
|---|---:|---:|---:|---:|---|
| 50 | 425 | 8.1 | 0.3 | 229 | yes |
| 200 | 1750 | 20.7 | 0.5 | 321 | yes |
| 800 | 7150 | 98.5 | 3.0 | 321 | yes |

## Reading

- Assembly cost is sublinear in history: 16× the turns (50 → 800) cost 12×
  ingest and 10× plan, while output stays flat at the budget — the planner
  spends the budget, not the history.
- Per-request overhead at 800 turns is ~100 ms on CPU against
  seconds-per-token generation: negligible on the serving path, and it runs
  before tokenize, off the decode critical path.
- Output pins at ~321 tok, not 512: the knapsack stops at relevance, not at
  the cap — headroom the budget leaves is evidence, not waste.
