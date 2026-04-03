---
strategy: slp-4096
status: complete
eval_version: eval-v1
metric: 0.3808700855139993
issue: 10
parent: slp-1024-multistart
---

# SLP Optimization (n=4096)

## Summary

Tiled the slp-1024-multistart best solution (0.380870088424490) to 4096 steps via `np.repeat(h_1024, 4)`.
Ran SLP with reduced LP cost: MAX_ACTIVE=300 shifts, time_limit=300s, 20-alpha line search, 150 iters/run + 8 perturbation restarts (scales 0.005, 0.01, 0.02).

**Result: 0.3808700855139993** — beats parent slp-1024-multistart (0.380870088424490) by ~2.9e-9.

## Configuration

- n = 4096, n/2 = 2048
- Seed: slp-1024-multistart tiled x4 (`np.repeat(h_1024, 4)`)
- Active shift cap: 300 (vs 1000 at n=1024)
- LP time_limit: 300s
- Line search: 20 alphas (logspace 1→1e-8) + 10 refinement
- Iterations per run: 150
- Perturbation restarts: 8 (scales: 0.005, 0.01, 0.02 cycling)
- LP solver: HiGHS (scipy.optimize.linprog)
- Trust region: adaptive delta (0.05 initial, range [1e-6, 0.5])

## Timing

- Initial run: 150 iters in 2138.7s (~14s/iter average, 1-40s range)
- Restart 1 (scale=0.005): 150 iters in 659.2s (~4.4s/iter) → 0.380922087982884 (no improvement)
- Restart 2 (scale=0.01): → 0.380962673780201 (no improvement)
- Restart 3 (scale=0.02): → 0.381061110522113 (no improvement)
- Restart 4 (scale=0.005): → 0.380923262751374 (no improvement)
- Restart 5 (scale=0.01): → 0.380966 (no improvement)
- Restart 6 (scale=0.02): → 0.380926658952063 (no improvement)
- Restart 7 (scale=0.005): → 0.380926 (no improvement)
- Restart 8 (scale=0.01): 150 iters in 2838.0s (~19s/iter) → 0.380969500352483 (no improvement)

## Results Table

| Run | Starting Point | Init Obj | Final Obj | Beat Best? |
|-----|---------------|----------|-----------|------------|
| 0 | tiled x4 from slp-1024-multistart | 0.380870088424490 | **0.380870085513999** | yes (best) |
| 1 | perturbed scale=0.005 | 0.381617807 | 0.380922087982884 | no |
| 2 | perturbed scale=0.010 | 0.382694910 | 0.380962673780201 | no |
| 3 | perturbed scale=0.020 | ~0.384 | 0.381061110522113 | no |
| 4 | perturbed scale=0.005 | 0.381766630 | 0.380923262751374 | no |
| 5 | perturbed scale=0.010 | 0.382382740 | ~0.380966 | no |
| 6 | perturbed scale=0.020 | ~0.384 | 0.380926658952063 | no |
| 7 | perturbed scale=0.005 | ~0.382 | ~0.380926 | no |
| 8 | perturbed scale=0.010 | 0.382744776 | 0.380969500352483 | no |

## Evaluator Verification

```json
{
  "valid": true,
  "errors": [],
  "n_steps": 4096,
  "metric": 0.3808700855139993,
  "sum": 2048.0
}
```

## Observations

- Tiling x4 preserves the cross-correlation structure and starts from an excellent basin (0.380870088).
- The initial SLP run improved the metric by ~2.9e-9 to 0.380870085513999.
- All 8 perturbation restarts converged to worse basins (0.38092-0.38107), confirming the tiled
  seed was already in the optimal basin.
- LP per-iteration time varied widely: 1-40s in the initial run (delta growth causes wider trust
  region, slowing convergence at higher deltas), 4-30s in restarts.
- Active shifts capped at 300 throughout — the landscape is uniformly flat at n=4096.
- Improvement diminished rapidly after iter 50 (1e-11 scale vs 1e-10 at start), suggesting this
  basin is close to a local optimum at this resolution.
- Higher resolution (4096 vs 1024) yields marginal additional improvement (~2.9e-9 total),
  suggesting the continuous limit is being approached.
