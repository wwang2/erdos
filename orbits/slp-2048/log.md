---
strategy: slp-2048
status: complete
eval_version: eval-v1
metric: 0.3808701340635162
issue: 5
parent: slp-1024
---

## Summary

SLP optimization at n=2048, seeded from slp-1024 solution via tiling (each element repeated twice). This preserves the exact metric from slp-1024 as the starting point.

## Key Findings

- **Tiling seed**: Repeating each element of the n=1024 solution twice gives exactly the same metric (0.380870134358437) at n=2048, because the cross-correlation structure is preserved.
- **Interpolation** (linear) gave a worse starting point (0.381358) and converged to 0.380892 — worse than the parent.
- **SLP refinement** from the tiled seed found marginal improvement: 0.380870134358437 → 0.380870134063516 (improvement ~3e-10).
- **20 perturbation restarts** (scales 0.0005–0.01) all converged to worse local optima (0.38088–0.38092), confirming the tiled basin is the best found.

## Approach

1. Load slp-1024 solution (n=1024, metric=0.380870134)
2. Tile to n=2048 by `np.repeat(h1024, 2)` — starting metric exactly matches parent
3. Run SLP with 500 active shifts, trust-region delta=0.05, early-stop at improvement < 1e-6
4. Initial run converged in 6 iterations (improvements reached numerical precision ~1e-11)
5. 20 perturbation restarts with small scales (0.0005–0.01) — none beat the tiled optimum

## Parameters

- n = 2048
- Active shift cap: 500 (out of 4095 possible)
- Trust-region delta: 0.05 → adapts up to 0.5
- LP solver: HiGHS via scipy.optimize.linprog
- Early convergence: improvement < 1e-6 after iter 20
- Restarts: 20, scales [0.003, 0.001, 0.005, 0.002, 0.008, 0.0005, 0.01, ...]

## Result

- **metric**: 0.3808701340635162
- n_steps: 2048, sum: 1024.0, valid: true
- Marginal improvement over parent (0.380870134358437): ~3e-10
