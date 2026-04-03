---
strategy: slp-1024-multistart
status: complete
eval_version: eval-v1
metric: 0.38087008842448966
issue: 6
parent: slp-1024
---

# SLP Multi-Start Optimization (n=1024)

## Summary

Ran SLP optimization from 15 diverse starting points to escape local optima of slp-1024.
Best result: **0.38087008842448966** from `perturbed_sota_scale=0.02`, beating the parent orbit
slp-1024 (0.380870134358437) by ~4.6e-8.

Key finding: only small perturbations (scale=0.01, 0.02) of the SOTA solution beat the parent.
Larger perturbations and all non-SOTA starts (random, block, alternating, sine, reversed, shuffled)
converged to worse local optima in the 0.3808–0.3814 range.

## Configuration

- n = 1024, n/2 = 512
- Active shift threshold: 1e-3 of maximum overlap
- Max active shifts per LP: 1000
- Iterations per start: 300 (with early stopping on no improvement)
- LP solver: HiGHS (scipy.optimize.linprog)
- Trust region: adaptive delta (0.05 initial, range [1e-6, 0.5])

## Results Table

| # | Starting Point | Init Obj | Final Obj | Beat SOTA? |
|---|---------------|----------|-----------|------------|
| 1 | perturbed_sota_scale=0.01 | 0.382278805447401 | 0.380870117892734 | yes |
| 2 | perturbed_sota_scale=0.02 | 0.383613300881958 | **0.380870088424490** | yes (best) |
| 3 | perturbed_sota_scale=0.05 | 0.388409219081628 | 0.380870300604153 | no |
| 4 | perturbed_sota_scale=0.1 | 0.396151426391481 | 0.380870344885536 | no |
| 5 | perturbed_sota_scale=0.2 | 0.412083810197952 | 0.380871466824518 | no |
| 6 | random_valid_seed=42 | 0.508061594601907 | 0.381142287420043 | no |
| 7 | random_valid_seed=43 | 0.505218272473885 | 0.381039037386572 | no |
| 8 | random_valid_seed=44 | 0.507700601571043 | 0.381368815167032 | no |
| 9 | random_valid_seed=45 | 0.504284887110635 | 0.380996553341050 | no |
| 10 | random_valid_seed=46 | 0.510195275050783 | 0.381148092323891 | no |
| 11 | block_construction | 1.000000000000000 | 0.381057552886908 | no |
| 12 | alternating_0.3_0.7 | 0.579824218750000 | 0.381157172357638 | no |
| 13 | smooth_sine | 0.704241230944980 | 0.381489250333688 | no |
| 14 | reversed_sota | 0.380870134358437 | 0.380870134137617 | marginal |
| 15 | shuffled_sota | 0.517861850813575 | 0.381227678247627 | no |

SOTA baseline (slp-1024): 0.380870134358437

## Evaluator Verification

```json
{
  "valid": true,
  "errors": [],
  "n_steps": 1024,
  "metric": 0.38087008842448966,
  "sum": 512.0
}
```

## Observations

- SLP convergence from large perturbations (scale >= 0.05) is slow and lands in worse basins.
- The reversed_sota start essentially stayed at the SOTA value (very slow improvement, same basin).
- Random/block/alternating/sine starts all converge to ~0.381, suggesting the good basin requires
  proximity to the SOTA solution structure.
- The scale=0.02 perturbed start found a slightly better optimum than scale=0.01, suggesting
  moderate noise helps escape micro-local-optima while staying in the good basin.
- Active-set size with threshold 1e-3 was ~810-820 shifts for near-SOTA starts, indicating
  a very flat landscape near the optimum with many nearly-tied shifts.
