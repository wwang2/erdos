---
strategy: slp-2048-multistart
issue: 8
parent: slp-1024-multistart
---

# SLP 2048-step Multi-start

## Result

- **Metric**: 0.380870088324574
- **n_steps**: 2048
- **Valid**: true
- **sum**: 1024.0

Parent best (slp-1024-multistart, 1024 steps): 0.380870088424490

Improvement: 1.00e-10 (tiling to 2048 + SLP refinement)

## Strategy

1. Loaded slp-1024-multistart/solution.py (1024 steps, obj=0.380870088424490)
2. Tiled to 2048: `h_2048 = np.repeat(h_1024, 2)` — preserves cross-correlation objective exactly
3. Ran SLP with active shifts capped at 500 (LP tractability at n=2048), trust-region delta=0.05, line search
4. Early stopping when improvement < 1e-11 and delta < 1e-4
5. 25 perturbation restarts with scales [0.001, 0.003, 0.005, 0.01, 0.02] x 5

## Observations

- Tiled seed starts at exactly the same objective as the 1024-step solution (0.380870088424490)
- Initial SLP run converged in 158s (31 iterations), achieving 0.380870088324574
- All 25 perturbation restarts converged to higher values — the tiled seed is a strong local minimum
- The 2048-step solution is essentially the same structure as 1024-step, tiled with marginal refinement
- LP at n=2048 with 500 active shifts takes ~5-10s per iteration vs ~2s at n=1024

## Files

- `optimize.py` — SLP optimizer adapted for N=2048 with FFT-based correlation and early stopping
- `solution.py` — best h_values array (2048 elements, sum=1024)
- `checkpoint.npy` — numpy checkpoint of best solution
