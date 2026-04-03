---
strategy: slp-1024
status: complete
eval_version: eval-v1
metric: 0.38087014293553945
issue: 2
parent: null
---

# SLP-1024 Optimization Log

## Approach

Sequential Linear Programming (SLP) with adaptive active-set selection, seeded from the Together AI 512/600-step baseline interpolated to 1024 steps.

## Key Algorithm Details

### 1. Seeding
- Loaded Together AI baseline (600 steps, obj=0.380870310586220)
- Interpolated to 1024 steps using `np.interp` on normalized x-axis
- Initial 1024-step objective: 0.381624 (degraded due to interpolation)

### 2. SLP Core Loop
At each iteration:
- Compute all cross-correlation overlaps: `get_all_overlaps(h)`
- Select all shifts within `1e-3` of the maximum (adaptive active-set, ~800-813 active shifts)
- Solve LP: minimize `t` subject to `grad_s @ h_lp - t <= grad_s @ h - F_s(h)` for each active shift s
- Apply line search (logspace alpha from 1.0 to 1e-8, then local refinement)
- Update h and repeat

### 3. Critical Fix: Analytic Gradient
Initial gradient computation was wrong (sign error for negative lags). The correct formula:

For `s = idx - (n-1)`:
- If `s >= 0`: `dF/dh_k = +(1-h[k-s])` for `k in [s,n-1]` and `-h[k+s]` for `k in [0,n-1-s]`
- If `s < 0` (t=-s): `dF/dh_k = +(1-h[k+t])` for `k in [0,n-1-t]` and `-h[k-t]` for `k in [t,n-1]`

Verified against finite differences.

### 4. Active-Set Size is Critical
Key discovery: using only top-K (5-10) shifts gives small improvements (~2e-5/iter) and gets stuck. Using ALL shifts within 1e-3 of maximum (~800 shifts) gives much larger jumps (~2e-4/iter) and can escape local minima.

## Optimization Trajectory

| Phase | Starting obj | Final obj | Iterations |
|-------|-------------|-----------|------------|
| Initial interpolation | — | 0.381624 | — |
| SLP with top-10 shifts | 0.381624 | 0.381148 | ~800 |
| SLP with 800+ active shifts | 0.381148 | 0.380933 | ~3 |
| Fine convergence | 0.380933 | 0.380870143 | ~100 |

## Results

- **Final metric**: 0.38087014293553945
- **Baseline metric**: 0.38087031058622
- **Improvement**: 1.68e-7 below baseline
- **Steps**: n=1024
- **Valid**: yes (sum=512, values in [0,1])

## Implementation Notes

- LP solver: scipy HiGHS with trust-region bounds `|h_lp - h| <= delta`
- delta starts at 0.05, adapts (increases on success, decreases on failure)
- Line search: logspace(0, -8, 40) + local refinement around best alpha
- ~10 seconds per LP iteration with 800+ active shifts

## Files

- `solution.py`: best h_values array (1024 elements)
- `optimize.py`: full optimization code
- `checkpoint.npy`: numpy checkpoint of best solution
