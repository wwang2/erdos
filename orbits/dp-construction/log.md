---
strategy: dp-construction
issue: 12
parent: null
---

# dp-construction: Combinatorial/Greedy/DP Approaches

## Problem
Minimize `max_k (2/n) Σ_i h_i(1 - h_{i+k})` via cross-correlation.
Constraints: `h_i ∈ [0,1]`, `Σh_i = n/2`.

## Reference
- Best known: **0.380870085514** (slp-4096, n=4096)

## Structural Analysis of Best Solution

Before implementing approaches, we analyzed the structure of slp-4096:

- **n = 4096** steps representing h : [0,2] → [0,1]
- **Block-of-4 symmetry**: consecutive groups of 4 steps have essentially identical values (max within-block diff < 3e-7), compressing effectively to 1024 unique values
- **Zero region**: ~first and last 10% of the interval are near-zero (h ≈ 0 for x ∈ [0, 0.20] ∪ [1.80, 2.00])
- **Active region**: spans roughly [0.20, 1.80] (about 80% of the interval)
- **Value distribution**: 28.9% near-zero, 13.4% near-one, 57.7% intermediate values
- **Highly irregular profile**: the active region has a complex, non-monotone profile (not a simple parametric shape)
- **Equioscillation**: the solution achieves the same maximum cross-correlation value at 10+ distinct shifts simultaneously — a hallmark of optimality

## Approach Results

### Approach 1: Greedy Level-Set Construction (n=512)
**Metric: 0.381835**

Sequential greedy: at each position i, choose h[i] from {0.0, 0.05, ..., 1.0} to minimize the running max cross-correlation, then fine-tune with L-BFGS-B. The naive greedy (choosing greedily in order 0..n-1) collapses (metric=1.0) because all budget is left to the end. The fix: resample from the best known solution as initialization, quantize to levels, then fine-tune.

- Raw greedy collapses (metric = 1.0) due to budget exhaustion
- Resampled initialization + L-BFGS-B: **0.381835**
- Quantized + fine-tuned: 0.382377
- Time: ~3s for n=512

### Approach 2: Block Construction + Differential Evolution (n=512, 32 blocks)
**Metric: 0.383002**

Divide [0,2] into 32 equal blocks, optimize constant block heights using scipy differential_evolution (popsize=20, maxiter=500), then fine-tune with L-BFGS-B. Seeded population from resampled best solution.

- DE converged to: 0.383085
- After L-BFGS-B polish: **0.383002**
- Time: ~30s

### Approach 3: Structural Analysis → Parametric Family (n=1024)
**Metric: 0.380870** (best)

Two sub-approaches:

**3a: Resample + Fine-tune (n=1024)**
Compress slp-4096 from n=4096 to n=1024 by taking every 4th value (exploiting block-of-4 structure), then fine-tune with L-BFGS-B:
- Compressed metric: 0.380870088248
- After L-BFGS-B (maxiter=15000): **0.380870086797**
- Time: ~4s

**3b: Symmetric Parametric DE (K=15 breakpoints)**
Parameterize h as a symmetric function: h(x) = f(min(x, 2-x)) for x ∈ [0,2], where f is piecewise-linear with K=15 breakpoints. Optimize with DE then L-BFGS-B:
- DE (K=15, 300 iterations): 0.382014
- After fine-tune: **0.381971**
- Time: ~40s

The parametric family is too constrained by the symmetric assumption — the true solution is not symmetric (the profile has complex non-monotone structure).

### Approach 4: Piecewise-Linear DE (n=512, 60 breakpoints)
**Metric: 0.385730**

Parameterize h as piecewise-linear with 60 breakpoints at evenly-spaced positions. Optimize breakpoint heights with DE (popsize=15, maxiter=500), convert to n=512 step function, then fine-tune.

- DE converged to: 0.386511
- After L-BFGS-B: **0.385730**
- Time: ~50s

## Key Findings

1. **None of the combinatorial/greedy approaches beat the record** (0.380870085514). The best we achieved was 0.380870086797 via resampling the existing best solution and fine-tuning — which is essentially starting from the known solution, not a new discovery.

2. **The greedy approach fundamentally fails** without a good initialization — the local cross-correlation objective used during construction does not provide enough global signal.

3. **Block/piecewise parameterizations** are heavily bottlenecked by the low-dimensional representation. The true solution has ~2952 distinct block transitions and cannot be captured by 32–60 parameters.

4. **The solution structure is highly irregular** — it is not a simple parametric family (not symmetric, not smooth, not a few-plateau function). This explains why continuous optimization (SLP) dominates: the true optimum is found by navigating a high-dimensional space with many near-degenerate local optima.

5. **Equioscillation is a key indicator** — the best solution has 10+ shifts all achieving the same cross-correlation maximum, suggesting the solution is at a saddle/critical point of the minimax problem. Future approaches should explicitly exploit this (e.g., via min-max optimization or chebyshev approximation).

6. **Larger n is better**: n=512 gives ~0.3818, n=1024 gives ~0.3809, n=4096 gives 0.38087. The metric improves with resolution.

## Best Solution
- **File**: `orbits/dp-construction/solution.py`
- **Method**: Resample slp-4096 (n=4096 → n=1024 via block-of-4 compression) + L-BFGS-B fine-tuning
- **n = 1024**
- **Metric = 0.380870086797** (gap of 1.28e-9 from best known)

## Recommendations for Future Work

- **Higher n**: Try n=8192 or n=16384 to get finer resolution
- **Min-max optimization**: Use subgradient methods or saddle-point formulations that explicitly minimize the maximum over all shifts
- **Equioscillation enforcement**: Force multiple shifts to achieve equal cross-correlation (exchange algorithm)
- **Coordinate descent on active shifts**: Identify the ~10 shifts achieving the maximum and optimize to equalize them
