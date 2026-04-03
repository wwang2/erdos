---
strategy: slp-1024-fourier
status: complete
eval_version: eval-v1
metric: 0.380870134092427
issue: 7
parent: slp-1024
---

# Fourier-Parameterized Optimization

## Approach

Instead of optimizing 1024 individual step values, parameterize the step function as:

  h_i = clip(0.5 + sum_{k=1}^{K} [a_k cos(2pi*k*i/n) + b_k sin(2pi*k*i/n)], 0, 1)

then rescale sum to n/2. This gives 2K parameters instead of 1024.

**Steps:**
1. Load slp-1024 solution (baseline: 0.380870134358437)
2. Extract K dominant Fourier coefficients via FFT
3. Optimize in Fourier space with L-BFGS-B (scipy)
4. Try K=50, 100, 200, 300
5. Multi-start: 20 random perturbations around best Fourier params (K=200)
6. SLP polish of best result

## Results by K

| K   | FFT reconstruction | L-BFGS-B optimized |
|-----|-------------------|---------------------|
| 50  | 0.384993257754    | 0.381250406392      |
| 100 | 0.382734691072    | 0.381105069524      |
| 200 | 0.381641230209    | 0.380939489386      |
| 300 | 0.381406935316    | 0.380944493197      |

SLP baseline (slp-1024): **0.380870134358437**

## Multi-start K=200

20 restarts with random perturbations around the FFT-fitted K=200 solution.
Best result: **0.380932470001205** (Run 0, FFT fit seed, 5000 L-BFGS-B iters).
Small perturbations (scale=0.001) converge to ~0.381, larger perturbations worse.
No restart beat the slp-1024 baseline.

## SLP Polish

SLP polish applied to slp-1024 solution (since no Fourier result beat it).
500 iterations, 813 active shifts per LP, ~10s/iter, ~5065s total.

Progress:
- Iter 20: 0.380870134325647
- Iter 100: 0.380870134231890
- Iter 200: 0.380870134170684
- Iter 300: 0.380870134134312
- Iter 400: 0.380870134108391
- Iter 500: 0.380870134092427

Total improvement from SLP polish: 0.380870134358437 → 0.380870134092427 (delta: ~2.7e-10).
Improvements are in the 10th-11th decimal place.

## Conclusion

The Fourier parameterization (K=50..300) could **not** beat the slp-1024 SLP solution.
Best Fourier result: 0.380939 (K=200) vs slp-1024 baseline: 0.380870.

**Reason**: The slp-1024 solution appears to be a sharp local minimum. Fourier 
representations with K=200 modes are smooth and miss the fine-grained structure 
that the SLP optimizer found. Increasing K makes reconstruction better but 
optimization harder. 20 random restarts in Fourier space also failed to improve.

The SLP polish on the slp-1024 solution itself produced a marginal improvement.

## Final Metric

0.380870134092427 (improvement of 2.7e-10 over slp-1024 parent via SLP polish)
