# Erdős Minimum Overlap Problem — Improving the Upper Bound on C₅

## Research Question

Find a step function construction that yields a tighter (lower) upper bound on the constant C₅ in the Erdős minimum overlap problem, beating the current best of **0.380871** (Together AI, March 2026).

## Background

The minimum overlap problem was posed by Paul Erdős in 1955. Given any partition of {1, 2, …, 2n} into two equal sets A and B, how large must the maximum overlap max_k |A ∩ (B + k)| be?

### Continuous Formulation

C₅ is the largest constant such that for all non-negative f, g : [-1,1] → [0,1] with f + g = 1 on [-1,1] and ∫f = 1:

$$\sup_{x \in [-2,2]} \int_{-1}^1 f(t) g(x+t) \, dt \geq C_5$$

### Equivalent Step Function Formulation (Haugland 2016)

C₅ equals the infimum, over all step functions h : [0,2] → [0,1] with ∫₀² h(x) dx = 1, of:

$$\max_k \int h(x)(1 - h(x+k)) \, dx$$

In discrete form: given a sequence h₁, h₂, …, hₙ with:
- 0 ≤ hᵢ ≤ 1 for all i
- Σ hᵢ = n/2

The upper bound is:

$$C_5 \leq \max_k \frac{2}{n} \sum_i h_i (1 - h_{i+k})$$

Upper bounds on C₅ are obtained by constructing explicit step functions. More steps (larger n) allow finer constructions and potentially tighter bounds.

### Known Bounds

| Bound | Value | Source | Year |
|-------|-------|--------|------|
| Lower | 0.379005 | White (convex programming) | 2022 |
| Upper | 0.380927 | Haugland (51 steps) | 2016 |
| Upper | 0.380924 | AlphaEvolve (95 steps) | 2025 |
| Upper | 0.380876 | TTT-Discover (600 steps) | 2026 |
| Upper | **0.380871** | Together AI (512 steps) | 2026 |

The gap between lower and upper bounds is ~0.00187.

## Known Approaches

1. **Manual construction** (Haugland 2016): 51-step symmetric step function, hand-tuned.
2. **Evolutionary search** (AlphaEvolve 2025): LLM-guided evolutionary coding agent, 95 steps.
3. **Sequential linear programming** (TTT-Discover 2026, Together AI 2026): Optimize step function values via SLP starting from known good solutions, 512-600 steps.

## Success Criteria

- **Metric**: Upper bound on C₅ computed as max_k (2/n) Σᵢ hᵢ(1 - h_{i+k}) via cross-correlation
- **Direction**: Minimize (lower is better)
- **Target**: < 0.380871 (current SOTA)
- **Stretch target**: < 0.380800

## Solution Representation

A solution is a numpy array `h_values` of length n where:
- Each value is in [0, 1]
- The sum equals n/2
- The upper bound is computed via cross-correlation: `max(correlate(h, 1-h, 'full')) / n * 2`
