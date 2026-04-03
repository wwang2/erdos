"""
Combinatorial/greedy/DP approaches to the Erdős Minimum Overlap problem.

Minimize: max_k (2/n) Σ_i h_i(1 - h_{i+k}) via cross-correlation.
Valid solution: numpy array h_values, 0 ≤ h_i ≤ 1, Σh_i = n/2.

Four approaches:
1. Greedy level-set construction
2. Block construction + scipy differential_evolution
3. Structural analysis of best solution → parametric family optimization
4. Genetic/evolutionary on piecewise-linear parameterization
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_metric(h: np.ndarray) -> float:
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return float(np.max(conv) / n * 2)


def validate(h: np.ndarray, atol: float = 1e-6) -> bool:
    n = len(h)
    if np.any(h < -atol) or np.any(h > 1 + atol):
        return False
    return np.isclose(np.sum(h), n / 2.0, atol=atol)


def normalize_sum(h: np.ndarray) -> np.ndarray:
    """Rescale h so that sum = n/2, clipping to [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    n = len(h)
    if s == 0:
        return np.full(n, 0.5)
    return np.clip(h * (n / 2.0) / s, 0.0, 1.0)


def project_to_feasible(h: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Project h onto {0 ≤ h_i ≤ 1, sum = n/2} using Dykstra's algorithm."""
    n = len(h)
    target = n / 2.0
    h = h.copy()
    for _ in range(max_iter):
        # Project onto [0, 1]^n
        h = np.clip(h, 0.0, 1.0)
        # Project onto sum = target
        h = h + (target - h.sum()) / n
        if np.isclose(h.sum(), target, atol=1e-9) and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
            break
    return np.clip(h, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Approach 1: Greedy level-set construction
# ---------------------------------------------------------------------------

def greedy_level_set(n: int = 512, verbose: bool = True) -> np.ndarray:
    """
    Build h one value at a time. At each step i, choose h[i] from a grid of
    candidate values to minimize the running maximum cross-correlation.

    Strategy: maintain partial cross-correlations for all shifts, pick h[i]
    that minimises the increase in the running max.
    """
    if verbose:
        print(f"\n=== Approach 1: Greedy level-set construction (n={n}) ===")

    t0 = time.time()
    target_sum = n / 2.0

    # Discretize h values
    n_levels = 21  # 0.0, 0.05, ..., 1.0
    levels = np.linspace(0.0, 1.0, n_levels)

    h = np.zeros(n)
    # partial[k] = sum_{i<current} h[i] * (1 - h[(i+k) % n]) for k=0..n-1
    # We track partial contributions; shift k means (i+k) mod n
    # For non-circular: just track for all lags k = -(n-1)..+(n-1)
    # Use cross-correlation prefix sums approach

    # At step i, for each lag k, the correlation contribution so far is:
    # C[k] = sum_{j=0}^{i-1} h[j] * (1 - h[j+k])  for j+k in [0, i-1]
    # This is O(n) per step if we maintain it incrementally.

    # Simpler greedy: use a reduced n for tractability
    # At each position i, try each level; compute the cross-correlation of h[:i+1]
    # with 1-h[:i+1] using FFT; pick the level that gives smallest max.

    # For speed, use FFT-based evaluation on partial arrays
    best_h = None
    best_metric = float('inf')

    # Do multiple random restarts with greedy
    for trial in range(3):
        np.random.seed(trial * 42)
        h = np.zeros(n)
        remaining_sum = target_sum

        for i in range(n):
            remaining = n - i
            # Feasibility: we need remaining_sum remaining over `remaining` slots
            # h[i] must be in [max(0, remaining_sum - (remaining-1)), min(1, remaining_sum)]
            lo = max(0.0, remaining_sum - (remaining - 1))
            hi = min(1.0, remaining_sum)
            feasible_levels = levels[(levels >= lo - 1e-9) & (levels <= hi + 1e-9)]
            if len(feasible_levels) == 0:
                feasible_levels = np.array([np.clip(remaining_sum / remaining, 0.0, 1.0)])

            best_choice = feasible_levels[0]
            best_local = float('inf')

            for v in feasible_levels:
                h[i] = v
                # Evaluate cross-correlation on h[:i+1]
                partial = h[:i + 1]
                conv = np.correlate(partial, 1 - partial, mode='full')
                # Normalize prospectively by n
                local_max = float(np.max(conv)) / n * 2
                if local_max < best_local:
                    best_local = local_max
                    best_choice = v

            h[i] = best_choice
            remaining_sum -= best_choice

        metric = compute_metric(h)
        if verbose:
            print(f"  Trial {trial}: metric = {metric:.12f}")
        if metric < best_metric:
            best_metric = metric
            best_h = h.copy()

    if verbose:
        print(f"  Best metric: {best_metric:.12f}  (time: {time.time()-t0:.1f}s)")
    return best_h


# ---------------------------------------------------------------------------
# Approach 2: Block construction + differential evolution
# ---------------------------------------------------------------------------

def block_construction_de(n: int = 512, n_blocks: int = 32, verbose: bool = True) -> np.ndarray:
    """
    Divide [0,1] (representing h) into `n_blocks` blocks of equal size.
    Within each block h is constant. Optimize block heights using
    differential_evolution, then fine-tune with L-BFGS-B.
    """
    if verbose:
        print(f"\n=== Approach 2: Block construction + DE (n={n}, blocks={n_blocks}) ===")

    t0 = time.time()
    block_size = n // n_blocks

    def params_to_h(params):
        # params: n_blocks heights in [0,1], will be scaled to meet sum constraint
        raw = np.clip(params, 0.0, 1.0)
        h = np.repeat(raw, block_size)
        if len(h) < n:
            h = np.append(h, np.full(n - len(h), raw[-1]))
        h = h[:n]
        # Scale to meet sum = n/2
        s = np.sum(h)
        if s < 1e-10:
            h = np.full(n, 0.5)
        else:
            h = h * (n / 2.0) / s
            h = np.clip(h, 0.0, 1.0)
        return h

    def objective(params):
        h = params_to_h(params)
        return compute_metric(h)

    bounds = [(0.0, 1.0)] * n_blocks

    if verbose:
        print(f"  Running differential_evolution (maxiter=500, popsize=15)...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=500,
        popsize=15,
        tol=1e-8,
        seed=42,
        workers=1,
        polish=True,
        mutation=(0.5, 1.5),
        recombination=0.9,
        disp=verbose,
    )

    h_de = params_to_h(result.x)
    metric_de = compute_metric(h_de)
    if verbose:
        print(f"  DE metric: {metric_de:.12f}")

    # Fine-tune with L-BFGS-B on the full n-dimensional problem
    if verbose:
        print(f"  Fine-tuning with L-BFGS-B on full {n}-dim problem...")

    def obj_full(x):
        h = project_to_feasible(x)
        return compute_metric(h)

    result2 = minimize(
        obj_full,
        h_de,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * n,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
    )
    h_lbfgs = project_to_feasible(result2.x)
    metric_lbfgs = compute_metric(h_lbfgs)

    if verbose:
        print(f"  L-BFGS-B metric: {metric_lbfgs:.12f}  (time: {time.time()-t0:.1f}s)")

    return h_lbfgs if metric_lbfgs < metric_de else h_de


# ---------------------------------------------------------------------------
# Approach 3: Parametric family from structural analysis
# ---------------------------------------------------------------------------

def parametric_from_structure(n: int = 1024, verbose: bool = True) -> np.ndarray:
    """
    From structural analysis of slp-4096:
    - Solution is zero in ~[0, 0.20] and ~[1.80, 2.00]  (about 10% each side)
    - Active region spans ~[0.20, 1.80]
    - Within active region, shape is roughly bell-shaped / trapezoidal
    - Values cluster near 0 (sparse), intermediate, and near 1
    - The block-of-4 symmetry suggests n/4 free parameters

    Parametric family: h(x) for x in [0,2] defined as:
      - 0            for x in [0, a] ∪ [2-a, 2]
      - f(x)         for x in [a, 1] where f is a piecewise polynomial
      - f(2-x)       for x in [1, 2-a]  (symmetric)

    We optimize parameters (a, shape params) using DE.
    """
    if verbose:
        print(f"\n=== Approach 3: Parametric family from structural analysis (n={n}) ===")

    t0 = time.time()

    # From the best solution analysis:
    # - Zero region: ~first 10% and last 10%
    # - Active region shape: starts low (~0.04), rises, peaks around 0.5, falls
    # - But within the active region there's complex structure

    # Parametric family 1: Symmetric step function with profile
    # Parameters: a (zero-region fraction), and K heights for a piecewise linear profile

    K = 20  # number of breakpoints in half the active region

    def params_to_h(params):
        # params[0]: a = zero region fraction (0.05 to 0.25)
        # params[1:K+1]: heights at K evenly-spaced points in [a, 1.0]
        a_frac = float(np.clip(params[0], 0.05, 0.40))
        heights = np.clip(params[1:K + 1], 0.0, 1.0)

        # Build full h by interpolating
        x = np.linspace(0, 2, n, endpoint=False) + 1.0 / n  # midpoints of steps

        # Active region: [a_frac, 2-a_frac]
        # Symmetric profile: define on [a_frac, 1], mirror for [1, 2-a_frac]
        h = np.zeros(n)

        # Breakpoints in [a_frac, 1.0]
        bp = np.linspace(a_frac, 1.0, K + 2)  # K+2 points including endpoints
        # heights at breakpoints: 0 at a_frac, then heights[0..K-1], then heights[-1] at 1.0
        bp_heights = np.concatenate([[0.0], heights, [heights[-1]]])

        for i in range(n):
            xi = x[i]
            # Mirror: use symmetry around x=1
            xi_sym = xi if xi <= 1.0 else 2.0 - xi
            if xi_sym < a_frac:
                h[i] = 0.0
            else:
                h[i] = float(np.interp(xi_sym, bp, bp_heights))

        # Normalize
        s = np.sum(h)
        if s < 1e-10:
            return np.full(n, 0.5)
        h = h * (n / 2.0) / s
        return np.clip(h, 0.0, 1.0)

    def objective(params):
        h = params_to_h(params)
        return compute_metric(h)

    bounds = [(0.05, 0.35)] + [(0.0, 1.0)] * K

    if verbose:
        print(f"  Running DE on {K+1}-parameter symmetric profile...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=1000,
        popsize=20,
        tol=1e-9,
        seed=7,
        polish=True,
        mutation=(0.5, 1.5),
        recombination=0.9,
        disp=False,
    )

    h_best = params_to_h(result.x)
    metric = compute_metric(h_best)
    if verbose:
        print(f"  Parametric DE metric: {metric:.12f}")

    # Now try seeding from actual best solution structure
    if verbose:
        print(f"  Seeding from slp-4096 structure...")

    ns_sol = {}
    exec(open('orbits/slp-4096/solution.py').read(), ns_sol)
    h_ref = np.asarray(ns_sol['h_values'])
    n_ref = len(h_ref)

    # Resample to n
    x_ref = np.linspace(0, 2, n_ref, endpoint=False) + 1.0 / n_ref
    x_new = np.linspace(0, 2, n, endpoint=False) + 1.0 / n
    h_resampled = np.interp(x_new, x_ref, h_ref)
    h_resampled = project_to_feasible(h_resampled)
    metric_resampled = compute_metric(h_resampled)
    if verbose:
        print(f"  Resampled slp-4096 at n={n}: metric = {metric_resampled:.12f}")

    if metric_resampled < metric:
        h_best = h_resampled
        metric = metric_resampled

    # Fine-tune resampled solution
    if verbose:
        print(f"  Fine-tuning resampled solution with L-BFGS-B...")

    def obj_full(x):
        h = project_to_feasible(x)
        return compute_metric(h)

    result2 = minimize(
        obj_full,
        h_resampled,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * n,
        options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-9},
    )
    h_ft = project_to_feasible(result2.x)
    metric_ft = compute_metric(h_ft)
    if verbose:
        print(f"  Fine-tuned metric: {metric_ft:.12f}  (time: {time.time()-t0:.1f}s)")

    if metric_ft < metric:
        return h_ft
    return h_best


# ---------------------------------------------------------------------------
# Approach 4: Genetic/evolutionary on piecewise-linear parameterization
# ---------------------------------------------------------------------------

def piecewise_linear_de(n: int = 1024, n_breakpoints: int = 100, verbose: bool = True) -> np.ndarray:
    """
    Parameterize h as piecewise linear with n_breakpoints breakpoints.
    Optimize breakpoint heights using differential_evolution.
    Convert to step function and evaluate.
    """
    if verbose:
        print(f"\n=== Approach 4: Piecewise-linear DE (n={n}, breakpoints={n_breakpoints}) ===")

    t0 = time.time()

    # Breakpoints at evenly-spaced positions
    bp_positions = np.linspace(0, n - 1, n_breakpoints)

    def params_to_h(heights):
        heights = np.clip(heights, 0.0, 1.0)
        # Interpolate to full n
        x = np.arange(n, dtype=float)
        h = np.interp(x, bp_positions, heights)
        # Project to feasible
        h = project_to_feasible(h)
        return h

    def objective(heights):
        h = params_to_h(heights)
        return compute_metric(h)

    bounds = [(0.0, 1.0)] * n_breakpoints

    if verbose:
        print(f"  Running DE (maxiter=800, popsize=15)...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=800,
        popsize=15,
        tol=1e-9,
        seed=99,
        polish=True,
        mutation=(0.5, 1.5),
        recombination=0.9,
        disp=False,
        workers=1,
    )

    h_de = params_to_h(result.x)
    metric_de = compute_metric(h_de)
    if verbose:
        print(f"  DE metric: {metric_de:.12f}")

    # Fine-tune on full n-dim
    if verbose:
        print(f"  Fine-tuning with L-BFGS-B on n={n}...")

    def obj_full(x):
        h = project_to_feasible(x)
        return compute_metric(h)

    result2 = minimize(
        obj_full,
        h_de,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * n,
        options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-9},
    )
    h_ft = project_to_feasible(result2.x)
    metric_ft = compute_metric(h_ft)

    if verbose:
        print(f"  L-BFGS-B metric: {metric_ft:.12f}  (time: {time.time()-t0:.1f}s)")

    return h_ft if metric_ft < metric_de else h_de


# ---------------------------------------------------------------------------
# Approach 3b: Direct optimization on large n using best solution as seed
# ---------------------------------------------------------------------------

def direct_optimize_large(n: int = 2048, verbose: bool = True) -> np.ndarray:
    """
    Resample the slp-4096 solution to n=2048 and fine-tune aggressively
    with L-BFGS-B. This is the most direct path to beating 0.380870.
    """
    if verbose:
        print(f"\n=== Approach 3b: Direct optimization seeded from slp-4096 (n={n}) ===")

    t0 = time.time()

    ns_sol = {}
    exec(open('orbits/slp-4096/solution.py').read(), ns_sol)
    h_ref = np.asarray(ns_sol['h_values'])
    n_ref = len(h_ref)

    # Resample
    x_ref = np.arange(n_ref, dtype=float)
    x_new = np.linspace(0, n_ref - 1, n)
    h_resampled = np.interp(x_new, x_ref, h_ref)
    h_resampled = project_to_feasible(h_resampled)
    metric_init = compute_metric(h_resampled)
    if verbose:
        print(f"  Resampled metric: {metric_init:.12f}")

    best_h = h_resampled.copy()
    best_metric = metric_init

    # Multiple restarts of L-BFGS-B with random perturbations
    for trial in range(5):
        if trial == 0:
            h_start = h_resampled.copy()
        else:
            np.random.seed(trial * 13)
            noise = np.random.normal(0, 0.02, n)
            h_start = project_to_feasible(h_resampled + noise)

        def obj(x):
            h = project_to_feasible(x)
            return compute_metric(h)

        result = minimize(
            obj,
            h_start,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * n,
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10},
        )
        h_opt = project_to_feasible(result.x)
        m = compute_metric(h_opt)
        if verbose:
            print(f"  Trial {trial}: metric = {m:.12f}")
        if m < best_metric:
            best_metric = m
            best_h = h_opt.copy()

    if verbose:
        print(f"  Best metric: {best_metric:.12f}  (time: {time.time()-t0:.1f}s)")
    return best_h


# ---------------------------------------------------------------------------
# Main: run all approaches and pick the best
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json
    from pathlib import Path

    BEST_KNOWN = 0.380870086

    results = {}
    best_h = None
    best_metric = float('inf')

    # --- Approach 1: Greedy (n=512, fast) ---
    h1 = greedy_level_set(n=512, verbose=True)
    m1 = compute_metric(h1)
    results['approach1_greedy'] = {'n': 512, 'metric': m1}
    print(f"  => Approach 1 final: {m1:.12f}")
    if m1 < best_metric:
        best_metric = m1
        best_h = h1

    # --- Approach 2: Block DE (n=512, blocks=32) ---
    h2 = block_construction_de(n=512, n_blocks=32, verbose=True)
    m2 = compute_metric(h2)
    results['approach2_block_de'] = {'n': 512, 'metric': m2}
    print(f"  => Approach 2 final: {m2:.12f}")
    if m2 < best_metric:
        best_metric = m2
        best_h = h2

    # --- Approach 3: Parametric from structure (n=1024) ---
    h3 = parametric_from_structure(n=1024, verbose=True)
    m3 = compute_metric(h3)
    results['approach3_parametric'] = {'n': 1024, 'metric': m3}
    print(f"  => Approach 3 final: {m3:.12f}")
    if m3 < best_metric:
        best_metric = m3
        best_h = h3

    # --- Approach 4: Piecewise-linear DE (n=512, bp=60) ---
    h4 = piecewise_linear_de(n=512, n_breakpoints=60, verbose=True)
    m4 = compute_metric(h4)
    results['approach4_piecewise_de'] = {'n': 512, 'metric': m4}
    print(f"  => Approach 4 final: {m4:.12f}")
    if m4 < best_metric:
        best_metric = m4
        best_h = h4

    # --- Approach 3b: Direct large-n optimization (n=2048) ---
    h3b = direct_optimize_large(n=2048, verbose=True)
    m3b = compute_metric(h3b)
    results['approach3b_direct_large'] = {'n': 2048, 'metric': m3b}
    print(f"  => Approach 3b final: {m3b:.12f}")
    if m3b < best_metric:
        best_metric = m3b
        best_h = h3b

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL APPROACHES")
    print("=" * 60)
    for name, info in results.items():
        marker = " *BEST*" if info['metric'] == best_metric else ""
        beat = " [BEATS RECORD!]" if info['metric'] < BEST_KNOWN else ""
        print(f"  {name}: n={info['n']}, metric={info['metric']:.12f}{marker}{beat}")
    print(f"\nBest overall: {best_metric:.12f}")
    print(f"Best known:   {BEST_KNOWN:.12f}")
    if best_metric < BEST_KNOWN:
        print(f"IMPROVEMENT: {BEST_KNOWN - best_metric:.2e}")
    else:
        print(f"Gap to record: {best_metric - BEST_KNOWN:.2e}")

    # Save best solution
    out_dir = Path(__file__).parent
    solution_path = out_dir / 'solution.py'

    # Save as solution.py
    lines = ['import numpy as np', '', 'h_values = np.array([']
    for v in best_h:
        lines.append(f'  np.float64({repr(float(v))}),')
    lines.append('])')
    solution_path.write_text('\n'.join(lines) + '\n')
    print(f"\nSaved best solution to {solution_path}")
    print(f"Solution n={len(best_h)}, metric={best_metric:.12f}")

    # Save results log
    results_path = out_dir / 'results.json'
    results_path.write_text(json.dumps({
        'best_known': BEST_KNOWN,
        'best_metric': best_metric,
        'improvement': BEST_KNOWN - best_metric,
        'approaches': results,
    }, indent=2))
    print(f"Saved results to {results_path}")
