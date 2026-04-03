"""
Run all four combinatorial/greedy/DP approaches and save the best solution.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution, minimize

OUT_DIR = Path(__file__).parent
BEST_KNOWN = 0.380870086


def compute_metric(h):
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return float(np.max(conv) / n * 2)


def project(h, max_iter=400):
    n = len(h)
    target = n / 2.0
    h = h.copy()
    for _ in range(max_iter):
        h = np.clip(h, 0.0, 1.0)
        h = h + (target - h.sum()) / n
        if abs(h.sum() - target) < 1e-9:
            break
    return np.clip(h, 0.0, 1.0)


def obj(x):
    return compute_metric(project(x))


def lbfgsb(h0, maxiter=5000):
    n = len(h0)
    r = minimize(obj, h0, method='L-BFGS-B', bounds=[(0, 1)] * n,
                 options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11})
    return project(r.x)


def load_best():
    ns = {}
    exec((Path(__file__).parent.parent / 'slp-4096' / 'solution.py').read_text(), ns)
    return np.asarray(ns['h_values'])


# -----------------------------------------------------------------------
# Approach 1: Greedy level-set (n=512)
# Random-order greedy: assign h[i] to minimise running max cross-correlation
# -----------------------------------------------------------------------
def approach1_greedy(n=512):
    print("\n=== Approach 1: Greedy level-set (n={}) ===".format(n), flush=True)
    t0 = time.time()
    levels = np.linspace(0, 1, 21)
    target = n / 2.0

    best_h, best_m = None, float('inf')
    for seed in range(8):
        np.random.seed(seed)
        h = np.zeros(n)
        remaining = target
        order = np.arange(n)  # sequential (random order degrades results)

        for step, i in enumerate(order):
            left = n - step
            lo = max(0.0, remaining - (left - 1))
            hi = min(1.0, remaining)
            feas = levels[(levels >= lo - 1e-9) & (levels <= hi + 1e-9)]
            if len(feas) == 0:
                feas = np.array([np.clip(remaining / max(left, 1), 0, 1)])

            bv, bm = feas[0], float('inf')
            for v in feas:
                h[i] = v
                m = compute_metric(h)
                if m < bm:
                    bm, bv = m, v
            h[i] = bv
            remaining -= bv

        h = project(h)
        m = compute_metric(h)
        print(f"  seed={seed}: raw={m:.8f}", flush=True)

        # Fine-tune
        h2 = lbfgsb(h, maxiter=2000)
        m2 = compute_metric(h2)
        print(f"  seed={seed}: fine-tuned={m2:.12f}", flush=True)
        if m2 < best_m:
            best_m, best_h = m2, h2.copy()

    print(f"  BEST approach 1: {best_m:.12f}  (t={time.time()-t0:.1f}s)", flush=True)
    return best_h, best_m


# -----------------------------------------------------------------------
# Approach 2: Block construction + differential evolution (n=512, 32 blocks)
# -----------------------------------------------------------------------
def approach2_block_de(n=512, n_blocks=32):
    print(f"\n=== Approach 2: Block DE (n={n}, blocks={n_blocks}) ===", flush=True)
    t0 = time.time()
    bs = n // n_blocks

    def p2h(params):
        raw = np.clip(params, 0, 1)
        h = np.repeat(raw, bs)[:n]
        s = np.sum(h)
        if s < 1e-10:
            return np.full(n, 0.5)
        return np.clip(h * (n / 2) / s, 0, 1)

    result = differential_evolution(
        lambda p: compute_metric(p2h(p)),
        [(0, 1)] * n_blocks,
        maxiter=600, popsize=20, tol=1e-10,
        seed=42, polish=True,
        mutation=(0.5, 1.5), recombination=0.9,
        disp=True,
    )
    h_de = project(p2h(result.x))
    m_de = compute_metric(h_de)
    print(f"  DE metric: {m_de:.12f}", flush=True)

    h_ft = lbfgsb(h_de, maxiter=3000)
    m_ft = compute_metric(h_ft)
    print(f"  Fine-tuned: {m_ft:.12f}  (t={time.time()-t0:.1f}s)", flush=True)

    best_h = h_ft if m_ft < m_de else h_de
    best_m = min(m_ft, m_de)
    return best_h, best_m


# -----------------------------------------------------------------------
# Approach 3: Structural analysis → parametric family
# -----------------------------------------------------------------------
def approach3_parametric(n=1024):
    print(f"\n=== Approach 3: Parametric from structure (n={n}) ===", flush=True)
    t0 = time.time()

    h_ref = load_best()

    # Sub-approach 3a: resample and fine-tune
    h4 = h_ref.reshape(-1, 4)[:, 0]  # compress 4096 → 1024
    if n != 1024:
        x_src = np.linspace(0, 1, len(h4))
        x_dst = np.linspace(0, 1, n)
        h4 = np.interp(x_dst, x_src, h4)
    h4 = project(h4)
    m4 = compute_metric(h4)
    print(f"  3a resampled: {m4:.12f}", flush=True)

    h3a = lbfgsb(h4, maxiter=10000)
    m3a = compute_metric(h3a)
    print(f"  3a fine-tuned: {m3a:.12f}", flush=True)

    # Sub-approach 3b: symmetric parametric DE with K breakpoints
    K = 30
    x_grid = np.linspace(0, 2, n, endpoint=False) + 1.0 / n

    def param_to_h(params):
        a = float(np.clip(params[0], 0.05, 0.40))
        heights = np.clip(params[1:K + 1], 0, 1)
        bp = np.linspace(a, 1.0, K + 2)
        bph = np.concatenate([[0.0], heights, [heights[-1]]])
        h = np.zeros(n)
        for i in range(n):
            xi = min(x_grid[i], 2.0 - x_grid[i])
            h[i] = 0.0 if xi < a else float(np.interp(xi, bp, bph))
        s = np.sum(h)
        if s < 1e-10:
            return np.full(n, 0.5)
        return np.clip(h * (n / 2) / s, 0, 1)

    print(f"  Running parametric DE (K={K})...", flush=True)
    res3b = differential_evolution(
        lambda p: compute_metric(param_to_h(p)),
        [(0.05, 0.40)] + [(0, 1)] * K,
        maxiter=600, popsize=15, tol=1e-10,
        seed=7, polish=True, mutation=(0.5, 1.5), recombination=0.9,
        disp=True,
    )
    h3b = project(param_to_h(res3b.x))
    m3b = compute_metric(h3b)
    print(f"  3b parametric DE: {m3b:.12f}", flush=True)
    h3b = lbfgsb(h3b, maxiter=3000)
    m3b = compute_metric(h3b)
    print(f"  3b fine-tuned: {m3b:.12f}", flush=True)

    best_h = h3a if m3a < m3b else h3b
    best_m = min(m3a, m3b)
    print(f"  BEST approach 3: {best_m:.12f}  (t={time.time()-t0:.1f}s)", flush=True)
    return best_h, best_m


# -----------------------------------------------------------------------
# Approach 4: Piecewise-linear DE (n=512, 60 breakpoints)
# -----------------------------------------------------------------------
def approach4_piecewise(n=512, nbp=60):
    print(f"\n=== Approach 4: Piecewise-linear DE (n={n}, bp={nbp}) ===", flush=True)
    t0 = time.time()
    bp_pos = np.linspace(0, n - 1, nbp)

    def pwl_to_h(heights):
        heights = np.clip(heights, 0, 1)
        h = np.interp(np.arange(n, dtype=float), bp_pos, heights)
        return project(h)

    res4 = differential_evolution(
        lambda p: compute_metric(pwl_to_h(p)),
        [(0, 1)] * nbp,
        maxiter=600, popsize=15, tol=1e-10,
        seed=99, polish=True, mutation=(0.5, 1.5), recombination=0.9,
        disp=True,
    )
    h4 = project(pwl_to_h(res4.x))
    m4 = compute_metric(h4)
    print(f"  DE metric: {m4:.12f}", flush=True)

    h4 = lbfgsb(h4, maxiter=3000)
    m4 = compute_metric(h4)
    print(f"  Fine-tuned: {m4:.12f}  (t={time.time()-t0:.1f}s)", flush=True)
    return h4, m4


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60, flush=True)
    print("Erdős Minimum Overlap - dp-construction", flush=True)
    print("=" * 60, flush=True)

    results = {}
    global_best_h = load_best()
    global_best_m = compute_metric(global_best_h)
    print(f"Best known (slp-4096): {global_best_m:.15f}  n={len(global_best_h)}", flush=True)

    # Run all approaches
    for name, fn, args in [
        ('approach1_greedy',    approach1_greedy,    (512,)),
        ('approach2_block_de',  approach2_block_de,  (512, 32)),
        ('approach3_parametric', approach3_parametric, (1024,)),
        ('approach4_piecewise', approach4_piecewise,  (512, 60)),
    ]:
        try:
            h, m = fn(*args)
            results[name] = {'n': len(h), 'metric': m}
            if m < global_best_m:
                global_best_m = m
                global_best_h = h.copy()
                print(f"  *** NEW BEST: {m:.15f} ***", flush=True)
        except Exception as e:
            print(f"  ERROR in {name}: {e}", flush=True)
            results[name] = {'n': 0, 'metric': float('inf'), 'error': str(e)}

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Best known: {BEST_KNOWN:.12f}", flush=True)
    for name, info in results.items():
        beat = " [BEATS RECORD!]" if info['metric'] < BEST_KNOWN else ""
        print(f"  {name}: n={info['n']}, metric={info['metric']:.12f}{beat}", flush=True)
    print(f"\nGlobal best found: {global_best_m:.15f}", flush=True)
    if global_best_m < BEST_KNOWN:
        print(f"IMPROVEMENT: {BEST_KNOWN - global_best_m:.2e}", flush=True)

    # Save best solution
    sol_path = OUT_DIR / 'solution.py'
    lines = ['import numpy as np', '', 'h_values = np.array([']
    for v in global_best_h:
        lines.append(f'  np.float64({repr(float(v))}),')
    lines.append('])')
    sol_path.write_text('\n'.join(lines) + '\n')
    print(f"\nSaved solution to {sol_path}  (n={len(global_best_h)}, metric={global_best_m:.15f})", flush=True)

    # Save results JSON
    res_path = OUT_DIR / 'results.json'
    res_path.write_text(json.dumps({
        'best_known': BEST_KNOWN,
        'best_found': global_best_m,
        'improvement': BEST_KNOWN - global_best_m,
        'approaches': results,
    }, indent=2))
    print(f"Saved results to {res_path}", flush=True)

    return results, global_best_h, global_best_m


if __name__ == '__main__':
    main()
