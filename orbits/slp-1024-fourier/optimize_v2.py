"""
Fourier optimization v2: More aggressive multi-start in Fourier space.

Key changes from v1:
- Use K=200 with random restarts around the FFT-fitted solution
- Run L-BFGS-B with tighter tolerances for more iterations
- Try Nelder-Mead as a local polisher for the best L-BFGS-B result
- Save any improvement found
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

WORK_DIR = Path(__file__).parent
_MAIN_REPO = Path("/Users/wujiewang/code/erdos")
SLP_SOLUTION_PATH = _MAIN_REPO / "orbits/slp-1024/solution.py"
SOLUTION_PATH = WORK_DIR / "solution.py"

N = 1024
N_HALF = N / 2.0


def load_slp_solution():
    ns = {}
    exec(SLP_SOLUTION_PATH.read_text(), ns)
    return np.asarray(ns["h_values"])


def compute_upper_bound(h):
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return float(np.max(conv) / n * 2)


def rescale(h):
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s > 1e-12:
        h = h * (N_HALF / s)
        h = np.clip(h, 0.0, 1.0)
    return h


def build_basis(n, K):
    i = np.arange(n)
    freqs = np.arange(1, K + 1)
    phases = 2 * np.pi * np.outer(i, freqs) / n
    return np.cos(phases), np.sin(phases)


def fit_fourier(h_target, K):
    n = len(h_target)
    h_centered = h_target - 0.5
    fft_vals = np.fft.rfft(h_centered) / n
    a = np.zeros(K)
    b = np.zeros(K)
    for k in range(1, min(K + 1, len(fft_vals))):
        a[k - 1] = 2.0 * fft_vals[k].real
        b[k - 1] = -2.0 * fft_vals[k].imag
    return np.concatenate([a, b])


def params_to_h(params, cos_basis, sin_basis):
    K = cos_basis.shape[1]
    a = params[:K]
    b = params[K:]
    h_raw = 0.5 + cos_basis @ a + sin_basis @ b
    return rescale(h_raw)


def objective_val(params, cos_basis, sin_basis):
    h = params_to_h(params, cos_basis, sin_basis)
    return compute_upper_bound(h)


def save_solution(h, path=None):
    if path is None:
        path = SOLUTION_PATH
    lines = ["import numpy as np\n\nh_values = np.array([\n"]
    for v in h:
        lines.append(f"  {v!r},\n")
    lines.append("])\n")
    Path(path).write_text("".join(lines))


def run_lbfgsb(params0, cos_basis, sin_basis, maxiter=5000, label=""):
    """Single L-BFGS-B run from params0."""
    best = [params0.copy(), objective_val(params0, cos_basis, sin_basis)]
    call_count = [0]

    def callback(params):
        call_count[0] += 1
        obj = objective_val(params, cos_basis, sin_basis)
        if obj < best[1]:
            best[1] = obj
            best[0] = params.copy()

    def obj_fn(params):
        return objective_val(params, cos_basis, sin_basis)

    result = minimize(
        obj_fn,
        params0,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': maxiter * 20},
        callback=callback,
    )

    # Take best of scipy result and tracked best
    obj_scipy = objective_val(result.x, cos_basis, sin_basis)
    if obj_scipy < best[1]:
        best[0] = result.x.copy()
        best[1] = obj_scipy

    return best[0], best[1]


def main():
    print("Loading slp-1024 solution...")
    h_slp = load_slp_solution()
    obj_slp = compute_upper_bound(h_slp)
    print(f"SLP baseline: {obj_slp:.15f}")

    K = 200
    print(f"\nBuilding K={K} Fourier basis...")
    cos_basis, sin_basis = build_basis(N, K)

    # Fit initial params to slp solution
    params_fit = fit_fourier(h_slp, K)
    obj_fit = objective_val(params_fit, cos_basis, sin_basis)
    print(f"FFT fit: obj={obj_fit:.15f}")

    global_best_params = params_fit.copy()
    global_best_obj = obj_fit
    global_best_h = params_to_h(params_fit, cos_basis, sin_basis)

    n_restarts = 20
    rng = np.random.RandomState(42)

    print(f"\nRunning {n_restarts} multi-start L-BFGS-B optimizations (K={K})...")

    # First run: from FFT fit
    print(f"\n  Run 0 (FFT fit seed): start obj={obj_fit:.15f}")
    t0 = time.time()
    p_opt, obj_opt = run_lbfgsb(params_fit, cos_basis, sin_basis, maxiter=5000)
    print(f"    -> obj={obj_opt:.15f} in {time.time()-t0:.1f}s")
    if obj_opt < global_best_obj:
        global_best_obj = obj_opt
        global_best_params = p_opt.copy()
        global_best_h = params_to_h(p_opt, cos_basis, sin_basis)
        print(f"    *** NEW BEST: {global_best_obj:.15f} ***")

    # Subsequent runs: perturbed from best found so far
    scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.001, 0.005, 0.01, 0.02, 0.05,
              0.003, 0.008, 0.015, 0.03, 0.08, 0.002, 0.004, 0.007, 0.012, 0.025]

    for i in range(n_restarts):
        scale = scales[i % len(scales)]
        noise = rng.randn(len(global_best_params)) * scale
        p_start = global_best_params + noise
        obj_start = objective_val(p_start, cos_basis, sin_basis)

        print(f"\n  Run {i+1}/{n_restarts} (scale={scale:.4f}): start obj={obj_start:.15f}")
        t0 = time.time()
        p_opt, obj_opt = run_lbfgsb(p_start, cos_basis, sin_basis, maxiter=3000)
        print(f"    -> obj={obj_opt:.15f} in {time.time()-t0:.1f}s")

        if obj_opt < global_best_obj:
            global_best_obj = obj_opt
            global_best_params = p_opt.copy()
            global_best_h = params_to_h(p_opt, cos_basis, sin_basis)
            print(f"    *** NEW BEST: {global_best_obj:.15f} ***")

    print(f"\n{'='*60}")
    print(f"Final best: {global_best_obj:.15f}")
    print(f"SLP baseline: {obj_slp:.15f}")
    print(f"Improvement: {obj_slp - global_best_obj:.2e}")

    # Only save if we beat slp-1024
    if global_best_obj < obj_slp:
        save_solution(global_best_h)
        print(f"*** Saved to {SOLUTION_PATH} ***")
        return global_best_h, global_best_obj
    else:
        print(f"Did not beat slp-1024 baseline.")
        # Save anyway for the record
        save_solution(global_best_h)
        return global_best_h, global_best_obj


if __name__ == "__main__":
    sys.path.insert(0, str(_MAIN_REPO))
    main()
