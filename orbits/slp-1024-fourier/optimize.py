"""
Fourier-parameterized optimization for Erdős Minimum Overlap Problem.

Strategy:
1. Fit Fourier coefficients to slp-1024 solution via FFT
2. Optimize in Fourier space using scipy L-BFGS-B
3. Vary K=50,100,200,300 Fourier modes
4. Polish best result with SLP iterations

Parameterization:
  h_i = clip(0.5 + sum_{k=1}^{K} [a_k cos(2pi*k*i/n) + b_k sin(2pi*k*i/n)], 0, 1)
  then rescale sum to n/2

The DC=0.5 component approximately satisfies sum=n/2 before clipping.
After clipping, rescale.
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.optimize import linprog

WORK_DIR = Path(__file__).parent
REPO_ROOT = WORK_DIR.parent.parent
# slp-1024 solution lives in the main repo (not the worktree)
_MAIN_REPO = Path("/Users/wujiewang/code/erdos")
SLP_SOLUTION_PATH = _MAIN_REPO / "orbits/slp-1024/solution.py"
SOLUTION_PATH = WORK_DIR / "solution.py"

N = 1024
N_HALF = N / 2.0


# ─── Core helpers ────────────────────────────────────────────────────────────

def load_slp_solution():
    ns = {}
    exec(SLP_SOLUTION_PATH.read_text(), ns)
    return np.asarray(ns["h_values"])


def compute_upper_bound(h):
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return float(np.max(conv) / n * 2)


def rescale(h):
    """Clip to [0,1] then rescale sum to N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s > 1e-12:
        h = h * (N_HALF / s)
        h = np.clip(h, 0.0, 1.0)
    return h


# ─── Fourier helpers ─────────────────────────────────────────────────────────

def build_basis(n, K):
    """Build cosine and sine basis matrices of shape (n, K) each."""
    i = np.arange(n)
    freqs = np.arange(1, K + 1)
    phases = 2 * np.pi * np.outer(i, freqs) / n  # (n, K)
    cos_basis = np.cos(phases)  # (n, K)
    sin_basis = np.sin(phases)  # (n, K)
    return cos_basis, sin_basis


def params_to_h(params, cos_basis, sin_basis):
    """Convert 2K Fourier parameters to h array of length N."""
    K = cos_basis.shape[1]
    a = params[:K]
    b = params[K:]
    h_raw = 0.5 + cos_basis @ a + sin_basis @ b
    return rescale(h_raw)


def fit_fourier(h_target, K):
    """Extract K dominant Fourier coefficients from h_target via FFT."""
    n = len(h_target)
    # Subtract DC=0.5 before fitting
    h_centered = h_target - 0.5
    fft_vals = np.fft.rfft(h_centered) / n
    # fft_vals[k] = (a_k - i*b_k)/2 for k>=1 (numpy convention)
    # So a_k = 2*Re(fft_vals[k]), b_k = -2*Im(fft_vals[k])
    a = np.zeros(K)
    b = np.zeros(K)
    for k in range(1, min(K + 1, len(fft_vals))):
        a[k - 1] = 2.0 * fft_vals[k].real
        b[k - 1] = -2.0 * fft_vals[k].imag
    return np.concatenate([a, b])


# ─── Objective with gradient ──────────────────────────────────────────────────

def objective_and_grad(params, cos_basis, sin_basis):
    """
    Objective: max_k (2/n) sum_i h_i (1 - h_{i+k})
    Gradient via chain rule through the clipping/rescaling.

    We use finite differences for the gradient since the clipping makes
    analytic gradients complex. L-BFGS-B will handle this well.
    """
    h = params_to_h(params, cos_basis, sin_basis)
    obj = compute_upper_bound(h)
    return obj


def objective_val(params, cos_basis, sin_basis):
    h = params_to_h(params, cos_basis, sin_basis)
    return compute_upper_bound(h)


# ─── SLP polishing ────────────────────────────────────────────────────────────

def get_all_overlaps(h):
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return conv / n * 2


def analytic_gradient(h, idx):
    n = len(h)
    s = idx - (n - 1)
    grad = np.zeros(n)
    if s >= 0:
        grad[s:n] += (1.0 - h[:n - s])
        grad[:n - s] -= h[s:n]
    else:
        t = -s
        grad[:n - t] += (1.0 - h[t:n])
        grad[t:n] -= h[:n - t]
    return grad / n * 2


def project_to_feasible(h):
    h = np.clip(h, 0.0, 1.0)
    diff = N_HALF - h.sum()
    if abs(diff) < 1e-12:
        return h
    if diff > 0:
        mask = h < 1.0 - 1e-12
    else:
        mask = h > 1e-12
    if mask.sum() > 0:
        delta = diff / mask.sum()
        h[mask] += delta
        h = np.clip(h, 0.0, 1.0)
    return h


def solve_lp_subproblem(h, shift_indices, delta=None):
    n = len(h)
    k = len(shift_indices)
    grads = []
    F_values = []
    overlaps = get_all_overlaps(h)

    for idx in shift_indices:
        F_s = float(overlaps[int(idx)])
        g = analytic_gradient(h, int(idx))
        grads.append(g)
        F_values.append(F_s)

    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub = np.zeros((k, n + 1))
    b_ub = np.zeros(k)
    for i, (g, F_s) in enumerate(zip(grads, F_values)):
        A_ub[i, :n] = g
        A_ub[i, n] = -1.0
        b_ub[i] = g @ h - F_s

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([N_HALF])

    if delta is not None:
        lo = np.maximum(0.0, h - delta)
        hi = np.minimum(1.0, h + delta)
        bounds = list(zip(lo.tolist(), hi.tolist())) + [(None, None)]
    else:
        bounds = [(0.0, 1.0)] * n + [(None, None)]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
        options={'disp': False, 'time_limit': 120.0, 'primal_feasibility_tolerance': 1e-9},
    )

    if result.status == 0:
        return np.clip(result.x[:n], 0.0, 1.0)
    return None


def line_search(h, h_lp, current_obj, n_steps=40):
    best_alpha = 0.0
    best_obj = current_obj
    alphas = np.logspace(0, -8, n_steps)
    for alpha in alphas:
        h_new = project_to_feasible(h + alpha * (h_lp - h))
        obj = compute_upper_bound(h_new)
        if obj < best_obj:
            best_obj = obj
            best_alpha = alpha
    if best_alpha > 0:
        for alpha in np.linspace(best_alpha * 0.5, best_alpha * 2.0, 20):
            h_new = project_to_feasible(h + alpha * (h_lp - h))
            obj = compute_upper_bound(h_new)
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha
    return best_alpha, best_obj


def run_slp_polish(h_init, n_iter=300, active_threshold=1e-3):
    """Run SLP iterations to polish a solution."""
    h = project_to_feasible(h_init.copy())
    obj = compute_upper_bound(h)
    best_h = h.copy()
    best_obj = obj
    no_improve = 0
    delta = 0.05
    threshold = active_threshold
    start = time.time()

    print(f"  SLP polish start: obj={obj:.15f}")

    for it in range(1, n_iter + 1):
        overlaps = get_all_overlaps(h)
        max_ov = overlaps.max()
        active_mask = overlaps >= max_ov - threshold
        shift_idx = np.where(active_mask)[0]
        n_active = len(shift_idx)

        if n_active > 1000:
            top_idx = np.argpartition(overlaps, -1000)[-1000:]
            shift_idx = top_idx[np.argsort(overlaps[top_idx])[::-1]]
            n_active = len(shift_idx)

        h_lp = solve_lp_subproblem(h, shift_idx, delta=delta)
        if h_lp is None:
            delta *= 0.5
            threshold *= 0.5
            no_improve += 1
            if no_improve >= 10:
                break
            continue

        alpha, new_obj = line_search(h, h_lp, obj)
        improvement = obj - new_obj

        if it % 20 == 0:
            elapsed = time.time() - start
            print(f"    Iter {it:4d}: obj={new_obj:.15f}  alpha={alpha:.5f}  "
                  f"active={n_active}  delta={delta:.4f}  t={elapsed:.1f}s")

        if alpha > 0 and improvement > 0:
            h = project_to_feasible(h + alpha * (h_lp - h))
            obj = new_obj
            no_improve = 0
            delta = min(delta * 1.1, 0.5)
            if obj < best_obj:
                best_obj = obj
                best_h = h.copy()
        else:
            no_improve += 1
            delta = max(delta * 0.7, 1e-6)
            if no_improve % 5 == 0:
                threshold = max(threshold * 0.5, 1e-8)
            if no_improve >= 30:
                print(f"    No improvement for 30 iters, stopping at iter {it}")
                break

    elapsed = time.time() - start
    print(f"  SLP polish done in {elapsed:.1f}s. Best obj={best_obj:.15f}")
    return best_h, best_obj


# ─── Main Fourier optimization ─────────────────────────────────────────────────

def optimize_fourier(K, h_seed, cos_basis, sin_basis, maxiter=2000):
    """Optimize Fourier coefficients using L-BFGS-B."""
    print(f"\n  === K={K} Fourier modes ===")

    # Fit initial params to seed solution
    params0 = fit_fourier(h_seed, K)
    h_recon = params_to_h(params0, cos_basis, sin_basis)
    obj0 = compute_upper_bound(h_recon)
    print(f"  Initial (from FFT fit): obj={obj0:.15f}")

    call_count = [0]
    best = [params0.copy(), obj0]

    def callback(params):
        call_count[0] += 1
        h = params_to_h(params, cos_basis, sin_basis)
        obj = compute_upper_bound(h)
        if obj < best[1]:
            best[1] = obj
            best[0] = params.copy()
        if call_count[0] % 100 == 0:
            print(f"    iter {call_count[0]}: obj={best[1]:.15f}")

    def obj_fn(params):
        return objective_val(params, cos_basis, sin_basis)

    start = time.time()
    result = minimize(
        obj_fn,
        params0,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': maxiter * 20},
        callback=callback,
    )
    elapsed = time.time() - start

    h_opt = params_to_h(result.x, cos_basis, sin_basis)
    obj_opt = compute_upper_bound(h_opt)

    # Also evaluate best tracked params
    h_best = params_to_h(best[0], cos_basis, sin_basis)
    obj_best = compute_upper_bound(h_best)

    final_h = h_best if obj_best <= obj_opt else h_opt
    final_obj = min(obj_best, obj_opt)

    print(f"  K={K} done in {elapsed:.1f}s: obj={final_obj:.15f} (scipy reports {obj_opt:.15f})")
    return final_h, final_obj


def save_solution(h, path=None):
    if path is None:
        path = SOLUTION_PATH
    lines = ["import numpy as np\n\nh_values = np.array([\n"]
    for v in h:
        lines.append(f"  {v!r},\n")
    lines.append("])\n")
    Path(path).write_text("".join(lines))


def main():
    print("Loading slp-1024 solution...")
    h_slp = load_slp_solution()
    obj_slp = compute_upper_bound(h_slp)
    print(f"slp-1024 baseline: obj={obj_slp:.15f}")

    results = {}
    global_best_h = h_slp.copy()
    global_best_obj = obj_slp

    K_values = [50, 100, 200, 300]

    for K in K_values:
        print(f"\nBuilding basis for K={K}...")
        cos_basis, sin_basis = build_basis(N, K)

        h_opt, obj_opt = optimize_fourier(K, h_slp, cos_basis, sin_basis, maxiter=3000)
        results[K] = obj_opt

        print(f"\n  Metrics so far: {results}")

        if obj_opt < global_best_obj:
            global_best_obj = obj_opt
            global_best_h = h_opt.copy()
            print(f"  *** New best from Fourier K={K}: {global_best_obj:.15f} ***")

    print(f"\n{'='*60}")
    print(f"Fourier optimization results:")
    for K, obj in results.items():
        print(f"  K={K}: {obj:.15f}")
    print(f"Best Fourier: {global_best_obj:.15f}")
    print(f"SLP baseline: {obj_slp:.15f}")

    # SLP polishing of best Fourier solution
    print(f"\n{'='*60}")
    print("SLP polishing of best Fourier solution...")
    h_polished, obj_polished = run_slp_polish(global_best_h, n_iter=500)
    print(f"After SLP polish: {obj_polished:.15f}")

    if obj_polished < global_best_obj:
        global_best_obj = obj_polished
        global_best_h = h_polished.copy()
        print(f"*** New best after SLP polish: {global_best_obj:.15f} ***")

    print(f"\nFinal best: {global_best_obj:.15f}")
    save_solution(global_best_h)
    print(f"Saved to {SOLUTION_PATH}")

    return global_best_h, global_best_obj, results, obj_polished


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
