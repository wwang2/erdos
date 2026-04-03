"""
Sequential Linear Programming (SLP) optimization for Erdős Minimum Overlap Problem.

Strategy:
1. Seed from slp-1024-multistart best solution
2. Iteratively solve LP subproblems to minimize the max cross-correlation
3. Multi-shift handling: track top-K shifts near the maximum
4. Line search for step size alpha
5. Save checkpoints every 100 iterations
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

# Paths
WORK_DIR = Path(__file__).parent
REPO_ROOT = WORK_DIR.parent.parent
SEED_PATH = REPO_ROOT / "orbits/slp-1024-multistart/solution.py"
SOLUTION_PATH = WORK_DIR / "solution.py"
CHECKPOINT_PATH = WORK_DIR / "checkpoint.npy"

# Problem constants
N = 1024
N_HALF = N / 2.0  # 512.0


def load_seed():
    """Load slp-1024-multistart solution as seed."""
    ns = {}
    exec(SEED_PATH.read_text(), ns)
    return np.asarray(ns["h_values"])


def compute_upper_bound(h):
    """Compute max_k (2/n) sum_i h_i (1 - h_{i+k}) and the argmax index into full correlate."""
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    idx = int(np.argmax(conv))
    return float(conv[idx] / n * 2), idx


def get_all_overlaps(h):
    """Return (2/n)*correlate(h, 1-h, 'full') array of length 2n-1."""
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return conv / n * 2


def analytic_gradient(h, idx):
    """
    Analytic gradient of F at correlation index idx, w.r.t. h.

    np.correlate(h, 1-h, 'full')[idx] = sum_j h[j] * (1-h[j - s])
    where s = idx - (n-1)  (the lag)

    For s >= 0:  sum over j in [s, n-1] of h[j] * (1 - h[j-s])
      dF/dh_k:
        from term j=k (h[k]*(1-h[k-s])): +(1-h[k-s])  if s <= k <= n-1
        from term j=k+s (h[k+s]*(1-h[k])): -h[k+s]    if 0 <= k <= n-1-s

    For s < 0 (t = -s > 0):  sum over j in [0, n-1-t] of h[j] * (1 - h[j+t])
      dF/dh_k:
        from term j=k (h[k]*(1-h[k+t])): +(1-h[k+t])  if 0 <= k <= n-1-t
        from term j=k-t (h[k-t]*(1-h[k])): -h[k-t]    if t <= k <= n-1
    """
    n = len(h)
    s = idx - (n - 1)
    grad = np.zeros(n)

    if s >= 0:
        # sum_j h[j]*(1-h[j-s]) for j in [s, n-1]
        # dF/dh_k:
        #   +(1-h[k-s]) for k in [s, n-1]
        grad[s:n] += (1.0 - h[:n - s])
        #   -h[k+s] for k in [0, n-1-s]
        grad[:n - s] -= h[s:n]
    else:
        t = -s  # positive
        # sum_j h[j]*(1-h[j+t]) for j in [0, n-1-t]
        # dF/dh_k:
        #   +(1-h[k+t]) for k in [0, n-1-t]
        grad[:n - t] += (1.0 - h[t:n])
        #   -h[k-t] for k in [t, n-1]
        grad[t:n] -= h[:n - t]

    return grad / n * 2


def project_to_feasible(h):
    """Project h to [0,1] with sum = N/2 using a simple iterative approach."""
    h = np.clip(h, 0.0, 1.0)
    diff = N_HALF - h.sum()
    if abs(diff) < 1e-12:
        return h
    # Distribute diff to elements with room
    if diff > 0:
        # Need to increase sum: add to elements < 1
        mask = h < 1.0 - 1e-12
    else:
        # Need to decrease sum: subtract from elements > 0
        mask = h > 1e-12
    if mask.sum() > 0:
        delta = diff / mask.sum()
        h[mask] += delta
        h = np.clip(h, 0.0, 1.0)
    return h


def solve_lp_subproblem(h, shift_indices, delta=None):
    """
    Solve LP to find h_lp that minimizes the max linearized overlap.

    Variables: x = [h_lp (n), t (1)]  total n+1
    Minimize:  t
    Subject to:
      grad_s @ h_lp - t <= grad_s @ h - F_s(h)   for each shift s in shift_indices
      sum(h_lp) = n/2
      0 <= h_lp_i <= 1
      |h_lp_i - h_i| <= delta  (trust region, optional)
    """
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

    # Objective: minimize t
    c = np.zeros(n + 1)
    c[-1] = 1.0

    # Inequality: grad_s @ h_lp - t <= grad_s @ h - F_s
    A_ub = np.zeros((k, n + 1))
    b_ub = np.zeros(k)
    for i, (g, F_s) in enumerate(zip(grads, F_values)):
        A_ub[i, :n] = g
        A_ub[i, n] = -1.0
        b_ub[i] = g @ h - F_s

    # Equality: sum h_lp = n/2
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([N_HALF])

    # Bounds: trust region clips [0,1] interval
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
        h_lp = np.clip(result.x[:n], 0.0, 1.0)
        return h_lp
    else:
        return None


def line_search(h, h_lp, current_obj, n_steps=40):
    """Grid search + refinement over alpha values to find best step size."""
    best_alpha = 0.0
    best_obj = current_obj

    # Coarse search: logspace from 1.0 down to 1e-8
    alphas = np.logspace(0, -8, n_steps)
    for alpha in alphas:
        h_new = h + alpha * (h_lp - h)
        h_new = project_to_feasible(h_new)
        obj, _ = compute_upper_bound(h_new)
        if obj < best_obj:
            best_obj = obj
            best_alpha = alpha

    # Refine around best_alpha if found
    if best_alpha > 0:
        lo = best_alpha * 0.5
        hi = best_alpha * 2.0
        for alpha in np.linspace(lo, hi, 20):
            h_new = h + alpha * (h_lp - h)
            h_new = project_to_feasible(h_new)
            obj, _ = compute_upper_bound(h_new)
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha

    return best_alpha, best_obj


def save_solution(h):
    """Save h_values as a Python file."""
    lines = ["import numpy as np\n\nh_values = np.array([\n"]
    for v in h:
        lines.append(f"  {v!r},\n")
    lines.append("])\n")
    SOLUTION_PATH.write_text("".join(lines))


def run_slp(n_iter=5000, top_k=10, checkpoint_every=100, seed_h=None):
    """Main SLP optimization loop with adaptive active-set selection."""
    if seed_h is not None:
        h = project_to_feasible(seed_h.copy())
        print(f"  Starting from provided seed: sum={h.sum():.6f}")
    else:
        print(f"Loading seed from slp-1024-multistart...")
        h = load_seed()
        h = project_to_feasible(h.copy())
        print(f"  Seed: shape={h.shape}, sum={h.sum():.6f}")

        # Load checkpoint if better
        if CHECKPOINT_PATH.exists():
            h_ckpt = np.load(CHECKPOINT_PATH)
            if len(h_ckpt) == N:
                obj_ckpt, _ = compute_upper_bound(h_ckpt)
                obj_init, _ = compute_upper_bound(h)
                if obj_ckpt < obj_init:
                    print(f"  Resuming from checkpoint (obj={obj_ckpt:.15f} < init {obj_init:.15f})")
                    h = h_ckpt.copy()

    obj, _ = compute_upper_bound(h)
    print(f"  Initial upper bound: {obj:.15f}")

    best_h = h.copy()
    best_obj = obj
    no_improve = 0
    delta = 0.05
    # Active-set threshold: include all shifts within threshold of max
    active_threshold = 1e-3
    start = time.time()

    for it in range(1, n_iter + 1):
        overlaps = get_all_overlaps(h)
        max_ov = overlaps.max()

        # Adaptively select active shifts
        active_mask = overlaps >= max_ov - active_threshold
        shift_idx = np.where(active_mask)[0]
        n_active = len(shift_idx)

        # Limit to avoid extremely slow LP (cap at 1000 shifts)
        if n_active > 1000:
            top_idx = np.argpartition(overlaps, -1000)[-1000:]
            shift_idx = top_idx[np.argsort(overlaps[top_idx])[::-1]]
            n_active = len(shift_idx)

        # Solve LP with trust region
        h_lp = solve_lp_subproblem(h, shift_idx, delta=delta)
        if h_lp is None:
            print(f"  Iter {it}: LP failed ({n_active} shifts), shrinking delta/threshold")
            delta *= 0.5
            active_threshold *= 0.5
            no_improve += 1
            if no_improve >= 10:
                break
            continue

        # Line search along the LP direction
        alpha, new_obj = line_search(h, h_lp, obj)
        improvement = obj - new_obj

        if it % 10 == 0 or it <= 5:
            elapsed = time.time() - start
            print(f"  Iter {it:5d}: obj={new_obj:.15f}  alpha={alpha:.5f}  "
                  f"active={n_active}  delta={delta:.4f}  improve={improvement:.2e}  t={elapsed:.1f}s")

        if alpha > 0 and improvement > 0:
            h = h + alpha * (h_lp - h)
            h = project_to_feasible(h)
            obj = new_obj
            no_improve = 0
            delta = min(delta * 1.1, 0.5)

            if obj < best_obj:
                best_obj = obj
                best_h = h.copy()
        else:
            no_improve += 1
            delta = max(delta * 0.7, 1e-6)
            # Try smaller threshold when stuck
            if no_improve % 5 == 0:
                active_threshold = max(active_threshold * 0.5, 1e-8)
            if no_improve >= 50:
                print(f"  No improvement for 50 iters, stopping at iter {it}")
                break

        if it % checkpoint_every == 0:
            np.save(CHECKPOINT_PATH, best_h)
            save_solution(best_h)
            print(f"  [Checkpoint {it}] best_obj={best_obj:.15f}")

        if improvement < 1e-14 and it > 100 and delta < 1e-5 and no_improve >= 20:
            print(f"  Converged at iter {it} (improvement={improvement:.2e}, delta={delta:.2e})")
            break

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Best obj={best_obj:.15f}")
    return best_h, best_obj


def run_with_restarts(n_restarts=50, n_iter_per_run=500, top_k=10):
    """Run SLP with perturbation restarts to escape local minima."""
    # Initial run from seed
    print("=== Initial run ===")
    best_h, best_obj = run_slp(n_iter=n_iter_per_run, top_k=top_k, checkpoint_every=500)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Initial best: {best_obj:.15f}")

    print(f"\n--- Starting perturbation restarts (best so far: {best_obj:.15f}) ---")

    scales = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]

    for restart in range(n_restarts):
        scale = scales[restart % len(scales)]
        np.random.seed(restart + 100)
        h_perturbed = best_h + np.random.randn(N) * scale
        h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
        h_perturbed = project_to_feasible(h_perturbed)

        obj_perturbed, _ = compute_upper_bound(h_perturbed)
        print(f"\n=== Restart {restart+1}/{n_restarts}: scale={scale:.4f}, "
              f"perturbed_obj={obj_perturbed:.15f} ===")

        h_new, obj_new = run_slp(n_iter=n_iter_per_run, top_k=top_k,
                                  checkpoint_every=500, seed_h=h_perturbed)

        if obj_new < best_obj:
            best_obj = obj_new
            best_h = h_new.copy()
            print(f"*** NEW BEST: {best_obj:.15f} ***")
            save_solution(best_h)
            np.save(CHECKPOINT_PATH, best_h)
        else:
            print(f"No improvement: {obj_new:.15f} >= {best_obj:.15f}")

    print(f"\nFinal best: {best_obj:.15f}")
    return best_h, best_obj


def main():
    best_h, best_obj = run_with_restarts(n_restarts=50, n_iter_per_run=500)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Saved to {SOLUTION_PATH}")


if __name__ == "__main__":
    # Add repo root to sys.path so we can import evaluator
    sys.path.insert(0, str(REPO_ROOT))
    main()
