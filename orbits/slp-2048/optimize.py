"""
Sequential Linear Programming (SLP) optimization for Erdős Minimum Overlap Problem.
n=2048, seeded from slp-1024 solution via tiling (repeat each element twice).

Strategy:
1. Seed from slp-1024 solution tiled to 2048 steps (preserves exact metric 0.380870)
2. Iteratively solve LP subproblems to minimize the max cross-correlation
3. Multi-shift handling: top-K active shifts near the maximum
4. Line search for step size alpha
5. Save checkpoints every 50 iterations
6. Perturbation restarts to escape local optima
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

WORK_DIR = Path(__file__).parent
REPO_ROOT = WORK_DIR.parent.parent
_MAIN_REPO = Path("/Users/wujiewang/code/erdos")
SEED_PATH = _MAIN_REPO / "orbits/slp-1024/solution.py"
SOLUTION_PATH = WORK_DIR / "solution.py"
CHECKPOINT_PATH = WORK_DIR / "checkpoint.npy"

N = 2048
N_HALF = N / 2.0

MAX_ACTIVE = 500


def load_seed_tiled():
    """Load slp-1024 solution and tile to 2048 by repeating each element twice."""
    ns = {}
    exec(SEED_PATH.read_text(), ns)
    h1024 = np.asarray(ns["h_values"], dtype=float)
    h2048 = np.repeat(h1024, 2)
    # Rescale to ensure exact sum
    h2048 *= N_HALF / h2048.sum()
    h2048 = np.clip(h2048, 0.0, 1.0)
    return h2048


def compute_upper_bound(h):
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    idx = int(np.argmax(conv))
    return float(conv[idx] / n * 2), idx


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
    mask = h < 1.0 - 1e-12 if diff > 0 else h > 1e-12
    if mask.sum() > 0:
        h[mask] += diff / mask.sum()
        h = np.clip(h, 0.0, 1.0)
    return h


def solve_lp_subproblem(h, shift_indices, delta=None):
    n = len(h)
    k = len(shift_indices)
    overlaps = get_all_overlaps(h)

    grads = []
    F_values = []
    for idx in shift_indices:
        g = analytic_gradient(h, int(idx))
        grads.append(g)
        F_values.append(float(overlaps[int(idx)]))

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

    for alpha in np.logspace(0, -8, n_steps):
        h_new = project_to_feasible(h + alpha * (h_lp - h))
        obj, _ = compute_upper_bound(h_new)
        if obj < best_obj:
            best_obj = obj
            best_alpha = alpha

    if best_alpha > 0:
        for alpha in np.linspace(best_alpha * 0.5, best_alpha * 2.0, 20):
            h_new = project_to_feasible(h + alpha * (h_lp - h))
            obj, _ = compute_upper_bound(h_new)
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha

    return best_alpha, best_obj


def save_solution(h):
    lines = ["import numpy as np\n\nh_values = np.array([\n"]
    for v in h:
        lines.append(f"  {v!r},\n")
    lines.append("])\n")
    SOLUTION_PATH.write_text("".join(lines))


def run_slp(n_iter=500, checkpoint_every=50, seed_h=None,
            init_delta=0.05, init_threshold=1e-3, no_improve_limit=50):
    if seed_h is not None:
        h = project_to_feasible(seed_h.copy())
        print(f"  Starting from seed: obj={compute_upper_bound(h)[0]:.15f}")
    else:
        # Use tiled seed (always starts at 0.380870134 — the slp-1024 best)
        print(f"Loading slp-1024 tiled to {N} steps...")
        h = load_seed_tiled()
        h = project_to_feasible(h)
        obj_seed, _ = compute_upper_bound(h)
        print(f"  Tiled seed: shape={h.shape}, sum={h.sum():.6f}, obj={obj_seed:.15f}")

        # Use checkpoint if it's better
        if CHECKPOINT_PATH.exists():
            h_ckpt = np.load(CHECKPOINT_PATH)
            if len(h_ckpt) == N:
                obj_ckpt, _ = compute_upper_bound(h_ckpt)
                if obj_ckpt < obj_seed:
                    print(f"  Using checkpoint (obj={obj_ckpt:.15f} < seed {obj_seed:.15f})")
                    h = h_ckpt.copy()

    obj, _ = compute_upper_bound(h)
    print(f"  Initial obj: {obj:.15f}")

    best_h = h.copy()
    best_obj = obj
    no_improve = 0
    delta = init_delta
    active_threshold = init_threshold
    start = time.time()

    for it in range(1, n_iter + 1):
        overlaps = get_all_overlaps(h)
        max_ov = overlaps.max()

        active_mask = overlaps >= max_ov - active_threshold
        shift_idx = np.where(active_mask)[0]
        n_active = len(shift_idx)

        if n_active > MAX_ACTIVE:
            top_idx = np.argpartition(overlaps, -MAX_ACTIVE)[-MAX_ACTIVE:]
            shift_idx = top_idx[np.argsort(overlaps[top_idx])[::-1]]
            n_active = MAX_ACTIVE

        h_lp = solve_lp_subproblem(h, shift_idx, delta=delta)
        if h_lp is None:
            print(f"  Iter {it}: LP failed ({n_active} shifts), shrinking")
            delta *= 0.5
            active_threshold *= 0.5
            no_improve += 1
            if no_improve >= 10:
                break
            continue

        alpha, new_obj = line_search(h, h_lp, obj)
        improvement = obj - new_obj

        if it % 10 == 0 or it <= 5:
            elapsed = time.time() - start
            print(f"  Iter {it:5d}: obj={new_obj:.15f}  alpha={alpha:.6f}  "
                  f"active={n_active}  delta={delta:.4f}  improve={improvement:.2e}  t={elapsed:.1f}s")
            sys.stdout.flush()

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
                active_threshold = max(active_threshold * 0.5, 1e-8)
            if no_improve >= no_improve_limit:
                print(f"  No improvement for {no_improve_limit} iters, stopping at iter {it}")
                break

        if it % checkpoint_every == 0:
            print(f"  [Checkpoint {it}] best_obj={best_obj:.15f}")
            sys.stdout.flush()

        # Early stop when converging (improvement below threshold)
        if improvement < 1e-6 and it > 20:
            print(f"  Converged at iter {it} (improvement={improvement:.2e} < 1e-10)")
            break

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Best obj={best_obj:.15f}")
    return best_h, best_obj


def run_with_restarts(n_restarts=10, n_iter_per_run=500):
    print("=== Initial run from tiled slp-1024 seed ===")
    best_h, best_obj = run_slp(n_iter=n_iter_per_run, checkpoint_every=50,
                                init_delta=0.05, no_improve_limit=50)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Initial best: {best_obj:.15f}")

    print(f"\n--- Perturbation restarts (best: {best_obj:.15f}) ---")
    # Use small scales to explore near the good basin, avoid jumping too far
    scales = [0.003, 0.001, 0.005, 0.002, 0.008, 0.0005, 0.01, 0.004, 0.007, 0.0015,
              0.003, 0.001, 0.005, 0.002, 0.008, 0.0005, 0.01, 0.004, 0.007, 0.0015]

    for restart in range(n_restarts):
        scale = scales[restart % len(scales)]
        np.random.seed(restart + 400)
        h_perturbed = best_h + np.random.randn(N) * scale
        h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
        h_perturbed = project_to_feasible(h_perturbed)

        obj_p, _ = compute_upper_bound(h_perturbed)
        print(f"\n=== Restart {restart+1}/{n_restarts}: scale={scale:.4f}, obj={obj_p:.15f} ===")
        sys.stdout.flush()

        h_new, obj_new = run_slp(n_iter=n_iter_per_run, checkpoint_every=50,
                                  seed_h=h_perturbed, init_delta=0.05, no_improve_limit=50)

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
    best_h, best_obj = run_with_restarts(n_restarts=20, n_iter_per_run=500)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Saved to {SOLUTION_PATH}")
    print(f"Final metric: {best_obj:.15f}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
