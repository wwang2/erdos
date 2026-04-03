"""
SLP optimization v2 for n=2048.
Seeds from current checkpoint (0.380892) and uses more active shifts (800)
to push past the local optima toward the n=1024 best of 0.380870.
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

# Increase active set cap for better coverage
MAX_ACTIVE = 800


def load_seed():
    ns = {}
    exec(SEED_PATH.read_text(), ns)
    return np.asarray(ns["h_values"], dtype=float)


def interpolate_to_n(h_src, n_target):
    n_src = len(h_src)
    x_src = np.arange(n_src) / n_src
    x_tgt = np.arange(n_target) / n_target
    h_tgt = np.interp(x_tgt, x_src, h_src)
    h_tgt = np.clip(h_tgt, 0.0, 1.0)
    s = h_tgt.sum()
    if s > 0:
        h_tgt *= (n_target / 2.0) / s
        h_tgt = np.clip(h_tgt, 0.0, 1.0)
    return h_tgt


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
        options={'disp': False, 'time_limit': 240.0, 'primal_feasibility_tolerance': 1e-9},
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


def run_slp(n_iter=1000, checkpoint_every=50, seed_h=None, init_delta=0.05,
            init_threshold=1e-3, no_improve_limit=80):
    if seed_h is not None:
        h = project_to_feasible(seed_h.copy())
        print(f"  Starting from seed: sum={h.sum():.6f}")
    else:
        # Load checkpoint if exists, else interpolate
        if CHECKPOINT_PATH.exists():
            h = np.load(CHECKPOINT_PATH)
            if len(h) != N:
                h_base = load_seed()
                h = interpolate_to_n(h_base, N)
                h = project_to_feasible(h)
        else:
            h_base = load_seed()
            h = interpolate_to_n(h_base, N)
            h = project_to_feasible(h)

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
            print(f"  Iter {it}: LP failed ({n_active} shifts), shrinking delta/threshold")
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
            np.save(CHECKPOINT_PATH, best_h)
            save_solution(best_h)
            print(f"  [Checkpoint {it}] best_obj={best_obj:.15f}")
            sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Best obj={best_obj:.15f}")
    return best_h, best_obj


def run_with_restarts(n_restarts=8, n_iter_per_run=600):
    print("=== Phase 1: Continue from checkpoint with 800 active shifts ===")
    best_h, best_obj = run_slp(n_iter=n_iter_per_run, checkpoint_every=50,
                                init_delta=0.5, init_threshold=1e-3,
                                no_improve_limit=80)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Phase 1 best: {best_obj:.15f}")

    print(f"\n--- Perturbation restarts (best: {best_obj:.15f}) ---")
    scales = [0.02, 0.01, 0.05, 0.005, 0.1, 0.003, 0.08, 0.001]

    for restart in range(n_restarts):
        scale = scales[restart % len(scales)]
        np.random.seed(restart + 300)
        h_perturbed = best_h + np.random.randn(N) * scale
        h_perturbed = np.clip(h_perturbed, 0.0, 1.0)
        h_perturbed = project_to_feasible(h_perturbed)

        obj_perturbed, _ = compute_upper_bound(h_perturbed)
        print(f"\n=== Restart {restart+1}/{n_restarts}: scale={scale:.4f}, "
              f"perturbed_obj={obj_perturbed:.15f} ===")
        sys.stdout.flush()

        h_new, obj_new = run_slp(n_iter=n_iter_per_run, checkpoint_every=50,
                                  seed_h=h_perturbed, init_delta=0.05,
                                  init_threshold=1e-3, no_improve_limit=80)

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
    best_h, best_obj = run_with_restarts(n_restarts=8, n_iter_per_run=600)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Saved to {SOLUTION_PATH}")
    print(f"Final metric: {best_obj:.15f}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
