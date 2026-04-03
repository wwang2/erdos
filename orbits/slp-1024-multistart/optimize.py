"""
Multi-start SLP optimization for Erdős Minimum Overlap Problem.

Strategy: Run SLP from diverse starting points to escape local optima.
Starting points:
  1. Perturbed SOTA (scales: 0.01, 0.02, 0.05, 0.1, 0.2)
  2. Random valid (random h in [0,1] with sum=512)
  3. Block construction (h=1 for first 512, 0 for rest)
  4. Alternating (0.3/0.7)
  5. Smooth sine
  6. Reversed SOTA
  7. Shuffled SOTA

Each start runs SLP for 300 iterations. Active shifts capped at 200 for speed.
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

# Paths
WORK_DIR = Path(__file__).parent
REPO_ROOT = WORK_DIR.parent.parent
SOTA_PATH = REPO_ROOT / "orbits/slp-1024/solution.py"
SOLUTION_PATH = WORK_DIR / "solution.py"

# Problem constants
N = 1024
N_HALF = N / 2.0  # 512.0


def load_sota():
    """Load slp-1024 SOTA solution."""
    ns = {}
    exec(SOTA_PATH.read_text(), ns)
    return np.asarray(ns["h_values"], dtype=float)


def compute_upper_bound(h):
    """Compute max_k (2/n) sum_i h_i (1 - h_{i+k})."""
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
    """Analytic gradient of F at correlation index idx w.r.t. h."""
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
    """Project h to [0,1] with sum = N/2."""
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
    """Solve LP to find h_lp that minimizes the max linearized overlap."""
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
        h_lp = np.clip(result.x[:n], 0.0, 1.0)
        return h_lp
    else:
        return None


def line_search(h, h_lp, current_obj, n_steps=30):
    """Grid search over alpha values to find best step size."""
    best_alpha = 0.0
    best_obj = current_obj

    alphas = np.logspace(0, -8, n_steps)
    for alpha in alphas:
        h_new = h + alpha * (h_lp - h)
        h_new = project_to_feasible(h_new)
        obj, _ = compute_upper_bound(h_new)
        if obj < best_obj:
            best_obj = obj
            best_alpha = alpha

    if best_alpha > 0:
        lo = best_alpha * 0.5
        hi = best_alpha * 2.0
        for alpha in np.linspace(lo, hi, 15):
            h_new = h + alpha * (h_lp - h)
            h_new = project_to_feasible(h_new)
            obj, _ = compute_upper_bound(h_new)
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha

    return best_alpha, best_obj


def save_solution(h, path=None):
    """Save h_values as a Python file."""
    if path is None:
        path = SOLUTION_PATH
    lines = ["import numpy as np\n\nh_values = np.array([\n"]
    for v in h:
        lines.append(f"  {v!r},\n")
    lines.append("])\n")
    path.write_text("".join(lines))


def run_slp(h_init, n_iter=300, label="", active_threshold=1e-3, max_active=200):
    """Run SLP optimization from a given starting point."""
    h = project_to_feasible(h_init.copy())
    obj, _ = compute_upper_bound(h)
    print(f"  [{label}] Start obj={obj:.15f}", flush=True)

    best_h = h.copy()
    best_obj = obj
    no_improve = 0
    delta = 0.05
    thresh = active_threshold
    start = time.time()

    for it in range(1, n_iter + 1):
        overlaps = get_all_overlaps(h)
        max_ov = overlaps.max()

        active_mask = overlaps >= max_ov - thresh
        shift_idx = np.where(active_mask)[0]
        n_active = len(shift_idx)

        # Cap active shifts for LP tractability
        if n_active > max_active:
            top_idx = np.argpartition(overlaps, -max_active)[-max_active:]
            shift_idx = top_idx[np.argsort(overlaps[top_idx])[::-1]]
            n_active = len(shift_idx)

        h_lp = solve_lp_subproblem(h, shift_idx, delta=delta)
        if h_lp is None:
            delta *= 0.5
            thresh *= 0.5
            no_improve += 1
            if no_improve >= 10:
                break
            continue

        alpha, new_obj = line_search(h, h_lp, obj)
        improvement = obj - new_obj

        if it % 10 == 0 or it <= 3:
            elapsed = time.time() - start
            print(f"  [{label}] iter={it:4d} obj={new_obj:.15f} alpha={alpha:.5f} "
                  f"active={n_active} delta={delta:.4f} improve={improvement:.2e} t={elapsed:.1f}s",
                  flush=True)

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
            if no_improve % 5 == 0:
                thresh = max(thresh * 0.5, 1e-8)
            if no_improve >= 50:
                break

        if improvement < 1e-14 and it > 50 and delta < 1e-5 and no_improve >= 20:
            break

    elapsed = time.time() - start
    print(f"  [{label}] Done in {elapsed:.1f}s. Best obj={best_obj:.15f}", flush=True)
    return best_h, best_obj


def make_starts(sota_h, rng):
    """Generate all diverse starting points."""
    starts = []

    # 1. Perturbed SOTA at various scales
    for scale in [0.01, 0.02, 0.05, 0.1, 0.2]:
        rng2 = np.random.default_rng(int(scale * 1000))
        noise = rng2.standard_normal(N) * scale
        h_p = np.clip(sota_h + noise, 0.0, 1.0)
        h_p = project_to_feasible(h_p)
        starts.append((f"perturbed_sota_scale={scale}", h_p))

    # 2. Random valid
    for seed_offset in range(5):
        rng2 = np.random.default_rng(42 + seed_offset)
        h_rand = rng2.uniform(0, 1, N)
        h_rand = project_to_feasible(h_rand)
        starts.append((f"random_valid_seed={42+seed_offset}", h_rand))

    # 3. Block construction: 1 for first 512, 0 for rest
    h_block = np.zeros(N)
    h_block[:512] = 1.0
    h_block = project_to_feasible(h_block)
    starts.append(("block_construction", h_block))

    # 4. Alternating 0.3/0.7
    h_alt = np.where(np.arange(N) % 2 == 0, 0.3, 0.7)
    h_alt = project_to_feasible(h_alt)
    starts.append(("alternating_0.3_0.7", h_alt))

    # 5. Smooth sine
    h_sine = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / N)
    h_sine = project_to_feasible(h_sine)
    starts.append(("smooth_sine", h_sine))

    # 6. Reversed SOTA
    h_rev = sota_h[::-1].copy()
    h_rev = project_to_feasible(h_rev)
    starts.append(("reversed_sota", h_rev))

    # 7. Shuffled SOTA
    h_shuf = sota_h.copy()
    rng.shuffle(h_shuf)
    h_shuf = project_to_feasible(h_shuf)
    starts.append(("shuffled_sota", h_shuf))

    return starts


def main():
    rng = np.random.default_rng(0)

    print("Loading SOTA solution from slp-1024...", flush=True)
    sota_h = load_sota()
    sota_obj, _ = compute_upper_bound(sota_h)
    print(f"SOTA objective: {sota_obj:.15f}", flush=True)

    starts = make_starts(sota_h, rng)
    print(f"\nTotal starting points: {len(starts)}", flush=True)

    best_h = sota_h.copy()
    best_obj = sota_obj
    best_label = "slp-1024-sota"

    results = []

    for i, (label, h_init) in enumerate(starts):
        print(f"\n=== Start {i+1}/{len(starts)}: {label} ===", flush=True)
        init_obj, _ = compute_upper_bound(h_init)
        print(f"  Initial obj={init_obj:.15f}", flush=True)

        h_result, result_obj = run_slp(h_init, n_iter=300, label=label, max_active=1000)
        results.append((label, init_obj, result_obj))

        if result_obj < best_obj:
            best_obj = result_obj
            best_h = h_result.copy()
            best_label = label
            print(f"*** NEW BEST: {best_obj:.15f} from {label} ***", flush=True)
            save_solution(best_h)
        else:
            print(f"  No improvement over best ({best_obj:.15f})", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL BEST: {best_obj:.15f} from '{best_label}'", flush=True)
    print(f"{'='*60}", flush=True)

    save_solution(best_h)
    print(f"Saved to {SOLUTION_PATH}", flush=True)

    # Print results table
    print("\nResults summary:", flush=True)
    print(f"{'Starting Point':<45} {'Init Obj':>20} {'Final Obj':>20}", flush=True)
    print("-" * 87, flush=True)
    for label, init_obj, final_obj in results:
        marker = " ***" if final_obj < sota_obj else ""
        print(f"{label:<45} {init_obj:>20.15f} {final_obj:>20.15f}{marker}", flush=True)

    return best_h, best_obj, results


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    best_h, best_obj, results = main()
