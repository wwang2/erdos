"""
Multi-start SLP optimization for Erdős Minimum Overlap Problem at N=2048.

Strategy:
1. Tile slp-1024-multistart solution: h_2048 = np.repeat(h_1024, 2)
   This preserves cross-correlation structure exactly.
2. Run initial SLP to convergence from the tiled seed.
3. Run 20+ perturbation restarts with scales [0.001, 0.003, 0.005, 0.01, 0.02].
4. Active shifts capped at 500 to keep LP tractable at n=2048.
5. Use FFT-based correlation for speed.
6. Early stopping when improvement < 1e-11 for 20 consecutive iters.
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
N = 2048
N_HALF = N / 2.0  # 1024.0


def load_seed():
    """Load slp-1024-multistart solution and tile to 2048."""
    ns = {}
    exec(SEED_PATH.read_text(), ns)
    h_1024 = np.asarray(ns["h_values"], dtype=float)
    h_2048 = np.repeat(h_1024, 2)
    return h_2048


def compute_upper_bound(h):
    """Compute max_k (2/n) sum_i h_i (1 - h_{i+k}) using FFT."""
    n = len(h)
    # Use FFT for O(n log n) correlation
    H = np.fft.rfft(h, n=2*n)
    OneMinusH = np.fft.rfft(1 - h, n=2*n)
    conv = np.fft.irfft(H * OneMinusH.conj())[:2*n-1]
    # np.correlate(h, 1-h, 'full') = conv_full but FFT gives circular; trim to linear
    # Actually use direct for correctness on small array
    conv = np.correlate(h, 1 - h, mode='full')
    idx = int(np.argmax(conv))
    return float(conv[idx] / n * 2), idx


def get_all_overlaps(h):
    """Return (2/n)*correlate(h, 1-h, 'full') array of length 2n-1."""
    n = len(h)
    conv = np.correlate(h, 1 - h, mode='full')
    return conv / n * 2


def get_all_overlaps_fft(h):
    """FFT-based correlation for speed at large n."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, n=fsize)
    G = np.fft.rfft(1 - h, n=fsize)
    # correlate(h, g) = ifft(fft(h) * conj(fft(g)))
    conv_full = np.fft.irfft(H * np.conj(G), n=fsize)[:size]
    return conv_full / n * 2


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
    overlaps = get_all_overlaps_fft(h)

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
        options={'disp': False, 'time_limit': 300.0, 'primal_feasibility_tolerance': 1e-9},
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


def run_slp(h_init, n_iter=200, label="", active_threshold=1e-3, max_active=500):
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
        overlaps = get_all_overlaps_fft(h)
        max_ov = overlaps.max()

        active_mask = overlaps >= max_ov - thresh
        shift_idx = np.where(active_mask)[0]
        n_active = len(shift_idx)

        # Cap active shifts for LP tractability at n=2048
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
            if no_improve >= 30:
                print(f"  [{label}] Early stop at iter {it} (no_improve={no_improve})", flush=True)
                break

        # Aggressive early stop: improvement too tiny
        if improvement < 1e-11 and it > 30 and delta < 1e-4:
            print(f"  [{label}] Early stop at iter {it} (improve={improvement:.2e}, delta={delta:.2e})",
                  flush=True)
            break

    elapsed = time.time() - start
    print(f"  [{label}] Done in {elapsed:.1f}s. Best obj={best_obj:.15f}", flush=True)
    return best_h, best_obj


def main():
    print("Loading slp-1024-multistart solution and tiling to 2048...", flush=True)
    h_seed = load_seed()
    assert len(h_seed) == N, f"Expected {N} elements, got {len(h_seed)}"
    seed_obj, _ = compute_upper_bound(h_seed)
    print(f"Tiled seed objective: {seed_obj:.15f}", flush=True)
    print(f"Seed sum: {h_seed.sum():.6f} (expected {N_HALF})", flush=True)

    # Check for existing checkpoint
    best_h = h_seed.copy()
    best_obj = seed_obj
    best_label = "tiled-seed"

    if CHECKPOINT_PATH.exists():
        h_ckpt = np.load(CHECKPOINT_PATH)
        if len(h_ckpt) == N:
            obj_ckpt, _ = compute_upper_bound(h_ckpt)
            if obj_ckpt < best_obj:
                print(f"Resuming from checkpoint (obj={obj_ckpt:.15f})", flush=True)
                best_h = h_ckpt.copy()
                best_obj = obj_ckpt
                best_label = "checkpoint"

    # === Initial run from tiled seed ===
    print("\n=== Initial run from tiled seed ===", flush=True)
    h_result, result_obj = run_slp(best_h, n_iter=200, label="tiled-seed", max_active=500)
    if result_obj < best_obj:
        best_obj = result_obj
        best_h = h_result.copy()
        best_label = "tiled-seed"
        print(f"*** NEW BEST: {best_obj:.15f} ***", flush=True)
    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)

    # === Perturbation restarts ===
    # 5 scales x 5 repeats = 25 restarts
    perturbation_scales = [0.001, 0.003, 0.005, 0.01, 0.02] * 5
    n_restarts = len(perturbation_scales)
    print(f"\n--- Starting {n_restarts} perturbation restarts (best so far: {best_obj:.15f}) ---",
          flush=True)

    results = [("tiled-seed", seed_obj, best_obj)]

    for restart_idx, scale in enumerate(perturbation_scales):
        seed = restart_idx + 1000
        rng2 = np.random.default_rng(seed)
        noise = rng2.standard_normal(N) * scale
        h_perturbed = np.clip(best_h + noise, 0.0, 1.0)
        h_perturbed = project_to_feasible(h_perturbed)

        init_obj, _ = compute_upper_bound(h_perturbed)
        label = f"restart_{restart_idx+1}_scale={scale}"
        print(f"\n=== Restart {restart_idx+1}/{n_restarts}: scale={scale:.4f}, "
              f"init_obj={init_obj:.15f} ===", flush=True)

        h_result, result_obj = run_slp(h_perturbed, n_iter=200, label=label, max_active=500)
        results.append((label, init_obj, result_obj))

        if result_obj < best_obj:
            best_obj = result_obj
            best_h = h_result.copy()
            best_label = label
            print(f"*** NEW BEST: {best_obj:.15f} from {label} ***", flush=True)
            save_solution(best_h)
            np.save(CHECKPOINT_PATH, best_h)
        else:
            print(f"  No improvement: {result_obj:.15f} >= {best_obj:.15f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL BEST: {best_obj:.15f} from '{best_label}'", flush=True)
    print(f"{'='*60}", flush=True)

    save_solution(best_h)
    np.save(CHECKPOINT_PATH, best_h)
    print(f"Saved to {SOLUTION_PATH}", flush=True)

    # Print results summary
    print("\nResults summary:", flush=True)
    print(f"{'Starting Point':<50} {'Init Obj':>20} {'Final Obj':>20}", flush=True)
    print("-" * 92, flush=True)
    for label, init_obj, final_obj in results:
        marker = " ***" if final_obj < seed_obj else ""
        print(f"{label:<50} {init_obj:>20.15f} {final_obj:>20.15f}{marker}", flush=True)

    return best_h, best_obj


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    best_h, best_obj = main()
