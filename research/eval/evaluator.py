"""
Evaluator for the Erdős minimum overlap problem.

Computes the upper bound on C₅ from a step function construction.
A valid step function h : [0,2] → [0,1] is represented as a numpy array
where values are in [0,1] and sum to n/2.

The upper bound is: max_k (2/n) Σ_i h_i (1 - h_{i+k})
computed via cross-correlation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def validate_solution(h: np.ndarray, atol: float = 1e-6) -> tuple[bool, list[str]]:
    """Check that h is a valid step function construction.

    Returns (is_valid, list_of_errors).
    """
    errors = []
    if not isinstance(h, np.ndarray):
        errors.append(f"Expected numpy array, got {type(h)}")
        return False, errors

    if h.ndim != 1:
        errors.append(f"Expected 1D array, got {h.ndim}D")
        return False, errors

    n = len(h)
    if n < 2:
        errors.append(f"Need at least 2 steps, got {n}")

    if np.any(h < -atol):
        errors.append(f"Values below 0: min = {h.min():.15e}")

    if np.any(h > 1 + atol):
        errors.append(f"Values above 1: max = {h.max():.15e}")

    target_sum = n / 2.0
    actual_sum = float(np.sum(h))
    if not np.isclose(actual_sum, target_sum, atol=atol):
        errors.append(f"Sum = {actual_sum:.10f}, expected {target_sum:.1f}")

    return len(errors) == 0, errors


def compute_upper_bound(h: np.ndarray) -> float:
    """Compute the upper bound on C₅ from a step function.

    Returns max_k (2/n) Σ_i h_i(1 - h_{i+k}) via cross-correlation.
    """
    n = len(h)
    convolution = np.correlate(h, 1 - h, mode='full')
    return float(np.max(convolution) / n * 2)


def evaluate(h: np.ndarray, atol: float = 1e-6) -> dict:
    """Full evaluation: validate and compute metric.

    Returns dict with keys: valid, errors, n_steps, metric, sum.
    """
    is_valid, errors = validate_solution(h, atol=atol)
    n = len(h)
    metric = compute_upper_bound(h) if is_valid else None

    return {
        "valid": is_valid,
        "errors": errors,
        "n_steps": n,
        "metric": metric,
        "sum": float(np.sum(h)),
    }


def load_solution(path: str) -> np.ndarray:
    """Load a solution from a .py or .npy file."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".py":
        ns = {}
        exec(p.read_text(), ns)
        if "h_values" not in ns:
            raise ValueError(f"{path} does not define h_values")
        return np.asarray(ns["h_values"])
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def run_golden_tests(config_path: str = None) -> bool:
    """Run golden example tests from config.yaml."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    config_dir = Path(config_path).parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    golden = config.get("golden_examples", [])
    if not golden:
        print("No golden examples found.")
        return True

    all_passed = True
    for ex in golden:
        name = ex["name"]
        print(f"  [{name}]", end=" ")

        input_path = str(config_dir / ex["input"]) if "input" in ex else None

        if ex.get("expected") == "error":
            # Expect validation failure
            try:
                h = load_solution(input_path)
                result = evaluate(h)
                if result["valid"]:
                    print(f"FAIL — expected error but got valid solution")
                    all_passed = False
                else:
                    print(f"PASS — correctly rejected: {result['errors'][0]}")
            except Exception as e:
                print(f"PASS — correctly raised: {e}")
            continue

        h = load_solution(input_path)
        result = evaluate(h)

        if not result["valid"]:
            print(f"FAIL — invalid: {result['errors']}")
            all_passed = False
            continue

        lo, hi = ex["expected_range"]
        if lo <= result["metric"] <= hi:
            print(f"PASS — metric={result['metric']:.15f} in [{lo}, {hi}]")
        else:
            print(f"FAIL — metric={result['metric']:.15f} not in [{lo}, {hi}]")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erdős minimum overlap evaluator")
    parser.add_argument("--test-golden", action="store_true", help="Run golden example tests")
    parser.add_argument("--evaluate", type=str, help="Evaluate a solution file (.py or .npy)")
    args = parser.parse_args()

    if args.test_golden:
        print("Running golden example tests...")
        passed = run_golden_tests()
        sys.exit(0 if passed else 1)

    if args.evaluate:
        h = load_solution(args.evaluate)
        result = evaluate(h)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if result["valid"] else 1)

    parser.print_help()
