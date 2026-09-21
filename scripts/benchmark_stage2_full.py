from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Callable, Iterable

import numpy as np

from tfe_ccp.dc_bounds import (
    stage2_lower_common_box,
    stage2_lower_exact_box,
    stage2_lower_q_box,
    stage2_lower_rough_box,
    stage2_true_softmax_box_bounds,
    stage2_upper_box,
)
from tfe_ccp.export import extract_parameters, load_checkpoint
from tfe_ccp.optimization.common import MPCBounds
from tfe_ccp.optimization.exact_ccp import ExactCCPConfig
from tfe_ccp.optimization.exact_ccp_probe import ExactConicCCPProbe, PBoxSnapshot


LOWER_METHODS = ("p_exact", "common_o_n", "q_box", "rough")
FAST_LOWER_METHODS = ("common_o_n", "q_box", "rough")
ALL_METHODS = LOWER_METHODS + ("u_upper",)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_mean(values: Iterable[float]) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _safe_median(values: Iterable[float]) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    return float(statistics.median(finite)) if finite else float("nan")


def _safe_max(values: Iterable[float]) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    return max(finite) if finite else float("nan")


def _time_call(function: Callable[[], np.ndarray], repeats: int) -> tuple[np.ndarray, float]:
    """Return function value and median-ish average wall time in milliseconds.

    The function is warmed once. Then the repeated calls are timed as one block,
    which is more stable than timing very small NumPy functions individually.
    """

    repeats = max(int(repeats), 1)
    value = np.asarray(function(), dtype=np.float64)
    start = time.perf_counter_ns()
    for _ in range(repeats):
        value = np.asarray(function(), dtype=np.float64)
    elapsed_ns = time.perf_counter_ns() - start
    return value, elapsed_ns / repeats / 1e6


def _parse_n_values(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 2:
            raise argparse.ArgumentTypeError("All scaling N values must be >= 2.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one scaling N value is required.")
    return sorted(set(values))


def _evaluate_methods(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
    *,
    maximum_exact_vertices: int,
    exact_repeats: int,
    fast_repeats: int,
    include_exact: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, np.ndarray]:
    true_lower, true_upper = stage2_true_softmax_box_bounds(p_lower, p_upper, d_key)

    methods: dict[str, tuple[Callable[[], np.ndarray], int]] = {
        "common_o_n": (
            lambda: stage2_lower_common_box(p_lower, p_upper, p0, d_key),
            fast_repeats,
        ),
        "q_box": (
            lambda: stage2_lower_q_box(p_lower, p_upper, p0, d_key),
            fast_repeats,
        ),
        "rough": (
            lambda: stage2_lower_rough_box(p_lower, p_upper, p0, d_key),
            fast_repeats,
        ),
        "u_upper": (
            lambda: stage2_upper_box(p_lower, p_upper, p0, d_key),
            fast_repeats,
        ),
    }
    if include_exact:
        methods = {
            "p_exact": (
                lambda: stage2_lower_exact_box(
                    p_lower,
                    p_upper,
                    p0,
                    d_key,
                    maximum_vertices=maximum_exact_vertices,
                ),
                exact_repeats,
            ),
            **methods,
        }

    values: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}
    for name, (function, repeats) in methods.items():
        value, runtime_ms = _time_call(function, repeats)
        values[name] = value
        runtimes[name] = runtime_ms
    return values, runtimes, true_lower, true_upper


def _geometry_from_snapshot(snapshot: PBoxSnapshot) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in range(snapshot.p_lower.shape[0]):
        lower = snapshot.p_lower[row]
        upper = snapshot.p_upper[row]
        widths = upper - lower
        centres = 0.5 * (lower + upper)
        common_scalar_width = float(np.max(upper) - np.min(lower))
        base_width = float(np.max(widths))
        expansion = common_scalar_width / base_width if base_width > 1e-14 else 1.0
        rows.append(
            {
                "iteration": snapshot.iteration,
                "row": row,
                "simplex_gamma1": snapshot.gamma1,
                "row_width_mean": float(np.mean(widths)),
                "row_width_std": float(np.std(widths)),
                "centre_mean": float(np.mean(centres)),
                "centre_std": float(np.std(centres)),
                "centre_range": float(np.max(centres) - np.min(centres)),
                "common_scalar_width": common_scalar_width,
                "common_scalar_expansion_ratio": expansion,
            }
        )
    return rows


def _real_benchmark(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[PBoxSnapshot], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], object]:
    model, scalers = load_checkpoint(args.checkpoint)
    params = extract_parameters(model, scalers)
    cfg = params.config

    largest_simplex = max(cfg.sequence_length**2 + 1, cfg.sequence_length * cfg.d_key + 1)
    if largest_simplex > args.maximum_simplex_vertices:
        raise SystemExit(
            f"Baseline exact simplex needs {largest_simplex} vertices, exceeding "
            f"--maximum-simplex-vertices={args.maximum_simplex_vertices}."
        )
    if (1 << cfg.sequence_length) > args.maximum_exact_box_vertices:
        raise SystemExit(
            f"Real-box exact P reference requires 2^{cfg.sequence_length} row "
            f"vertices, exceeding --maximum-exact-box-vertices="
            f"{args.maximum_exact_box_vertices}."
        )

    probe = ExactConicCCPProbe(
        params,
        MPCBounds.constant(cfg.n_inputs, -2.0, 2.0, -0.5, 0.5),
        ExactCCPConfig(
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            solver=args.solver,
            maximum_vertices=args.maximum_simplex_vertices,
            verbose=args.solver_verbose,
        ),
    )
    past_u = np.zeros((cfg.past_window, cfg.n_inputs))
    past_y = np.zeros((cfg.past_window, cfg.n_outputs))
    reference = np.full((cfg.horizon, cfg.n_outputs), args.reference)
    initial = np.zeros((cfg.horizon, cfg.n_inputs))
    result = probe.solve(past_u, past_y, reference, np.zeros(cfg.n_inputs), initial)
    snapshots = result.p_box_history
    if not snapshots:
        raise RuntimeError("Probe returned no P-box snapshots.")

    np.savez(
        output_dir / "real_p_box_history.npz",
        p0=np.stack([s.p0 for s in snapshots]),
        p_optimized=np.stack([s.p_optimized for s in snapshots]),
        p_lower=np.stack([s.p_lower for s in snapshots]),
        p_upper=np.stack([s.p_upper for s in snapshots]),
        alpha1=np.stack([s.alpha1 for s in snapshots]),
        beta1=np.array([s.beta1 for s in snapshots]),
        gamma1=np.array([s.gamma1 for s in snapshots]),
        objective=np.array([s.objective for s in snapshots]),
        steps=np.array(result.step_history),
    )

    entries: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    geometry: list[dict[str, object]] = []
    sanity: list[dict[str, object]] = []

    for snapshot in snapshots:
        values, times, true_lower, true_upper = _evaluate_methods(
            snapshot.p_lower,
            snapshot.p_upper,
            snapshot.p0,
            cfg.d_key,
            maximum_exact_vertices=args.maximum_exact_box_vertices,
            exact_repeats=args.real_exact_repeats,
            fast_repeats=args.real_fast_repeats,
            include_exact=True,
        )

        for method, runtime_ms in times.items():
            runtimes.append(
                {
                    "iteration": snapshot.iteration,
                    "method": method,
                    "runtime_ms_full_matrix": runtime_ms,
                    "runtime_us_per_output_entry": runtime_ms * 1000.0 / (cfg.sequence_length**2),
                    "sequence_length": cfg.sequence_length,
                }
            )

        geometry.extend(_geometry_from_snapshot(snapshot))
        sanity.append(
            {
                "iteration": snapshot.iteration,
                "solver_status": snapshot.status,
                "objective": snapshot.objective,
                "p0_inside_new_box": snapshot.p0_inside_box,
                "p0_box_max_violation": snapshot.p0_box_max_violation,
                "p_optimized_inside_box": snapshot.p_optimized_inside_box,
                "p_optimized_box_max_violation": snapshot.p_optimized_box_max_violation,
                "p_optimized_inside_simplex": snapshot.p_optimized_inside_simplex,
                "p_optimized_simplex_lower_max_violation": snapshot.p_optimized_simplex_lower_max_violation,
                "p_optimized_simplex_sum_violation": snapshot.p_optimized_simplex_sum_violation,
            }
        )

        p_exact = values["p_exact"]
        u_upper = values["u_upper"]
        for row in range(cfg.sequence_length):
            for column in range(cfg.sequence_length):
                true_lo = float(true_lower[row, column])
                true_hi = float(true_upper[row, column])
                exact_lo = float(p_exact[row, column])
                for method in LOWER_METHODS:
                    value = float(values[method][row, column])
                    entries.append(
                        {
                            "iteration": snapshot.iteration,
                            "row": row,
                            "column": column,
                            "method": method,
                            "lower_bound": value,
                            "true_softmax_lower": true_lo,
                            "p_exact_L_lower": exact_lo,
                            "gap_to_true": true_lo - value,
                            "difference_from_p_exact": exact_lo - value,
                            "r_box_width": float(u_upper[row, column] - value),
                            "valid_vs_true": bool(value <= true_lo + args.validity_tol),
                            "sequence_length": cfg.sequence_length,
                        }
                    )
                upper_value = float(u_upper[row, column])
                entries.append(
                    {
                        "iteration": snapshot.iteration,
                        "row": row,
                        "column": column,
                        "method": "u_upper",
                        "lower_bound": float("nan"),
                        "true_softmax_lower": true_lo,
                        "p_exact_L_lower": exact_lo,
                        "gap_to_true": upper_value - true_hi,
                        "difference_from_p_exact": float("nan"),
                        "r_box_width": float("nan"),
                        "valid_vs_true": bool(upper_value >= true_hi - args.validity_tol),
                        "sequence_length": cfg.sequence_length,
                    }
                )

    return snapshots, entries, runtimes, geometry, sanity, result


def _summarise_real(
    entries: list[dict[str, object]],
    runtimes: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        method_entries = [r for r in entries if r["method"] == method]
        method_times = [float(r["runtime_ms_full_matrix"]) for r in runtimes if r["method"] == method]
        gaps = [float(r["gap_to_true"]) for r in method_entries]
        differences = [float(r["difference_from_p_exact"]) for r in method_entries]
        widths = [float(r["r_box_width"]) for r in method_entries]
        rows.append(
            {
                "method": method,
                "mean_gap_to_true": _safe_mean(gaps),
                "median_gap_to_true": _safe_median(gaps),
                "worst_gap_to_true": _safe_max(gaps),
                "mean_difference_from_p_exact": _safe_mean(differences),
                "mean_r_box_width": _safe_mean(widths),
                "invalid_count_vs_true": sum(not bool(r["valid_vs_true"]) for r in method_entries),
                "median_runtime_ms_full_matrix": _safe_median(method_times),
                "mean_runtime_ms_full_matrix": _safe_mean(method_times),
            }
        )
    return rows


def _fit_synthetic_geometry(snapshots: list[PBoxSnapshot]) -> dict[str, float | list[float]]:
    centres = np.concatenate(
        [0.5 * (s.p_lower + s.p_upper).reshape(-1) for s in snapshots]
    )
    offsets = np.concatenate(
        [(s.p0 - 0.5 * (s.p_lower + s.p_upper)).reshape(-1) for s in snapshots]
    )
    widths = np.array([s.gamma1 for s in snapshots], dtype=np.float64)
    return {
        "centre_mean": float(np.mean(centres)),
        "centre_std": float(max(np.std(centres), 1e-6)),
        "offset_mean": float(np.mean(offsets)),
        "offset_std": float(max(np.std(offsets), 1e-6)),
        "width_values": [float(max(w, 1e-8)) for w in widths],
        "median_width": float(max(np.median(widths), 1e-8)),
    }


def _synthetic_box(
    rng: np.random.Generator,
    n: int,
    calibration: dict[str, float | list[float]],
    width_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centre_mean = float(calibration["centre_mean"])
    centre_std = float(calibration["centre_std"])
    offset_mean = float(calibration["offset_mean"])
    offset_std = float(calibration["offset_std"])
    widths = np.asarray(calibration["width_values"], dtype=np.float64)
    width = float(rng.choice(widths)) * width_scale
    width = max(width, 1e-8)

    centre = rng.normal(centre_mean, centre_std, size=(n, n))
    p_lower = centre - 0.5 * width
    p_upper = centre + 0.5 * width
    p0 = centre + rng.normal(offset_mean, offset_std, size=(n, n))
    return p_lower, p_upper, p0, width


def _scaling_benchmark(
    args: argparse.Namespace,
    output_dir: Path,
    snapshots: list[PBoxSnapshot],
    d_key: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    calibration = _fit_synthetic_geometry(snapshots)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for n in args.scaling_n:
        exact_available = (
            n <= args.scaling_exact_max_n
            and (1 << n) <= args.maximum_exact_box_vertices
        )
        for trial in range(args.synthetic_trials):
            p_lower, p_upper, p0, width = _synthetic_box(
                rng, n, calibration, args.synthetic_width_scale
            )
            values, times, true_lower, true_upper = _evaluate_methods(
                p_lower,
                p_upper,
                p0,
                d_key,
                maximum_exact_vertices=args.maximum_exact_box_vertices,
                exact_repeats=args.scaling_exact_repeats,
                fast_repeats=args.scaling_fast_repeats,
                include_exact=exact_available,
            )
            p_exact = values.get("p_exact")
            u_upper = values["u_upper"]

            for method in ("p_exact", "common_o_n", "q_box", "rough"):
                if method not in values:
                    continue
                value = values[method]
                gap = true_lower - value
                difference = (
                    p_exact - value if p_exact is not None else np.full_like(value, np.nan)
                )
                rows.append(
                    {
                        "N": n,
                        "trial": trial,
                        "method": method,
                        "exact_available": exact_available,
                        "synthetic_common_width": width,
                        "runtime_ms_full_matrix": times[method],
                        "runtime_us_per_output_entry": times[method] * 1000.0 / (n * n),
                        "mean_gap_to_true": float(np.mean(gap)),
                        "median_gap_to_true": float(np.median(gap)),
                        "worst_gap_to_true": float(np.max(gap)),
                        "mean_difference_from_p_exact": float(np.nanmean(difference)) if p_exact is not None else float("nan"),
                        "mean_r_box_width": float(np.mean(u_upper - value)),
                        "invalid_count_vs_true": int(np.count_nonzero(value > true_lower + args.validity_tol)),
                    }
                )

            upper_gap = u_upper - true_upper
            rows.append(
                {
                    "N": n,
                    "trial": trial,
                    "method": "u_upper",
                    "exact_available": exact_available,
                    "synthetic_common_width": width,
                    "runtime_ms_full_matrix": times["u_upper"],
                    "runtime_us_per_output_entry": times["u_upper"] * 1000.0 / (n * n),
                    "mean_gap_to_true": float(np.mean(upper_gap)),
                    "median_gap_to_true": float(np.median(upper_gap)),
                    "worst_gap_to_true": float(np.max(upper_gap)),
                    "mean_difference_from_p_exact": float("nan"),
                    "mean_r_box_width": float("nan"),
                    "invalid_count_vs_true": int(np.count_nonzero(u_upper < true_upper - args.validity_tol)),
                }
            )
        print(f"  scaling N={n} complete (exact={'yes' if exact_available else 'no'})")

    summary: list[dict[str, object]] = []
    for n in args.scaling_n:
        for method in ALL_METHODS:
            selected = [r for r in rows if r["N"] == n and r["method"] == method]
            if not selected:
                continue
            summary.append(
                {
                    "N": n,
                    "method": method,
                    "trials": len(selected),
                    "median_runtime_ms_full_matrix": _safe_median(r["runtime_ms_full_matrix"] for r in selected),
                    "mean_runtime_ms_full_matrix": _safe_mean(r["runtime_ms_full_matrix"] for r in selected),
                    "median_runtime_us_per_output_entry": _safe_median(r["runtime_us_per_output_entry"] for r in selected),
                    "mean_gap_to_true": _safe_mean(r["mean_gap_to_true"] for r in selected),
                    "worst_gap_to_true": _safe_max(r["worst_gap_to_true"] for r in selected),
                    "mean_difference_from_p_exact": _safe_mean(r["mean_difference_from_p_exact"] for r in selected),
                    "mean_r_box_width": _safe_mean(r["mean_r_box_width"] for r in selected),
                    "invalid_count_vs_true": int(sum(int(r["invalid_count_vs_true"]) for r in selected)),
                }
            )

    with (output_dir / "synthetic_calibration.json").open("w") as handle:
        json.dump(calibration, handle, indent=2)
    return rows, summary, calibration


def _make_plots(
    output_dir: Path,
    real_entries: list[dict[str, object]],
    real_summary: list[dict[str, object]],
    geometry: list[dict[str, object]],
    sanity: list[dict[str, object]],
    scaling_summary: list[dict[str, object]],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; CSV/NPZ/report results were saved, "
            "but plots were skipped. Install with: python -m pip install matplotlib"
        )
        return False

    lower_methods = list(LOWER_METHODS)

    # Real-box conservatism.
    mean_gaps = [
        max(
            next(float(r["mean_gap_to_true"]) for r in real_summary if r["method"] == method),
            1e-16,
        )
        for method in lower_methods
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(lower_methods, mean_gaps)
    plt.yscale("log")
    plt.ylabel("Mean gap to exact true-softmax minimum (log scale)")
    plt.title("Real CCP P-boxes: Stage-2 lower-bound conservatism")
    plt.tight_layout()
    plt.savefig(output_dir / "real_conservatism_mean_gap.png", dpi=180)
    plt.close()

    # Distribution of real gaps.
    gap_data = [
        [
            max(float(r["gap_to_true"]), 1e-16)
            for r in real_entries
            if r["method"] == method
        ]
        for method in lower_methods
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(gap_data, tick_labels=lower_methods, showfliers=False)
    plt.yscale("log")
    plt.ylabel("Gap to exact true-softmax minimum (log scale)")
    plt.title("Real CCP P-boxes: gap distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "real_conservatism_boxplot.png", dpi=180)
    plt.close()

    # Real runtime.
    labels = list(ALL_METHODS)
    runtimes = [
        next(float(r["median_runtime_ms_full_matrix"]) for r in real_summary if r["method"] == method)
        for method in labels
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, runtimes)
    plt.ylabel("Median full N x N evaluation time [ms]")
    plt.title("Real CCP P-boxes: Stage-2 evaluation time")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "real_runtime.png", dpi=180)
    plt.close()

    # Geometry / common scalar expansion.
    iterations = sorted({int(r["iteration"]) for r in geometry})
    mean_expansion = [
        _safe_mean(
            float(r["common_scalar_expansion_ratio"])
            for r in geometry
            if int(r["iteration"]) == iteration
        )
        for iteration in iterations
    ]
    max_expansion = [
        _safe_max(
            float(r["common_scalar_expansion_ratio"])
            for r in geometry
            if int(r["iteration"]) == iteration
        )
        for iteration in iterations
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, mean_expansion, marker="o", label="mean row expansion")
    plt.plot(iterations, max_expansion, marker="x", label="max row expansion")
    plt.xlabel("CCP iteration")
    plt.ylabel("Common-scalar width / simplex-box width")
    plt.title("Real CCP P-box geometry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "real_common_box_expansion.png", dpi=180)
    plt.close()

    # Probe sanity: optimized P should be in simplex/box up to solver tolerance.
    x = [int(r["iteration"]) for r in sanity]
    box_v = [max(float(r["p_optimized_box_max_violation"]), 1e-18) for r in sanity]
    simplex_v = [
        max(
            float(r["p_optimized_simplex_lower_max_violation"]),
            float(r["p_optimized_simplex_sum_violation"]),
            1e-18,
        )
        for r in sanity
    ]
    plt.figure(figsize=(8, 5))
    plt.semilogy(x, box_v, marker="o", label="P(U^{k+1}) box violation")
    plt.semilogy(x, simplex_v, marker="x", label="P(U^{k+1}) simplex violation")
    plt.xlabel("CCP iteration")
    plt.ylabel("Maximum violation")
    plt.title("Probe sanity checks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "real_probe_sanity.png", dpi=180)
    plt.close()

    if scaling_summary:
        methods_present = [
            method for method in ALL_METHODS if any(r["method"] == method for r in scaling_summary)
        ]
        plt.figure(figsize=(8, 5))
        for method in methods_present:
            rows = sorted(
                [r for r in scaling_summary if r["method"] == method],
                key=lambda r: int(r["N"]),
            )
            plt.semilogy(
                [int(r["N"]) for r in rows],
                [max(float(r["median_runtime_ms_full_matrix"]), 1e-12) for r in rows],
                marker="o",
                label=method,
            )
        plt.xlabel("Sequence length N")
        plt.ylabel("Median full N x N evaluation time [ms]")
        plt.title("Stage-2 runtime scaling")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "scaling_runtime.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for method in lower_methods:
            rows = sorted(
                [r for r in scaling_summary if r["method"] == method],
                key=lambda r: int(r["N"]),
            )
            if not rows:
                continue
            plt.semilogy(
                [int(r["N"]) for r in rows],
                [max(float(r["mean_gap_to_true"]), 1e-16) for r in rows],
                marker="o",
                label=method,
            )
        plt.xlabel("Sequence length N")
        plt.ylabel("Mean gap to exact true-softmax minimum")
        plt.title("Synthetic common-width boxes: conservatism vs N")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "scaling_conservatism.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for method in lower_methods:
            rows = sorted(
                [r for r in scaling_summary if r["method"] == method],
                key=lambda r: int(r["N"]),
            )
            if not rows:
                continue
            plt.semilogy(
                [int(r["N"]) for r in rows],
                [max(float(r["mean_r_box_width"]), 1e-16) for r in rows],
                marker="o",
                label=method,
            )
        plt.xlabel("Sequence length N")
        plt.ylabel("Mean resulting R-box width (log scale)")
        plt.title("Synthetic common-width boxes: propagated R-box width")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "scaling_r_box_width.png", dpi=180)
        plt.close()

    return True


def _write_report(
    output_dir: Path,
    args: argparse.Namespace,
    real_summary: list[dict[str, object]],
    geometry: list[dict[str, object]],
    sanity: list[dict[str, object]],
    scaling_summary: list[dict[str, object]],
    plots_generated: bool,
) -> None:
    def row_for(method: str) -> dict[str, object]:
        return next(r for r in real_summary if r["method"] == method)

    lines: list[str] = []
    lines.append("# Stage-2 softmax bound benchmark")
    lines.append("")
    lines.append("This report is generated automatically by `benchmark_stage2_full.py`.")
    lines.append("It benchmarks Stage 2 only; it is not yet a closed-loop box-CCP controller comparison.")
    lines.append("")
    lines.append("## Real CCP P-box experiment")
    lines.append("")
    lines.append("The baseline simplex CCP is run first. Each optimized Stage-1 simplex is replaced only for diagnostic purposes by its exact axis-aligned bounding box. All Stage-2 methods are then evaluated on the same P box.")
    lines.append("")
    lines.append("| method | mean gap to true min | worst gap | invalid | median full-matrix ms | mean R-box width |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in ALL_METHODS:
        row = row_for(method)
        lines.append(
            f"| {method} | {float(row['mean_gap_to_true']):.6e} | "
            f"{float(row['worst_gap_to_true']):.6e} | "
            f"{int(row['invalid_count_vs_true'])} | "
            f"{float(row['median_runtime_ms_full_matrix']):.6f} | "
            f"{float(row['mean_r_box_width']):.6e} |"
        )
    lines.append("")

    mean_expansion = _safe_mean(float(r["common_scalar_expansion_ratio"]) for r in geometry)
    max_expansion = _safe_max(float(r["common_scalar_expansion_ratio"]) for r in geometry)
    lines.append(
        f"Common-scalar enclosure width / original simplex-box width: mean {mean_expansion:.3f}, max {max_expansion:.3f}."
    )
    lines.append("")

    max_box_violation = _safe_max(float(r["p_optimized_box_max_violation"]) for r in sanity)
    max_simplex_violation = _safe_max(
        max(
            float(r["p_optimized_simplex_lower_max_violation"]),
            float(r["p_optimized_simplex_sum_violation"]),
        )
        for r in sanity
    )
    lines.append("## Probe sanity")
    lines.append("")
    lines.append(
        f"Maximum P(U^(k+1)) bounding-box violation: {max_box_violation:.3e}.  "
        f"Maximum Stage-1 simplex violation: {max_simplex_violation:.3e}."
    )
    lines.append("The linearisation point P^k is reported separately and is not required to belong to the newly optimized simplex/box.")
    lines.append("")

    if scaling_summary:
        lines.append("## Synthetic scaling experiment")
        lines.append("")
        lines.append("Synthetic boxes preserve the common-width / different-centres geometry observed in the baseline Stage-1 simplex boxes. Their centre and P^k-offset statistics are calibrated from the real probe.")
        lines.append("")
        lines.append("| N | method | median full-matrix ms | mean gap to true min | invalid |")
        lines.append("|---:|---|---:|---:|---:|")
        for row in scaling_summary:
            lines.append(
                f"| {int(row['N'])} | {row['method']} | "
                f"{float(row['median_runtime_ms_full_matrix']):.6f} | "
                f"{float(row['mean_gap_to_true']):.6e} | "
                f"{int(row['invalid_count_vs_true'])} |"
            )
        lines.append("")
        lines.append(
            "`p_exact` is intentionally stopped once vertex enumeration exceeds the configured exact-size limit; the other methods continue to larger N."
        )
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `p_exact`: exact minimum of the original paper lower function L(P; P^k) on the supplied P hyperrectangle, using vertex enumeration. It is the small-N P-space reference.")
    lines.append("- `common_o_n`: exact on the enlarged common-scalar box, but potentially conservative relative to the original componentwise box.")
    lines.append("- `q_box`: transformed-Q critical-vertex method; cheap but potentially conservative because the selected Q-box vertex need not be reachable from one P vertex.")
    lines.append("- `rough`: termwise closed-form bound on 1-L; cheap and conservative.")
    lines.append("- `u_upper`: direct critical P-vertex for the upper function U; no gradient-magnitude bound is used.")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("After selecting a Stage-2 formulation from these results, integrate it with the Stage-1 and Stage-3 hyperrectangle propagation to build a separate box-CCP controller. Then compare full CCP solve time, iteration count, tracking performance and closed-loop behavior against the existing simplex `ExactConicCCP` baseline.")
    lines.append("")
    if not plots_generated:
        lines.append("Plots were not generated because matplotlib was unavailable. Install it with `python -m pip install matplotlib` and rerun.")
        lines.append("")

    (output_dir / "MARK_STAGE2_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full Stage-2 benchmark: (1) run the existing simplex CCP and benchmark "
            "Stage-2 bounds on its real P boxes, and (2) run an N-scaling study on "
            "synthetic common-width boxes calibrated from those real boxes."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/small_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2_full_benchmark"))
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--solver", type=str, default="CLARABEL")
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument("--maximum-simplex-vertices", type=int, default=200)
    parser.add_argument("--maximum-exact-box-vertices", type=int, default=1_048_576)
    parser.add_argument("--real-fast-repeats", type=int, default=200)
    parser.add_argument("--real-exact-repeats", type=int, default=3)
    parser.add_argument("--scaling-fast-repeats", type=int, default=5)
    parser.add_argument("--scaling-exact-repeats", type=int, default=1)
    parser.add_argument("--scaling-n", type=_parse_n_values, default=_parse_n_values("4,6,8,10,12,14,16,24,32,48,64"))
    parser.add_argument("--scaling-exact-max-n", type=int, default=14)
    parser.add_argument("--synthetic-trials", type=int, default=3)
    parser.add_argument("--synthetic-width-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--validity-tol", type=float, default=1e-9)
    parser.add_argument("--reference", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Short smoke test before the full run.")
    args = parser.parse_args()

    if args.quick:
        args.max_iterations = min(args.max_iterations, 5)
        args.real_fast_repeats = min(args.real_fast_repeats, 30)
        args.real_exact_repeats = 1
        args.scaling_fast_repeats = 2
        args.scaling_exact_repeats = 1
        args.synthetic_trials = 1
        args.scaling_n = [n for n in args.scaling_n if n <= 16]
        args.scaling_exact_max_n = min(args.scaling_exact_max_n, 12)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "benchmark_config.json").open("w") as handle:
        serializable = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        json.dump(serializable, handle, indent=2)

    print("Running real CCP-box benchmark...")
    snapshots, real_entries, real_runtimes, geometry, sanity, probe_result = _real_benchmark(args, output_dir)
    real_summary = _summarise_real(real_entries, real_runtimes)

    _write_csv(output_dir / "real_stage2_entries.csv", real_entries)
    _write_csv(output_dir / "real_stage2_runtime.csv", real_runtimes)
    _write_csv(output_dir / "real_stage2_summary.csv", real_summary)
    _write_csv(output_dir / "real_p_box_geometry.csv", geometry)
    _write_csv(output_dir / "real_probe_sanity.csv", sanity)

    model, scalers = load_checkpoint(args.checkpoint)
    params = extract_parameters(model, scalers)

    print("Running synthetic N-scaling benchmark...")
    scaling_rows, scaling_summary, calibration = _scaling_benchmark(
        args, output_dir, snapshots, params.config.d_key
    )
    _write_csv(output_dir / "scaling_trials.csv", scaling_rows)
    _write_csv(output_dir / "scaling_summary.csv", scaling_summary)

    plots_generated = False
    if not args.no_plots:
        plots_generated = _make_plots(
            output_dir,
            real_entries,
            real_summary,
            geometry,
            sanity,
            scaling_summary,
        )
    _write_report(
        output_dir,
        args,
        real_summary,
        geometry,
        sanity,
        scaling_summary,
        plots_generated,
    )

    print("\nBaseline exact-CCP probe")
    print("  iterations:", len(snapshots))
    print("  first optimized input:", probe_result.future_u[0])
    print("  final objective:", probe_result.objective_history[-1])
    print("  final step:", probe_result.step_history[-1])
    print(
        "  all P(U^{k+1}) inside optimized P boxes:",
        all(s.p_optimized_inside_box for s in snapshots),
    )
    print(
        "  all P(U^{k+1}) inside optimized Stage-1 simplexes:",
        all(s.p_optimized_inside_simplex for s in snapshots),
    )
    print(
        "  max optimized-P box violation:",
        max(s.p_optimized_box_max_violation for s in snapshots),
    )

    print("\nReal-box Stage-2 summary")
    header = (
        f"{'method':<14} {'mean gap':>12} {'worst gap':>12} "
        f"{'invalid':>9} {'median ms':>12} {'R width':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in real_summary:
        print(
            f"{str(row['method']):<14} "
            f"{float(row['mean_gap_to_true']):>12.4e} "
            f"{float(row['worst_gap_to_true']):>12.4e} "
            f"{int(row['invalid_count_vs_true']):>9d} "
            f"{float(row['median_runtime_ms_full_matrix']):>12.4f} "
            f"{float(row['mean_r_box_width']):>12.4e}"
        )

    print(f"\nSaved full benchmark to: {output_dir}")
    print("  MARK_STAGE2_SUMMARY.md  <- easiest file to show/read")
    print("  real_stage2_summary.csv")
    print("  real_p_box_geometry.csv")
    print("  real_probe_sanity.csv")
    print("  scaling_summary.csv")
    print("  scaling_trials.csv")
    print("  real_p_box_history.npz   <- raw data; no need to open manually")
    if plots_generated:
        print("  real_conservatism_mean_gap.png")
        print("  real_conservatism_boxplot.png")
        print("  real_runtime.png")
        print("  real_common_box_expansion.png")
        print("  real_probe_sanity.png")
        print("  scaling_runtime.png")
        print("  scaling_conservatism.png")
        print("  scaling_r_box_width.png")


if __name__ == "__main__":
    main()
