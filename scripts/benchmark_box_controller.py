from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from tfe_ccp.dc_bounds import forward_from_parameters
from tfe_ccp.export import extract_parameters, load_checkpoint
from tfe_ccp.optimization import ExactCCPConfig, ExactConicCCP, MPCBounds
from tfe_ccp.optimization.box_ccp import BoxCCPConfig, BoxConicCCP
from tfe_ccp.optimization.common import delta_offset, difference_matrix


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _actual_cost(params, config, past_u, past_y, future_u, reference, previous_u):
    forward = forward_from_parameters(params, past_u, past_y, future_u)
    tracking = config.tracking_weight * float(np.sum((forward.y - reference) ** 2))
    dmat = difference_matrix(params.config.horizon, params.config.n_inputs)
    delta = dmat @ future_u.reshape(-1) - delta_offset(previous_u, params.config.horizon)
    control = config.control_weight * float(delta @ delta)
    return forward, tracking, control, tracking + control


def _run_exact_instrumented(
    solver: ExactConicCCP,
    past_u: np.ndarray,
    past_y: np.ndarray,
    reference: np.ndarray,
    previous_u: np.ndarray,
    initial_u: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    u = np.asarray(initial_u, dtype=np.float64).copy()
    rows: list[dict] = []
    for iteration in range(solver.config.max_iterations):
        start = time.perf_counter()
        u_next, surrogate, status = solver._one_iteration(  # intentional instrumentation of baseline
            past_u, past_y, reference, previous_u, u
        )
        elapsed = time.perf_counter() - start
        step = float(np.linalg.norm(u_next - u))
        _, tracking, control, actual = _actual_cost(
            solver.parameters,
            solver.config,
            past_u,
            past_y,
            u_next,
            reference,
            previous_u,
        )
        rows.append(
            {
                "controller": "exact_simplex",
                "iteration": iteration,
                "surrogate_objective": surrogate,
                "actual_tracking_cost": tracking,
                "control_cost": control,
                "actual_objective": actual,
                "step_norm": step,
                "solve_time_seconds": elapsed,
                "status": status,
            }
        )
        u = u_next
        if step <= solver.config.tolerance:
            break
    return u, rows


def _plot_results(outdir: Path, iteration_rows: list[dict], box_rows: list[dict], final_rows: list[dict], reference: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV files were saved, plots skipped.")
        return

    controllers = sorted({row["controller"] for row in iteration_rows})

    fig, ax = plt.subplots()
    for controller in controllers:
        rows = [r for r in iteration_rows if r["controller"] == controller]
        ax.plot([r["iteration"] for r in rows], [r["actual_objective"] for r in rows], marker="o", label=controller)
    ax.set_xlabel("CCP iteration")
    ax.set_ylabel("Actual model objective")
    ax.set_title("Controller comparison: actual model objective")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "actual_objective_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    for controller in controllers:
        rows = [r for r in iteration_rows if r["controller"] == controller]
        ax.plot([r["iteration"] for r in rows], [1000.0 * r["solve_time_seconds"] for r in rows], marker="o", label=controller)
    ax.set_xlabel("CCP iteration")
    ax.set_ylabel("Subproblem solve time [ms]")
    ax.set_yscale("log")
    ax.set_title("Controller comparison: solve time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "solve_time_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    for controller in controllers:
        rows = [r for r in iteration_rows if r["controller"] == controller]
        ax.plot([r["iteration"] for r in rows], [max(float(r["step_norm"]), 1e-16) for r in rows], marker="o", label=controller)
    ax.set_xlabel("CCP iteration")
    ax.set_ylabel("||U^(k+1)-U^k||")
    ax.set_yscale("log")
    ax.set_title("CCP step convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "step_history.png", dpi=180)
    plt.close(fig)

    if box_rows:
        fig, ax = plt.subplots()
        box_controllers = sorted({r["controller"] for r in box_rows})
        for controller in box_controllers:
            rows = [r for r in box_rows if r["controller"] == controller]
            it = [r["iteration"] for r in rows]
            ax.plot(it, [max(r["p_mean_width"], 1e-16) for r in rows], marker="o", label=f"{controller}: P")
            ax.plot(it, [max(r["r_mean_width"], 1e-16) for r in rows], marker="o", label=f"{controller}: R")
            ax.plot(it, [max(r["s_mean_width"], 1e-16) for r in rows], marker="o", label=f"{controller}: S")
        ax.set_xlabel("CCP iteration")
        ax.set_ylabel("Mean box width")
        ax.set_yscale("log")
        ax.set_title("Box propagation through Stages 1-3")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(outdir / "box_width_history.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots()
        for controller in box_controllers:
            rows = [r for r in box_rows if r["controller"] == controller]
            it = [r["iteration"] for r in rows]
            ax.plot(it, [max(r["rough_mean_gap_to_true_softmax_min"], 1e-16) for r in rows], marker="o", label=f"{controller}: Rough diagnostic")
            ax.plot(it, [max(r["qbox_mean_gap_to_true_softmax_min"], 1e-16) for r in rows], marker="o", label=f"{controller}: exact Q-box diagnostic")
        ax.set_xlabel("CCP iteration")
        ax.set_ylabel("Mean Stage-2 lower-bound gap")
        ax.set_yscale("log")
        ax.set_title("Solved P-boxes: post-solve Stage-2 diagnostics")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(outdir / "rough_vs_qbox_on_controller_boxes.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots()
    horizon = reference.shape[0]
    t = np.arange(horizon)
    for row in final_rows:
        y = np.asarray(row["predicted_y_array"])
        ax.plot(t, y[:, 0], marker="o", label=row["controller"])
    ax.plot(t, reference[:, 0], linestyle="--", label="reference")
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Predicted output")
    ax.set_title("Final predicted output")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "final_prediction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    for row in final_rows:
        u = np.asarray(row["future_u_array"])
        ax.plot(np.arange(u.shape[0]), u[:, 0], marker="o", label=row["controller"])
    ax.set_xlabel("Control horizon step")
    ax.set_ylabel("Input")
    ax.set_title("Final optimized input sequence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "final_input_sequence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/box_controller_benchmark"))
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--u-min", type=float, default=-2.0)
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument("--du-min", type=float, default=-0.5)
    parser.add_argument("--du-max", type=float, default=0.5)
    parser.add_argument("--reference", type=float, default=1.0)
    args = parser.parse_args()

    model, scalers = load_checkpoint(args.checkpoint)
    params = extract_parameters(model, scalers)
    cfg = params.config
    bounds = MPCBounds.constant(cfg.n_inputs, args.u_min, args.u_max, args.du_min, args.du_max)
    past_u = np.zeros((cfg.past_window, cfg.n_inputs))
    past_y = np.zeros((cfg.past_window, cfg.n_outputs))
    reference = np.full((cfg.horizon, cfg.n_outputs), args.reference, dtype=np.float64)
    previous_u = np.zeros(cfg.n_inputs)
    initial = np.zeros((cfg.horizon, cfg.n_inputs))

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    iteration_rows: list[dict] = []
    final_rows_internal: list[dict] = []

    box_rows: list[dict] = []
    for method, controller_name in (("rough", "box_rough"), ("q_box_linearized", "box_q_linearized")):
        box_solver = BoxConicCCP(
            params,
            bounds,
            BoxCCPConfig(
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                stage2_method=method,
            ),
        )
        box_start = time.perf_counter()
        box_result = box_solver.solve(past_u, past_y, reference, previous_u, initial)
        box_total = time.perf_counter() - box_start
        method_box_rows = []
        for diag in box_result.diagnostics:
            row = asdict(diag)
            row["controller"] = controller_name
            row["stage2_method"] = method
            iteration_rows.append(
                {
                    "controller": controller_name,
                    "iteration": diag.iteration,
                    "surrogate_objective": diag.surrogate_objective,
                    "actual_tracking_cost": diag.actual_tracking_cost,
                    "control_cost": diag.control_cost,
                    "actual_objective": diag.actual_objective,
                    "step_norm": diag.step_norm,
                    "solve_time_seconds": diag.solve_time_seconds,
                    "status": diag.status,
                }
            )
            method_box_rows.append(row)
            box_rows.append(row)
        box_forward, box_tracking, box_control, box_actual = _actual_cost(
            params, box_solver.config, past_u, past_y, box_result.future_u, reference, previous_u
        )
        final_rows_internal.append(
            {
                "controller": controller_name,
                "iterations": len(box_result.diagnostics),
                "total_wall_time_seconds": box_total,
                "median_subproblem_time_seconds": float(np.median(box_result.solve_time_history)),
                "final_surrogate_objective": box_result.objective_history[-1],
                "final_actual_tracking_cost": box_tracking,
                "final_control_cost": box_control,
                "final_actual_objective": box_actual,
                "first_input": float(box_result.future_u.reshape(cfg.horizon, cfg.n_inputs)[0, 0]),
                "max_p_containment_violation": max(d.p_containment_violation for d in box_result.diagnostics),
                "max_r_containment_violation": max(d.r_containment_violation for d in box_result.diagnostics),
                "max_s_containment_violation": max(d.s_containment_violation for d in box_result.diagnostics),
                "future_u_array": box_result.future_u,
                "predicted_y_array": box_forward.y,
            }
        )

    if not args.skip_exact:
        largest_simplex = max(cfg.sequence_length**2 + 1, cfg.sequence_length * cfg.d_key + 1)
        if largest_simplex <= 200:
            exact_solver = ExactConicCCP(
                params,
                bounds,
                ExactCCPConfig(max_iterations=args.max_iterations, tolerance=args.tolerance, maximum_vertices=200),
            )
            exact_start = time.perf_counter()
            exact_u, exact_rows = _run_exact_instrumented(
                exact_solver, past_u, past_y, reference, previous_u, initial
            )
            exact_total = time.perf_counter() - exact_start
            iteration_rows.extend(exact_rows)
            exact_forward, exact_tracking, exact_control, exact_actual = _actual_cost(
                params, exact_solver.config, past_u, past_y, exact_u, reference, previous_u
            )
            final_rows_internal.append(
                {
                    "controller": "exact_simplex",
                    "iterations": len(exact_rows),
                    "total_wall_time_seconds": exact_total,
                    "median_subproblem_time_seconds": float(np.median([r["solve_time_seconds"] for r in exact_rows])),
                    "final_surrogate_objective": exact_rows[-1]["surrogate_objective"],
                    "final_actual_tracking_cost": exact_tracking,
                    "final_control_cost": exact_control,
                    "final_actual_objective": exact_actual,
                    "first_input": float(exact_u[0, 0]),
                    "max_p_containment_violation": "",
                    "max_r_containment_violation": "",
                    "max_s_containment_violation": "",
                    "future_u_array": exact_u,
                    "predicted_y_array": exact_forward.y,
                }
            )
        else:
            print(f"Skipping exact simplex controller: largest simplex has {largest_simplex} vertices (>200).")

    summary_rows = []
    for row in final_rows_internal:
        summary_rows.append({k: v for k, v in row.items() if not k.endswith("_array")})
    _write_csv(outdir / "controller_summary.csv", summary_rows)
    _write_csv(outdir / "iteration_history.csv", iteration_rows)
    _write_csv(outdir / "box_diagnostics.csv", box_rows)

    trajectory_rows = []
    for row in final_rows_internal:
        u = np.asarray(row["future_u_array"])
        y = np.asarray(row["predicted_y_array"])
        for t in range(cfg.horizon):
            trajectory_rows.append(
                {
                    "controller": row["controller"],
                    "step": t,
                    "u": float(u[t, 0]) if cfg.n_inputs == 1 else repr(u[t].tolist()),
                    "predicted_y": float(y[t, 0]) if cfg.n_outputs == 1 else repr(y[t].tolist()),
                    "reference": float(reference[t, 0]) if cfg.n_outputs == 1 else repr(reference[t].tolist()),
                }
            )
    _write_csv(outdir / "final_trajectories.csv", trajectory_rows)
    _plot_results(outdir, iteration_rows, box_rows, final_rows_internal, reference)

    print("\nController summary")
    print("controller       iters   total ms   median iter ms   actual obj   tracking   control   first u")
    print("-" * 100)
    for row in summary_rows:
        print(
            f"{row['controller']:<16} {row['iterations']:>5d} "
            f"{1000*row['total_wall_time_seconds']:>10.2f} "
            f"{1000*row['median_subproblem_time_seconds']:>16.2f} "
            f"{row['final_actual_objective']:>12.6g} "
            f"{row['final_actual_tracking_cost']:>10.6g} "
            f"{row['final_control_cost']:>9.6g} "
            f"{row['first_input']:>9.5f}"
        )
    print(f"\nSaved results to: {outdir}")
    print("Key files:")
    print("  controller_summary.csv")
    print("  iteration_history.csv")
    print("  box_diagnostics.csv")
    print("  actual_objective_history.png")
    print("  solve_time_history.png")
    print("  box_width_history.png")
    print("  rough_vs_qbox_on_controller_boxes.png")
    print("  final_prediction.png")


if __name__ == "__main__":
    main()
