from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

import numpy as np

from tfe_ccp.export import extract_parameters, load_checkpoint
from tfe_ccp.optimization import MPCBounds
from tfe_ccp.optimization.box_ccp import BoxCCPConfig, BoxConicCCP
from tfe_ccp.systems import CoupledTanks, piecewise_constant_controls


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_initial_history(
    plant: CoupledTanks,
    *,
    past_window: int,
    history_length: int,
    control_low: float,
    control_high: float,
    hold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one deterministic burn-in trajectory and leave plant at its final state.

    This is an explicit benchmark initialization condition. It is not training data
    and is not inferred from the checkpoint.
    """
    if history_length < past_window:
        raise ValueError(
            f"--history-length={history_length} must be >= past_window={past_window}"
        )

    controls = piecewise_constant_controls(
        length=history_length,
        n_inputs=plant.n_inputs,
        low=control_low,
        high=control_high,
        hold=hold,
        seed=seed,
    )

    plant.reset()
    outputs = np.empty((history_length, plant.n_outputs), dtype=np.float64)
    for i, control in enumerate(controls):
        outputs[i] = plant.step(control)

    return controls[-past_window:].copy(), outputs[-past_window:].copy()



def _true_open_loop_rollout(
    plant: CoupledTanks,
    future_u: np.ndarray,
) -> np.ndarray:
    """Diagnostic only: roll out a deep copy of the current physical plant.

    The real closed-loop plant is not modified. The returned trajectory therefore
    gives the true nonlinear-plant response to the same optimized future input
    sequence used by the TFE prediction.
    """
    diagnostic_plant = copy.deepcopy(plant)
    future_u = np.asarray(future_u, dtype=np.float64)
    ys = []
    for u in future_u:
        y = diagnostic_plant.step(np.asarray(u, dtype=np.float64))
        ys.append(np.asarray(y, dtype=np.float64).reshape(-1))
    return np.asarray(ys, dtype=np.float64)

def _shift_warm_start(future_u: np.ndarray) -> np.ndarray:
    """Shift the previous MPC solution by one sample and repeat its last action."""
    future_u = np.asarray(future_u, dtype=np.float64)
    if future_u.shape[0] == 1:
        return future_u.copy()
    return np.vstack((future_u[1:], future_u[-1:]))


def _plot_results(
    outdir: Path,
    trajectory_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV files were saved, plots skipped.")
        return

    t = np.asarray([r["step"] for r in trajectory_rows], dtype=int)
    y = np.asarray([r["measured_output"] for r in trajectory_rows], dtype=float)
    ref = np.asarray([r["reference"] for r in trajectory_rows], dtype=float)
    u = np.asarray([r["applied_input"] for r in trajectory_rows], dtype=float)

    fig, ax = plt.subplots()
    ax.plot(t, y, marker="o", label="measured output")
    ax.plot(t, ref, linestyle="--", label="reference")
    ax.set_xlabel("Closed-loop step")
    ax.set_ylabel("Tank-2 level")
    ax.set_title("Closed-loop tracking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "closed_loop_output.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.step(t, u, where="post")
    ax.set_xlabel("Closed-loop step")
    ax.set_ylabel("Applied input")
    ax.set_title("Closed-loop control input")
    fig.tight_layout()
    fig.savefig(outdir / "closed_loop_input.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(
        [r["step"] for r in controller_rows],
        [r["ccp_iterations"] for r in controller_rows],
        marker="o",
    )
    ax.set_xlabel("Closed-loop step")
    ax.set_ylabel("CCP iterations")
    ax.set_title("CCP iterations per MPC update")
    fig.tight_layout()
    fig.savefig(outdir / "closed_loop_ccp_iterations.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(
        [r["step"] for r in controller_rows],
        [1000.0 * r["controller_wall_time_seconds"] for r in controller_rows],
        marker="o",
    )
    ax.set_xlabel("Closed-loop step")
    ax.set_ylabel("Controller wall time [ms]")
    ax.set_title("Controller time per MPC update")
    fig.tight_layout()
    fig.savefig(outdir / "closed_loop_controller_time.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3: closed-loop Rough box-CCP control of the nonlinear "
            "CoupledTanks plant using a trained TFE model."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/scaling_L4.pt"),
        help="Trained TFE checkpoint. Default: artifacts/scaling_L4.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment3_closed_loop"),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--tracking-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        type=str.upper,
        default="CLARABEL",
        choices=["CLARABEL", "SCS"],
        help="Convex solver used for each CCP subproblem. Default: CLARABEL",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Show the underlying CVXPY/solver log (separate from --verbose-diagnostics).",
    )

    # Explicit benchmark/controller assumptions.
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=10.0)
    parser.add_argument("--du-min", type=float, default=-2.0)
    parser.add_argument("--du-max", type=float, default=2.0)
    parser.add_argument("--reference-step", type=float, default=1.0)
    parser.add_argument("--reference-value", type=float, default=None)

    # Deterministic burn-in used to construct a physically consistent history.
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--history-seed", type=int, default=1)
    parser.add_argument("--history-control-low", type=float, default=0.0)
    parser.add_argument("--history-control-high", type=float, default=10.0)
    parser.add_argument("--history-hold", type=int, default=8)

    parser.add_argument("--plant-seed", type=int, default=1)
    parser.add_argument("--process-noise-std", type=float, default=0.0)
    parser.add_argument("--verbose-diagnostics", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, scalers = load_checkpoint(args.checkpoint)
    params = extract_parameters(model, scalers)
    cfg = params.config

    print(
        f"Checkpoint: {args.checkpoint}\n"
        f"Model: L={cfg.sequence_length}, w={cfg.past_window}, "
        f"H={cfg.horizon}, d_key={cfg.d_key}, "
        f"n_inputs={cfg.n_inputs}, n_outputs={cfg.n_outputs}\n"
        f"Solver: {args.solver}"
    )

    if cfg.n_inputs != 1 or cfg.n_outputs != 1:
        raise ValueError(
            "This first closed-loop script is intentionally written for the "
            "single-input/single-output CoupledTanks checkpoint."
        )

    bounds = MPCBounds.constant(
        cfg.n_inputs,
        args.u_min,
        args.u_max,
        args.du_min,
        args.du_max,
    )

    solver = BoxConicCCP(
        params,
        bounds,
        BoxCCPConfig(
            tracking_weight=args.tracking_weight,
            control_weight=args.control_weight,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            solver=args.solver,
            verbose=args.solver_verbose,
            stage2_method="rough",
            print_diagnostics=args.verbose_diagnostics,
        ),
    )


    plant = CoupledTanks(
        process_noise_std=args.process_noise_std,
        seed=args.plant_seed,
    )
    past_u, past_y = _build_initial_history(
        plant,
        past_window=cfg.past_window,
        history_length=args.history_length,
        control_low=args.history_control_low,
        control_high=args.history_control_high,
        hold=args.history_hold,
        seed=args.history_seed,
    )

    previous_u = past_u[-1].copy()
    current_y = past_y[-1].copy()

    if args.reference_value is None:
        reference_value = current_y + args.reference_step
        reference_mode = "initial_output_plus_step"
    else:
        reference_value = np.full(
            cfg.n_outputs, args.reference_value, dtype=np.float64
        )
        reference_mode = "absolute_value"

    # First MPC call: repeat the previously applied plant input over the horizon.
    initial_u = np.tile(previous_u.reshape(1, -1), (cfg.horizon, 1))

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    metadata_rows = [
        {
            "checkpoint": str(args.checkpoint),
            "L": cfg.sequence_length,
            "past_window_w": cfg.past_window,
            "prediction_control_horizon_H": cfg.horizon,
            "d_key": cfg.d_key,
            "closed_loop_steps": args.steps,
            "u_min": args.u_min,
            "u_max": args.u_max,
            "du_min": args.du_min,
            "du_max": args.du_max,
            "reference_mode": reference_mode,
            "reference_step": (
                args.reference_step if args.reference_value is None else ""
            ),
            "reference_value": float(reference_value[0]),
            "history_source": (
                "deterministic CoupledTanks burn-in benchmark condition; "
                "not training data"
            ),
            "history_length": args.history_length,
            "history_seed": args.history_seed,
            "history_control_low": args.history_control_low,
            "history_control_high": args.history_control_high,
            "history_hold": args.history_hold,
            "initial_measured_output": float(current_y[0]),
            "initial_previous_input": float(previous_u[0]),
            "process_noise_std": args.process_noise_std,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "tracking_weight": args.tracking_weight,
            "control_weight": args.control_weight,
            "solver": args.solver,
            "solver_verbose": args.solver_verbose,
            "stage2_method": "rough",
            "warm_start": "shift previous U* by one and repeat final action",
        }
    ]
    _write_csv(outdir / "experiment_metadata.csv", metadata_rows)

    trajectory_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []

    for step in range(args.steps):

        reference = np.tile(
            reference_value.reshape(1, -1), (cfg.horizon, 1)
        )

        solve_start = time.perf_counter()
        try:
            result = solver.solve(
                past_u,
                past_y,
                reference,
                previous_u,
                initial_u,
            )
        except Exception as exc:
            controller_rows.append(
                {
                    "step": step,
                    "run_status": "error",
                    "error": repr(exc),
                }
            )
            _write_csv(outdir / "controller_steps.csv", controller_rows)
            _write_csv(outdir / "closed_loop_trajectory.csv", trajectory_rows)
            _write_csv(outdir / "ccp_iteration_history.csv", iteration_rows)
            _write_csv(outdir / "box_diagnostics.csv", diagnostic_rows)
            _write_csv(outdir / "rollout_comparison.csv", rollout_rows)
            print(f"Step {step}: controller failed: {exc!r}")
            break

        controller_wall = time.perf_counter() - solve_start
        future_u = np.asarray(result.future_u, dtype=np.float64).reshape(
            cfg.horizon, cfg.n_inputs
        )

        # Compare the TFE prediction against the true nonlinear plant from the
        # SAME current physical state, using the SAME optimized future input U*.
        # This is diagnostic only: _true_open_loop_rollout uses a deep copy and
        # therefore does not advance or modify the real closed-loop plant.
        predicted_rollout = np.asarray(result.predicted_y, dtype=np.float64).reshape(
            cfg.horizon, cfg.n_outputs
        )
        true_rollout = _true_open_loop_rollout(plant, future_u).reshape(
            cfg.horizon, cfg.n_outputs
        )
        rollout_error = predicted_rollout - true_rollout

        for k in range(cfg.horizon):
            rollout_rows.append(
                {
                    "closed_loop_step": step,
                    "horizon_step": k + 1,
                    "optimized_input": float(future_u[k, 0]),
                    "tfe_predicted_output": float(predicted_rollout[k, 0]),
                    "true_plant_output": float(true_rollout[k, 0]),
                    "prediction_error": float(rollout_error[k, 0]),
                    "reference": float(reference_value[0]),
                }
            )

        print(
            f"  U*       = "
            + np.array2string(future_u[:, 0], precision=5, separator=", ")
        )
        print(
            f"  TFE yhat = "
            + np.array2string(predicted_rollout[:, 0], precision=5, separator=", ")
        )
        print(
            f"  True y   = "
            + np.array2string(true_rollout[:, 0], precision=5, separator=", ")
        )
        print(
            f"  Error    = "
            + np.array2string(rollout_error[:, 0], precision=5, separator=", ")
        )

        # Receding-horizon MPC: apply only the first optimized action.
        u_apply = future_u[0].copy()

        # Nonlinear plant update. CoupledTanks.step returns the measured tank-2 level.
        y_next = np.asarray(plant.step(u_apply), dtype=np.float64).reshape(
            cfg.n_outputs
        )

        final_diag = result.diagnostics[-1]
        total_core = float(np.sum(result.core_iteration_wall_time_history))

        trajectory_rows.append(
            {
                "step": step,
                "measured_output": float(y_next[0]),
                "reference": float(reference_value[0]),
                "tracking_error": float(y_next[0] - reference_value[0]),
                "applied_input": float(u_apply[0]),
                "predicted_first_output": float(predicted_rollout[0, 0]),
                "true_rollout_first_output": float(true_rollout[0, 0]),
                "first_output_prediction_error": float(rollout_error[0, 0]),
                "previous_input": float(previous_u[0]),
            }
        )

        controller_rows.append(
            {
                "step": step,
                "run_status": "completed",
                "ccp_iterations": len(result.diagnostics),
                "controller_wall_time_seconds": controller_wall,
                "total_ccp_core_wall_time_seconds": total_core,
                "median_subproblem_wall_time_seconds": float(
                    np.median(result.solve_time_history)
                ),
                "final_step_norm": float(result.step_history[-1]),
                "final_status": result.status_history[-1],
                "surrogate_objective": float(result.objective_history[-1]),
                "max_p_containment_violation": max(
                    d.p_containment_violation for d in result.diagnostics
                ),
                "max_r_containment_violation": max(
                    d.r_containment_violation for d in result.diagnostics
                ),
                "max_s_containment_violation": max(
                    d.s_containment_violation for d in result.diagnostics
                ),
                "max_p_ordering_violation": max(
                    d.p_ordering_violation for d in result.diagnostics
                ),
                "max_r_ordering_violation": max(
                    d.r_ordering_violation for d in result.diagnostics
                ),
                "max_s_ordering_violation": max(
                    d.s_ordering_violation for d in result.diagnostics
                ),
                "num_scalar_variables": final_diag.num_scalar_variables,
                "num_scalar_eq_constraints": final_diag.num_scalar_eq_constraints,
                "num_scalar_leq_constraints": final_diag.num_scalar_leq_constraints,
            }
        )

        for diag in result.diagnostics:
            row = asdict(diag)
            row["closed_loop_step"] = step
            diagnostic_rows.append(row)
            iteration_rows.append(
                {
                    "closed_loop_step": step,
                    "ccp_iteration": diag.iteration,
                    "surrogate_objective": diag.surrogate_objective,
                    "tracking_upper_bound": diag.tracking_upper_bound,
                    "actual_objective": diag.actual_objective,
                    "actual_tracking_cost": diag.actual_tracking_cost,
                    "control_cost": diag.control_cost,
                    "step_norm": diag.step_norm,
                    "subproblem_wall_time_seconds": diag.solve_time_seconds,
                    "core_iteration_wall_time_seconds": (
                        diag.core_iteration_wall_time_seconds
                    ),
                    "solver_internal_time_seconds": (
                        diag.solver_internal_time_seconds
                    ),
                    "solver_num_iters": diag.solver_num_iters,
                    "status": diag.status,
                }
            )

        print(
            f"step={step:02d} "
            f"y={y_next[0]:.5f} ref={reference_value[0]:.5f} "
            f"u={u_apply[0]:.5f} "
            f"CCP={len(result.diagnostics)} "
            f"core={1000.0 * total_core:.1f} ms "
            f"status={result.status_history[-1]}"
        )

        # Update measured input/output history for the next TFE call.
        if cfg.past_window > 1:
            past_u = np.vstack((past_u[1:], u_apply.reshape(1, -1)))
            past_y = np.vstack((past_y[1:], y_next.reshape(1, -1)))
        else:
            past_u = u_apply.reshape(1, -1)
            past_y = y_next.reshape(1, -1)

        previous_u = u_apply.copy()

        # Warm-start next MPC optimization with shifted previous solution.
        initial_u = _shift_warm_start(future_u)

        # Save incrementally in case a later closed-loop step fails.
        _write_csv(outdir / "closed_loop_trajectory.csv", trajectory_rows)
        _write_csv(outdir / "controller_steps.csv", controller_rows)
        _write_csv(outdir / "ccp_iteration_history.csv", iteration_rows)
        _write_csv(outdir / "box_diagnostics.csv", diagnostic_rows)
        _write_csv(outdir / "rollout_comparison.csv", rollout_rows)

    completed_rows = [
        r for r in controller_rows if r.get("run_status") == "completed"
    ]

    summary_rows: list[dict[str, Any]] = []
    if trajectory_rows and completed_rows:
        errors = np.asarray(
            [r["tracking_error"] for r in trajectory_rows], dtype=np.float64
        )
        applied = np.asarray(
            [r["applied_input"] for r in trajectory_rows], dtype=np.float64
        )
        previous_inputs = np.asarray(
            [r["previous_input"] for r in trajectory_rows], dtype=np.float64
        )
        delta_u = applied - previous_inputs

        summary_rows.append(
            {
                "completed_closed_loop_steps": len(trajectory_rows),
                "requested_closed_loop_steps": args.steps,
                "tracking_rmse": float(np.sqrt(np.mean(errors**2))),
                "tracking_mae": float(np.mean(np.abs(errors))),
                "mean_abs_delta_u": float(np.mean(np.abs(delta_u))),
                "mean_ccp_iterations": float(
                    np.mean([r["ccp_iterations"] for r in completed_rows])
                ),
                "median_ccp_iterations": float(
                    np.median([r["ccp_iterations"] for r in completed_rows])
                ),
                "mean_controller_wall_time_seconds": float(
                    np.mean(
                        [
                            r["controller_wall_time_seconds"]
                            for r in completed_rows
                        ]
                    )
                ),
                "median_controller_wall_time_seconds": float(
                    np.median(
                        [
                            r["controller_wall_time_seconds"]
                            for r in completed_rows
                        ]
                    )
                ),
                "mean_total_ccp_core_wall_time_seconds": float(
                    np.mean(
                        [
                            r["total_ccp_core_wall_time_seconds"]
                            for r in completed_rows
                        ]
                    )
                ),
                "max_p_containment_violation": float(
                    np.max(
                        [
                            r["max_p_containment_violation"]
                            for r in completed_rows
                        ]
                    )
                ),
                "max_r_containment_violation": float(
                    np.max(
                        [
                            r["max_r_containment_violation"]
                            for r in completed_rows
                        ]
                    )
                ),
                "max_s_containment_violation": float(
                    np.max(
                        [
                            r["max_s_containment_violation"]
                            for r in completed_rows
                        ]
                    )
                ),
                "solver": args.solver,
                "solver_statuses": ";".join(
                    sorted({str(r["final_status"]) for r in completed_rows})
                ),
            }
        )
        _write_csv(outdir / "closed_loop_summary.csv", summary_rows)
        _plot_results(outdir, trajectory_rows, completed_rows)

        s = summary_rows[0]
        print("\nClosed-loop summary")
        print(
            f"  completed steps: {s['completed_closed_loop_steps']}/"
            f"{s['requested_closed_loop_steps']}"
        )
        print(f"  tracking RMSE: {s['tracking_rmse']:.6g}")
        print(f"  tracking MAE:  {s['tracking_mae']:.6g}")
        print(f"  mean CCP iterations: {s['mean_ccp_iterations']:.3f}")
        print(
            "  median controller wall time: "
            f"{1000.0 * s['median_controller_wall_time_seconds']:.1f} ms"
        )
        print(
            "  max containment violations (P/R/S): "
            f"{s['max_p_containment_violation']:.3e} / "
            f"{s['max_r_containment_violation']:.3e} / "
            f"{s['max_s_containment_violation']:.3e}"
        )
    else:
        print("\nNo completed closed-loop steps; summary/plots not generated.")

    print(f"\nResults: {outdir}")
    print("Key files:")
    print("  experiment_metadata.csv")
    print("  closed_loop_trajectory.csv")
    print("  controller_steps.csv")
    print("  ccp_iteration_history.csv")
    print("  box_diagnostics.csv")
    print("  rollout_comparison.csv")
    print("  closed_loop_summary.csv")
    print("  closed_loop_output.png")
    print("  closed_loop_input.png")
    print("  closed_loop_ccp_iterations.png")
    print("  closed_loop_controller_time.png")


if __name__ == "__main__":
    main()
