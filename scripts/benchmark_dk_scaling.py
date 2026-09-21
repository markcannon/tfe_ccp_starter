from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from tfe_ccp.dc_bounds import forward_from_parameters
from tfe_ccp.export import extract_parameters, load_checkpoint
from tfe_ccp.optimization import ExactCCPConfig, ExactConicCCP, MPCBounds
from tfe_ccp.optimization.box_ccp import BoxCCPConfig, BoxConicCCP
from tfe_ccp.optimization.common import delta_offset, difference_matrix
from tfe_ccp.systems import CoupledTanks, piecewise_constant_controls, simulate


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


def _checkpoint_validation_mse(path: Path) -> tuple[float, float]:
    """Return (best, final) validation MSE stored in the training checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", {}) or {}
    validation = history.get("validation", []) or []
    if not validation:
        return float("nan"), float("nan")
    values = np.asarray(validation, dtype=np.float64)
    return float(np.min(values)), float(values[-1])


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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    u = np.asarray(initial_u, dtype=np.float64).copy()
    rows: list[dict[str, Any]] = []
    for iteration in range(solver.config.max_iterations):
        start = time.perf_counter()
        u_next, surrogate, status = solver._one_iteration(  # intentional benchmark instrumentation
            past_u, past_y, reference, previous_u, u
        )
        iteration_wall = time.perf_counter() - start
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
        stats = solver.last_subproblem_stats
        rows.append(
            {
                "iteration": iteration,
                "surrogate_objective": surrogate,
                "actual_tracking_cost": tracking,
                "control_cost": control,
                "actual_objective": actual,
                "step_norm": step,
                "subproblem_wall_time_seconds": iteration_wall,
                "solver_internal_time_seconds": (
                    stats.solver_internal_time_seconds if stats is not None else float("nan")
                ),
                "solver_num_iters": stats.solver_num_iters if stats is not None else -1,
                "num_scalar_variables": stats.num_scalar_variables if stats is not None else -1,
                "num_scalar_eq_constraints": stats.num_scalar_eq_constraints if stats is not None else -1,
                "num_scalar_leq_constraints": stats.num_scalar_leq_constraints if stats is not None else -1,
                "status": status,
            }
        )
        u = u_next
        if step <= solver.config.tolerance:
            break
    return u, rows


def _build_common_coupled_tanks_history(args: argparse.Namespace, max_past_window: int):
    """Create one deterministic physical operating history shared by every model.

    This history is a benchmark condition, not training data and not inferred from
    the checkpoints. All parameters are written to experiment_metadata.csv.
    """

    if args.history_length < max_past_window:
        raise ValueError(
            f"--history-length={args.history_length} is shorter than max past window "
            f"{max_past_window}. Increase --history-length."
        )
    plant = CoupledTanks(process_noise_std=0.0, seed=args.history_seed)
    controls = piecewise_constant_controls(
        length=args.history_length,
        n_inputs=plant.n_inputs,
        low=args.history_control_low,
        high=args.history_control_high,
        hold=args.history_hold,
        seed=args.history_seed,
    )
    trajectory = simulate(plant, controls)
    return trajectory


def _aggregate_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        if row.get("run_status") != "completed":
            continue
        key = (int(row["d_key"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    numeric_fields = [
        "best_validation_mse",
        "final_validation_mse",
        "ccp_iterations",
        "median_subproblem_wall_time_seconds",
        "median_solver_internal_time_seconds",
        "total_ccp_core_wall_time_seconds",
        "observed_controller_call_wall_time_seconds",
        "actual_objective",
        "tracking_cost",
        "control_cost",
        "first_input",
        "final_step_norm",
        "max_p_containment_violation",
        "max_r_containment_violation",
        "max_s_containment_violation",
        "num_scalar_variables",
        "num_scalar_eq_constraints",
        "num_scalar_leq_constraints",
        "median_solver_num_iters",
    ]
    for (d_key, method), rows in sorted(groups.items()):
        first = rows[0]
        out: dict[str, Any] = {
            "L": first["L"],
            "w": first["w"],
            "H": first["H"],
            "d_key": d_key,
            "method": method,
            "completed_repeats": len(rows),
            "final_statuses": ";".join(sorted({str(r["final_status"]) for r in rows})),
        }
        for field in numeric_fields:
            values = []
            for row in rows:
                try:
                    value = float(row[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            out[field] = float(np.median(values)) if values else float("nan")
        summary.append(out)
    return summary


def _plot_scaling(
    outdir: Path,
    summary_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> None:
    """Plot Experiment 2: scaling with attention key/query dimension d_key."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV files were saved, plots skipped.")
        return

    completed = [
        row for row in summary_rows
        if np.isfinite(float(row["total_ccp_core_wall_time_seconds"]))
    ]
    if not completed:
        print("No completed controller runs; scaling plots skipped.")
        return

    all_dkeys = sorted({int(row["d_key"]) for row in run_rows if "d_key" in row})
    methods = sorted({row["method"] for row in completed})

    def _completed_runs(method: str, d_key: int) -> list[dict[str, Any]]:
        return [
            row for row in run_rows
            if row.get("method") == method
            and int(row.get("d_key", -1)) == d_key
            and row.get("run_status") == "completed"
        ]

    def _median_with_range(method: str, d_key: int, field: str, scale: float = 1.0):
        values = []
        for row in _completed_runs(method, d_key):
            try:
                value = scale * float(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if not values:
            return None
        median = float(np.median(values))
        return median, median - float(np.min(values)), float(np.max(values)) - median

    def _plot_method_with_ranges(ax, method: str, field: str, scale: float = 1.0):
        rows = sorted([r for r in completed if r["method"] == method], key=lambda r: int(r["d_key"]))
        xs = [int(r["d_key"]) for r in rows]
        ys = [scale * float(r[field]) for r in rows]
        marker = "o" if method == "rough_box" else "s"
        linestyle = "-" if method == "rough_box" else "--"
        ax.plot(xs, ys, marker=marker, linestyle=linestyle, label=method)
        err_x=[]; err_y=[]; err_low=[]; err_high=[]
        for x in xs:
            stats=_median_with_range(method,x,field,scale)
            if stats is None: continue
            median,low,high=stats
            if low>0.0 or high>0.0:
                err_x.append(x); err_y.append(median); err_low.append(low); err_high.append(high)
        if err_x:
            ax.errorbar(err_x,err_y,yerr=[err_low,err_high],fmt="none",capsize=4)

    def _annotate_exact_failures(ax):
        failed=sorted({int(r["d_key"]) for r in run_rows if r.get("method")=="exact_simplex" and r.get("run_status")=="error"})
        if not failed: return
        _,ymax=ax.get_ylim(); marker_y=ymax/1.8
        for d_key in failed:
            ax.scatter([d_key],[marker_y],marker="x",s=90,zorder=5)
            ax.annotate("Exact failed",xy=(d_key,marker_y),xytext=(0,-16),textcoords="offset points",ha="center",va="top",fontsize=9)

    fig,ax=plt.subplots()
    for method in methods: _plot_method_with_ranges(ax,method,"total_ccp_core_wall_time_seconds",1000.0)
    ax.set_xlabel("Attention dimension d_key"); ax.set_ylabel("Total CCP core wall time [ms]"); ax.set_yscale("log")
    ax.set_title("Controller scaling with attention dimension"); ax.set_xticks(all_dkeys); _annotate_exact_failures(ax); ax.legend(); fig.tight_layout()
    fig.savefig(outdir/"dk_scaling_total_ccp_time.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots()
    for method in methods: _plot_method_with_ranges(ax,method,"median_subproblem_wall_time_seconds",1000.0)
    ax.set_xlabel("Attention dimension d_key"); ax.set_ylabel("Median convex-subproblem wall time [ms]"); ax.set_yscale("log")
    ax.set_title("Convex-subproblem scaling with attention dimension"); ax.set_xticks(all_dkeys); _annotate_exact_failures(ax); ax.legend(); fig.tight_layout()
    fig.savefig(outdir/"dk_scaling_subproblem_time.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots()
    for method in methods: _plot_method_with_ranges(ax,method,"actual_objective")
    ax.set_xlabel("Attention dimension d_key"); ax.set_ylabel("Actual TFE objective"); ax.set_title("Solution quality across attention dimensions")
    ax.set_xticks(all_dkeys); ax.legend(); fig.tight_layout(); fig.savefig(outdir/"dk_scaling_actual_objective.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots()
    for method in methods: _plot_method_with_ranges(ax,method,"first_input")
    ax.set_xlabel("Attention dimension d_key"); ax.set_ylabel("First optimized input u0"); ax.set_title("First MPC action across attention dimensions")
    ax.set_xticks(all_dkeys); ax.legend(); fig.tight_layout(); fig.savefig(outdir/"dk_scaling_first_input.png",dpi=180); plt.close(fig)

    rough_rows=sorted([r for r in completed if r["method"]=="rough_box"],key=lambda r:int(r["d_key"]))
    if rough_rows:
        fig,ax=plt.subplots()
        ax.plot([int(r["d_key"]) for r in rough_rows],[float(r["best_validation_mse"]) for r in rough_rows],marker="o")
        ax.set_xlabel("Attention dimension d_key"); ax.set_ylabel("Best validation MSE"); ax.set_yscale("log")
        ax.set_title("Prediction quality across attention dimensions"); ax.set_xticks(all_dkeys); fig.tight_layout()
        fig.savefig(outdir/"dk_scaling_validation_mse.png",dpi=180); plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: full-CCP d_key scaling on trained CoupledTanks TFE checkpoints."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dkeys", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--fixed-l", type=int, default=8)
    parser.add_argument("--baseline-dk2-checkpoint", default="scaling_L8.pt")
    parser.add_argument("--checkpoint-pattern", default="scaling_L8_dk{d_key}.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiment2_dk_scaling"))
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--exact-vertex-limit", type=int, default=200)
    parser.add_argument("--exact-max-dkey", type=int, default=4, help="Largest d_key for which Exact simplex is attempted.")
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=10.0)
    parser.add_argument("--du-min", type=float, default=-2.0)
    parser.add_argument("--du-max", type=float, default=2.0)
    parser.add_argument("--reference-step", type=float, default=1.0)
    parser.add_argument("--reference-value", type=float, default=None)
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--history-seed", type=int, default=1)
    parser.add_argument("--history-control-low", type=float, default=0.0)
    parser.add_argument("--history-control-high", type=float, default=10.0)
    parser.add_argument("--history-hold", type=int, default=8)
    parser.add_argument("--verbose-diagnostics", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    checkpoint_records: list[dict[str, Any]] = []
    for requested_dkey in args.dkeys:
        if requested_dkey == 2:
            checkpoint = args.checkpoint_dir / args.baseline_dk2_checkpoint
        else:
            checkpoint = args.checkpoint_dir / args.checkpoint_pattern.format(d_key=requested_dkey)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model, scalers = load_checkpoint(checkpoint)
        params = extract_parameters(model, scalers)
        cfg = params.config
        if cfg.sequence_length != args.fixed_l:
            raise ValueError(f"{checkpoint}: expected fixed L={args.fixed_l}, but checkpoint has L={cfg.sequence_length}.")
        if cfg.d_key != requested_dkey:
            raise ValueError(f"{checkpoint}: requested d_key={requested_dkey}, but checkpoint has d_key={cfg.d_key}.")
        best_val, final_val = _checkpoint_validation_mse(checkpoint)
        checkpoint_records.append({
            "checkpoint": checkpoint, "params": params, "L": cfg.sequence_length, "w": cfg.past_window,
            "H": cfg.horizon, "d_key": cfg.d_key, "n_inputs": cfg.n_inputs, "n_outputs": cfg.n_outputs,
            "best_validation_mse": best_val, "final_validation_mse": final_val,
        })


    fixed_signatures = {
        (r["L"], r["w"], r["H"], r["n_inputs"], r["n_outputs"]) for r in checkpoint_records
    }
    if len(fixed_signatures) != 1:
        raise ValueError(
            "Experiment 2 expects L, w, H, n_inputs and n_outputs to be fixed across checkpoints. "
            f"Found: {sorted(fixed_signatures)}"
        )

    max_w = max(int(r["w"]) for r in checkpoint_records)
    trajectory = _build_common_coupled_tanks_history(args, max_w)
    current_y = np.asarray(trajectory.y[-1], dtype=np.float64)
    previous_u_common = np.asarray(trajectory.u[-1], dtype=np.float64)
    if args.reference_value is None:
        reference_value = current_y + args.reference_step
        reference_mode = "current_output_plus_step"
    else:
        reference_value = np.full_like(current_y, args.reference_value, dtype=np.float64)
        reference_mode = "absolute_value"

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = [
        {
            "operating_history_source": "deterministic CoupledTanks simulation (benchmark condition, not training data)",
            "history_length": args.history_length,
            "history_seed": args.history_seed,
            "history_control_low": args.history_control_low,
            "history_control_high": args.history_control_high,
            "history_hold": args.history_hold,
            "common_current_output": repr(current_y.tolist()),
            "common_previous_input": repr(previous_u_common.tolist()),
            "reference_mode": reference_mode,
            "reference_step": args.reference_step if args.reference_value is None else "",
            "reference_value": repr(reference_value.tolist()),
            "u_min": args.u_min,
            "u_max": args.u_max,
            "du_min": args.du_min,
            "du_max": args.du_max,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "repeats": args.repeats,
            "exact_vertex_limit": args.exact_vertex_limit,
            "fixed_L": args.fixed_l,
            "tested_dkeys": repr(list(args.dkeys)),
            "exact_max_dkey": args.exact_max_dkey,
        }
    ]
    _write_csv(outdir / "experiment_metadata.csv", metadata)

    run_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    box_diagnostic_rows: list[dict[str, Any]] = []

    for record in checkpoint_records:
        checkpoint = Path(record["checkpoint"])
        params = record["params"]
        cfg = params.config
        L = cfg.sequence_length
        w = cfg.past_window
        H = cfg.horizon

        past_u = np.asarray(trajectory.u[-w:], dtype=np.float64)
        past_y = np.asarray(trajectory.y[-w:], dtype=np.float64)
        previous_u = previous_u_common.copy()
        reference = np.tile(reference_value.reshape(1, -1), (H, 1))
        initial_u = np.tile(previous_u.reshape(1, -1), (H, 1))
        bounds = MPCBounds.constant(
            cfg.n_inputs, args.u_min, args.u_max, args.du_min, args.du_max
        )

        print(
            f"\n=== L={L} (w={w}, H={H}, d_key={cfg.d_key}) | "
            f"checkpoint={checkpoint.name} | best val MSE={record['best_validation_mse']:.6g} ==="
        )

        for repeat in range(args.repeats):
            print(f"Rough repeat {repeat + 1}/{args.repeats} ...")
            try:
                box_solver = BoxConicCCP(
                    params,
                    bounds,
                    BoxCCPConfig(
                        max_iterations=args.max_iterations,
                        tolerance=args.tolerance,
                        stage2_method="rough",
                        print_diagnostics=args.verbose_diagnostics,
                    ),
                )
                outer_start = time.perf_counter()
                box_result = box_solver.solve(
                    past_u, past_y, reference, previous_u, initial_u
                )
                observed_wall = time.perf_counter() - outer_start
                box_forward, tracking, control, actual = _actual_cost(
                    params,
                    box_solver.config,
                    past_u,
                    past_y,
                    box_result.future_u,
                    reference,
                    previous_u,
                )
                final_diag = box_result.diagnostics[-1]
                total_core = float(np.sum(box_result.core_iteration_wall_time_history))
                run_row = {
                    "checkpoint": str(checkpoint),
                    "L": L,
                    "w": w,
                    "H": H,
                    "d_key": cfg.d_key,
                    "method": "rough_box",
                    "repeat": repeat,
                    "run_status": "completed",
                    "skip_reason": "",
                    "best_validation_mse": record["best_validation_mse"],
                    "final_validation_mse": record["final_validation_mse"],
                    "ccp_iterations": len(box_result.diagnostics),
                    "median_subproblem_wall_time_seconds": float(np.median(box_result.solve_time_history)),
                    "median_solver_internal_time_seconds": float(
                        np.nanmedian([d.solver_internal_time_seconds for d in box_result.diagnostics])
                    ),
                    "total_ccp_core_wall_time_seconds": total_core,
                    "observed_controller_call_wall_time_seconds": observed_wall,
                    "actual_objective": actual,
                    "tracking_cost": tracking,
                    "control_cost": control,
                    "first_input": float(box_result.future_u.reshape(H, cfg.n_inputs)[0, 0]),
                    "final_step_norm": box_result.step_history[-1],
                    "final_status": box_result.status_history[-1],
                    "max_p_containment_violation": max(d.p_containment_violation for d in box_result.diagnostics),
                    "max_r_containment_violation": max(d.r_containment_violation for d in box_result.diagnostics),
                    "max_s_containment_violation": max(d.s_containment_violation for d in box_result.diagnostics),
                    "max_p_ordering_violation": max(d.p_ordering_violation for d in box_result.diagnostics),
                    "max_r_ordering_violation": max(d.r_ordering_violation for d in box_result.diagnostics),
                    "max_s_ordering_violation": max(d.s_ordering_violation for d in box_result.diagnostics),
                    "num_scalar_variables": final_diag.num_scalar_variables,
                    "num_scalar_eq_constraints": final_diag.num_scalar_eq_constraints,
                    "num_scalar_leq_constraints": final_diag.num_scalar_leq_constraints,
                    "median_solver_num_iters": float(np.median([d.solver_num_iters for d in box_result.diagnostics])),
                    "largest_simplex_vertices": max(L**2 + 1, L * cfg.d_key + 1),
                }
                run_rows.append(run_row)

                for diag in box_result.diagnostics:
                    row = asdict(diag)
                    row.update(
                        {
                            "checkpoint": str(checkpoint),
                            "L": L,
                            "w": w,
                            "H": H,
                            "d_key": cfg.d_key,
                            "method": "rough_box",
                            "repeat": repeat,
                        }
                    )
                    box_diagnostic_rows.append(row)
                    iteration_rows.append(
                        {
                            "checkpoint": str(checkpoint),
                            "L": L,
                            "w": w,
                            "H": H,
                            "d_key": cfg.d_key,
                            "method": "rough_box",
                            "repeat": repeat,
                            "iteration": diag.iteration,
                            "surrogate_objective": diag.surrogate_objective,
                            "actual_objective": diag.actual_objective,
                            "actual_tracking_cost": diag.actual_tracking_cost,
                            "control_cost": diag.control_cost,
                            "step_norm": diag.step_norm,
                            "subproblem_wall_time_seconds": diag.solve_time_seconds,
                            "core_iteration_wall_time_seconds": diag.core_iteration_wall_time_seconds,
                            "solver_internal_time_seconds": diag.solver_internal_time_seconds,
                            "solver_num_iters": diag.solver_num_iters,
                            "status": diag.status,
                        }
                    )
                print(
                    f"  done: iters={len(box_result.diagnostics)}, "
                    f"core={1000*total_core:.1f} ms, "
                    f"median subproblem={1000*np.median(box_result.solve_time_history):.1f} ms, "
                    f"J={actual:.6g}, u0={run_row['first_input']:.5g}, status={run_row['final_status']}"
                )
            except Exception as exc:
                run_rows.append(
                    {
                        "checkpoint": str(checkpoint),
                        "L": L,
                        "w": w,
                        "H": H,
                        "d_key": cfg.d_key,
                        "method": "rough_box",
                        "repeat": repeat,
                        "run_status": "error",
                        "skip_reason": repr(exc),
                        "best_validation_mse": record["best_validation_mse"],
                        "final_validation_mse": record["final_validation_mse"],
                    }
                )
                print(f"  Rough failed: {exc!r}")

            if args.skip_exact:
                continue

            largest_simplex = max(L**2 + 1, L * cfg.d_key + 1)

            if cfg.d_key > args.exact_max_dkey:
                if repeat == 0:
                    run_rows.append({
                        "checkpoint": str(checkpoint), "L": L, "w": w, "H": H, "d_key": cfg.d_key,
                        "method": "exact_simplex", "repeat": "", "run_status": "skipped",
                        "skip_reason": f"d_key={cfg.d_key} exceeds --exact-max-dkey={args.exact_max_dkey}",
                        "largest_simplex_vertices": largest_simplex,
                        "best_validation_mse": record["best_validation_mse"],
                        "final_validation_mse": record["final_validation_mse"],
                    })
                continue

            if largest_simplex > args.exact_vertex_limit:
                if repeat == 0:
                    run_rows.append(
                        {
                            "checkpoint": str(checkpoint),
                            "L": L,
                            "w": w,
                            "H": H,
                            "d_key": cfg.d_key,
                            "method": "exact_simplex",
                            "repeat": "",
                            "run_status": "skipped",
                            "skip_reason": (
                                f"largest simplex has {largest_simplex} vertices, above "
                                f"limit {args.exact_vertex_limit}"
                            ),
                            "largest_simplex_vertices": largest_simplex,
                            "best_validation_mse": record["best_validation_mse"],
                            "final_validation_mse": record["final_validation_mse"],
                        }
                    )
                continue

            print(f"Exact repeat {repeat + 1}/{args.repeats} ...")
            try:
                exact_solver = ExactConicCCP(
                    params,
                    bounds,
                    ExactCCPConfig(
                        max_iterations=args.max_iterations,
                        tolerance=args.tolerance,
                        maximum_vertices=args.exact_vertex_limit,
                    ),
                )
                outer_start = time.perf_counter()
                exact_u, exact_rows = _run_exact_instrumented(
                    exact_solver,
                    past_u,
                    past_y,
                    reference,
                    previous_u,
                    initial_u,
                )
                observed_wall = time.perf_counter() - outer_start
                _, tracking, control, actual = _actual_cost(
                    params,
                    exact_solver.config,
                    past_u,
                    past_y,
                    exact_u,
                    reference,
                    previous_u,
                )
                last = exact_rows[-1]
                total_core = float(np.sum([r["subproblem_wall_time_seconds"] for r in exact_rows]))
                run_row = {
                    "checkpoint": str(checkpoint),
                    "L": L,
                    "w": w,
                    "H": H,
                    "d_key": cfg.d_key,
                    "method": "exact_simplex",
                    "repeat": repeat,
                    "run_status": "completed",
                    "skip_reason": "",
                    "best_validation_mse": record["best_validation_mse"],
                    "final_validation_mse": record["final_validation_mse"],
                    "ccp_iterations": len(exact_rows),
                    "median_subproblem_wall_time_seconds": float(
                        np.median([r["subproblem_wall_time_seconds"] for r in exact_rows])
                    ),
                    "median_solver_internal_time_seconds": float(
                        np.nanmedian([r["solver_internal_time_seconds"] for r in exact_rows])
                    ),
                    "total_ccp_core_wall_time_seconds": total_core,
                    "observed_controller_call_wall_time_seconds": observed_wall,
                    "actual_objective": actual,
                    "tracking_cost": tracking,
                    "control_cost": control,
                    "first_input": float(exact_u.reshape(H, cfg.n_inputs)[0, 0]),
                    "final_step_norm": last["step_norm"],
                    "final_status": last["status"],
                    "max_p_containment_violation": float("nan"),
                    "max_r_containment_violation": float("nan"),
                    "max_s_containment_violation": float("nan"),
                    "max_p_ordering_violation": float("nan"),
                    "max_r_ordering_violation": float("nan"),
                    "max_s_ordering_violation": float("nan"),
                    "num_scalar_variables": last["num_scalar_variables"],
                    "num_scalar_eq_constraints": last["num_scalar_eq_constraints"],
                    "num_scalar_leq_constraints": last["num_scalar_leq_constraints"],
                    "median_solver_num_iters": float(np.median([r["solver_num_iters"] for r in exact_rows])),
                    "largest_simplex_vertices": largest_simplex,
                }
                run_rows.append(run_row)
                for row in exact_rows:
                    row = dict(row)
                    row.update(
                        {
                            "checkpoint": str(checkpoint),
                            "L": L,
                            "w": w,
                            "H": H,
                            "d_key": cfg.d_key,
                            "method": "exact_simplex",
                            "repeat": repeat,
                        }
                    )
                    iteration_rows.append(row)
                print(
                    f"  done: iters={len(exact_rows)}, core={1000*total_core:.1f} ms, "
                    f"median subproblem={1000*run_row['median_subproblem_wall_time_seconds']:.1f} ms, "
                    f"J={actual:.6g}, u0={run_row['first_input']:.5g}, status={run_row['final_status']}"
                )
            except Exception as exc:
                run_rows.append(
                    {
                        "checkpoint": str(checkpoint),
                        "L": L,
                        "w": w,
                        "H": H,
                        "d_key": cfg.d_key,
                        "method": "exact_simplex",
                        "repeat": repeat,
                        "run_status": "error",
                        "skip_reason": repr(exc),
                        "largest_simplex_vertices": largest_simplex,
                        "best_validation_mse": record["best_validation_mse"],
                        "final_validation_mse": record["final_validation_mse"],
                    }
                )
                print(f"  Exact failed: {exc!r}")

        # Save incrementally so long exact runs do not risk losing earlier results.
        _write_csv(outdir / "scaling_runs.csv", run_rows)
        _write_csv(outdir / "iteration_history.csv", iteration_rows)
        _write_csv(outdir / "box_diagnostics.csv", box_diagnostic_rows)
        _write_csv(outdir / "scaling_summary.csv", _aggregate_rows(run_rows))

    summary_rows = _aggregate_rows(run_rows)
    _write_csv(outdir / "scaling_runs.csv", run_rows)
    _write_csv(outdir / "iteration_history.csv", iteration_rows)
    _write_csv(outdir / "box_diagnostics.csv", box_diagnostic_rows)
    _write_csv(outdir / "scaling_summary.csv", summary_rows)
    _plot_scaling(outdir, summary_rows, run_rows)

    print("\nExperiment 2 finished.")
    print(f"Results: {outdir}")
    print("Key files:")
    print("  experiment_metadata.csv")
    print("  scaling_runs.csv")
    print("  scaling_summary.csv")
    print("  iteration_history.csv")
    print("  box_diagnostics.csv")
    print("  dk_scaling_total_ccp_time.png")
    print("  dk_scaling_subproblem_time.png")
    print("  dk_scaling_actual_objective.png")
    print("  dk_scaling_first_input.png")
    print("  dk_scaling_validation_mse.png")


if __name__ == "__main__":
    main()
