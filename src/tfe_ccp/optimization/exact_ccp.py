from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..dc_bounds import forward_from_parameters, softmax_rows
from ..export import TFEParameters
from .common import MPCBounds, delta_offset, difference_matrix


@dataclass(frozen=True)
class ExactCCPConfig:
    tracking_weight: float = 1.0
    control_weight: float = 1.0
    max_iterations: int = 5
    tolerance: float = 1e-4
    solver: str = "CLARABEL"
    verbose: bool = False
    maximum_vertices: int = 200


@dataclass(frozen=True)
class ExactCCPResult:
    future_u: np.ndarray
    objective_history: list[float]
    step_history: list[float]
    status_history: list[str]


@dataclass(frozen=True)
class ExactSubproblemStats:
    solver_internal_time_seconds: float
    solver_num_iters: int
    num_scalar_variables: int
    num_scalar_eq_constraints: int
    num_scalar_leq_constraints: int


def _require_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "The exact paper formulation requires CVXPY and an exponential-cone solver. "
            "Install with: pip install cvxpy clarabel"
        ) from exc
    return cp


def _simplex_vertices(
    cp: Any, alpha: Any, beta: Any, maximum_vertices: int
) -> tuple[list[Any], Any]:
    alpha_flat = cp.reshape(alpha, (int(np.prod(alpha.shape)),), order="C")
    dimension = alpha_flat.shape[0]
    if dimension + 1 > maximum_vertices:
        raise ValueError(
            f"Simplex has {dimension + 1} vertices, exceeding "
            f"maximum_vertices={maximum_vertices}. Use smaller w/p/d_key for "
            "the exact reference implementation."
        )
    gamma = beta + cp.sum(alpha_flat)
    vertices = [-alpha_flat]
    for index in range(dimension):
        basis = np.zeros(dimension)
        basis[index] = 1.0
        vertices.append(-alpha_flat + gamma * basis)
    return vertices, gamma


def _stage1_bounds_cvx(
    cp: Any,
    q: Any,
    k: Any,
    q0: np.ndarray,
    k0: np.ndarray,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Corrected equations (11)-(12) for unscaled P = Q K.T."""

    length = q.shape[0]
    lower: list[list[Any]] = [[None for _ in range(length)] for _ in range(length)]
    upper: list[list[Any]] = [[None for _ in range(length)] for _ in range(length)]
    for i in range(length):
        for j in range(length):
            plus = q[i, :] + k[j, :]
            minus = q[i, :] - k[j, :]
            plus0 = q0[i] + k0[j]
            minus0 = q0[i] - k0[j]
            p1_linear = 0.25 * float(plus0 @ plus0) + 0.5 * plus0 @ (plus - plus0)
            p2 = 0.25 * cp.sum_squares(minus)
            p1 = 0.25 * cp.sum_squares(plus)
            p2_linear = 0.25 * float(minus0 @ minus0) + 0.5 * minus0 @ (minus - minus0)
            lower[i][j] = p1_linear - p2
            upper[i][j] = p1 - p2_linear
    return lower, upper


def _stage2_bounds_cvx(
    cp: Any,
    p: Any,
    p0: np.ndarray,
    d_key: int,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Corrected equations (14)-(15) for R = softmax(P/sqrt(d_key))."""

    scale = math.sqrt(d_key)
    p_scaled = p / scale
    p0_scaled = np.asarray(p0, dtype=np.float64) / scale
    length = p.shape[0]
    r0 = softmax_rows(p0_scaled)
    lower: list[list[Any]] = [[None for _ in range(length)] for _ in range(length)]
    upper: list[list[Any]] = [[None for _ in range(length)] for _ in range(length)]
    for row in range(length):
        maximum = float(np.max(p0_scaled[row]))
        lse0 = maximum + math.log(float(np.exp(p0_scaled[row] - maximum).sum()))
        tangent_lse = lse0 + r0[row] @ (p_scaled[row, :] - p0_scaled[row])
        exp_terms = [
            cp.exp(p_scaled[row, column] - tangent_lse)
            for column in range(length)
        ]
        for column in range(length):
            upper[row][column] = exp_terms[column]
            lower[row][column] = 1.0 - cp.sum(
                cp.hstack([exp_terms[j] for j in range(length) if j != column])
            )
    return lower, upper


def _stage3_bounds_cvx(
    cp: Any,
    r: Any,
    v: Any,
    r0: np.ndarray,
    v0: np.ndarray,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Corrected entrywise equations (17)-(18) for S = R V."""

    length, d_key = v.shape
    lower: list[list[Any]] = [[None for _ in range(d_key)] for _ in range(length)]
    upper: list[list[Any]] = [[None for _ in range(d_key)] for _ in range(length)]
    for i in range(length):
        for a in range(d_key):
            x, y = r[i, :], v[:, a]
            x0, y0 = r0[i], v0[:, a]
            plus, minus = x + y, x - y
            plus0, minus0 = x0 + y0, x0 - y0
            p1_linear = 0.25 * float(plus0 @ plus0) + 0.5 * plus0 @ (plus - plus0)
            p2 = 0.25 * cp.sum_squares(minus)
            p1 = 0.25 * cp.sum_squares(plus)
            p2_linear = 0.25 * float(minus0 @ minus0) + 0.5 * minus0 @ (minus - minus0)
            lower[i][a] = p1_linear - p2
            upper[i][a] = p1 - p2_linear
    return lower, upper


class ExactConicCCP:
    """Small-scale CVXPY reference for corrected Algorithm 1.

    The implementation enumerates simplex vertices exactly. Its purpose is
    validation and comparison, not large-horizon real-time control. Input and
    input-rate constraints are included as a practical extension.
    """

    def __init__(
        self,
        parameters: TFEParameters,
        bounds: MPCBounds,
        config: ExactCCPConfig = ExactCCPConfig(),
    ) -> None:
        self.parameters = parameters
        self.bounds = bounds
        self.config = config
        self.last_subproblem_stats: ExactSubproblemStats | None = None

    def _one_iteration(
        self,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        linearization_u: np.ndarray,
    ) -> tuple[np.ndarray, float, str]:
        cp = _require_cvxpy()
        prm = self.parameters
        cfg = prm.config
        current = forward_from_parameters(prm, past_u, past_y, linearization_u)
        length, d_key = cfg.sequence_length, cfg.d_key

        u = cp.Variable((cfg.horizon, cfg.n_inputs), name="U")
        alpha1 = cp.Variable((length, length), name="alpha1")
        beta1 = cp.Variable(name="beta1")
        alpha2 = cp.Variable((length, length), name="alpha2")
        beta2 = cp.Variable(name="beta2")
        alpha3 = cp.Variable((length, d_key), name="alpha3")
        beta3 = cp.Variable(name="beta3")
        worst_tracking = cp.Variable(nonneg=True, name="worst_tracking")

        past_u_n = (np.asarray(past_u) - prm.u_mean) / prm.u_scale
        past_y_n = (np.asarray(past_y) - prm.y_mean) / prm.y_scale
        past_xy_n = np.concatenate((past_u_n, past_y_n), axis=1)
        z_past = past_xy_n @ prm.w_em1.T + prm.b_em1
        u_n = cp.multiply(u - prm.u_mean, 1.0 / prm.u_scale)
        z_future = (
            u_n @ prm.w_em2.T
            + np.ones((cfg.horizon, 1)) @ prm.b_em2.reshape(1, -1)
        )
        z1 = cp.vstack((z_past, z_future))
        q = z1 @ prm.w_q.T + np.ones((length, 1)) @ prm.b_q.reshape(1, -1)
        k = z1 @ prm.w_k.T + np.ones((length, 1)) @ prm.b_k.reshape(1, -1)
        v = z1 @ prm.w_v.T + np.ones((length, 1)) @ prm.b_v.reshape(1, -1)

        constraints: list[Any] = []
        lower_p, upper_p = _stage1_bounds_cvx(cp, q, k, current.q, current.k)
        for i in range(length):
            for j in range(length):
                constraints.append(-alpha1[i, j] <= lower_p[i][j])
        constraints.append(
            cp.sum(
                cp.hstack(
                    [upper_p[i][j] for i in range(length) for j in range(length)]
                )
            )
            <= beta1
        )
        p_vertices, gamma1 = _simplex_vertices(
            cp, alpha1, beta1, self.config.maximum_vertices
        )
        constraints.append(gamma1 >= 0.0)

        for p_vertex in p_vertices:
            p_matrix = cp.reshape(p_vertex, (length, length), order="C")
            lower_r, upper_r = _stage2_bounds_cvx(
                cp, p_matrix, current.p, d_key
            )
            for i in range(length):
                for j in range(length):
                    constraints.append(-alpha2[i, j] <= lower_r[i][j])
            constraints.append(
                cp.sum(
                    cp.hstack(
                        [upper_r[i][j] for i in range(length) for j in range(length)]
                    )
                )
                <= beta2
            )
        r_vertices, gamma2 = _simplex_vertices(
            cp, alpha2, beta2, self.config.maximum_vertices
        )
        constraints.append(gamma2 >= 0.0)

        for r_vertex in r_vertices:
            r_matrix = cp.reshape(r_vertex, (length, length), order="C")
            lower_s, upper_s = _stage3_bounds_cvx(
                cp, r_matrix, v, current.r, current.v
            )
            for i in range(length):
                for a in range(d_key):
                    constraints.append(-alpha3[i, a] <= lower_s[i][a])
            constraints.append(
                cp.sum(
                    cp.hstack(
                        [upper_s[i][a] for i in range(length) for a in range(d_key)]
                    )
                )
                <= beta3
            )
        s_vertices, gamma3 = _simplex_vertices(
            cp, alpha3, beta3, self.config.maximum_vertices
        )
        constraints.append(gamma3 >= 0.0)

        reference = np.asarray(y_reference, dtype=np.float64).reshape(
            cfg.horizon, cfg.n_outputs
        )
        for s_vertex in s_vertices:
            s_matrix = cp.reshape(s_vertex, (length, d_key), order="C")
            h = cp.reshape(
                (z1 + s_matrix) @ prm.w_l1,
                (length,),
                order="C",
            ) + prm.b_l1
            y_n_flat = prm.w_l2 @ h + prm.b_l2
            y_n = cp.reshape(
                y_n_flat, (cfg.horizon, cfg.n_outputs), order="C"
            )
            y_physical = cp.multiply(y_n, prm.y_scale) + prm.y_mean
            constraints.append(
                self.config.tracking_weight
                * cp.sum_squares(y_physical - reference)
                <= worst_tracking
            )

        u_flat = cp.reshape(u, (cfg.horizon * cfg.n_inputs,), order="C")
        d = difference_matrix(cfg.horizon, cfg.n_inputs)
        offset = delta_offset(previous_u, cfg.horizon)
        delta = d @ u_flat - offset
        constraints.extend(
            [
                u >= np.asarray(self.bounds.u_min),
                u <= np.asarray(self.bounds.u_max),
                delta >= np.tile(np.asarray(self.bounds.du_min), cfg.horizon),
                delta <= np.tile(np.asarray(self.bounds.du_max), cfg.horizon),
            ]
        )
        objective = cp.Minimize(
            worst_tracking + self.config.control_weight * cp.sum_squares(delta)
        )
        problem = cp.Problem(objective, constraints)
        size_metrics = problem.size_metrics
        problem.solve(
            solver=self.config.solver,
            verbose=self.config.verbose,
            warm_start=True,
        )
        solver_stats = problem.solver_stats
        solver_internal_time = getattr(solver_stats, "solve_time", None)
        solver_num_iters = getattr(solver_stats, "num_iters", None)
        self.last_subproblem_stats = ExactSubproblemStats(
            solver_internal_time_seconds=(
                float(solver_internal_time) if solver_internal_time is not None else float("nan")
            ),
            solver_num_iters=int(solver_num_iters) if solver_num_iters is not None else -1,
            num_scalar_variables=int(size_metrics.num_scalar_variables),
            num_scalar_eq_constraints=int(size_metrics.num_scalar_eq_constr),
            num_scalar_leq_constraints=int(size_metrics.num_scalar_leq_constr),
        )
        if u.value is None:
            raise RuntimeError(
                f"Exact conic CCP subproblem failed with status {problem.status}"
            )
        return np.asarray(u.value), float(problem.value), str(problem.status)

    def solve(
        self,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        initial_u: np.ndarray,
    ) -> ExactCCPResult:
        u = np.asarray(initial_u, dtype=np.float64).reshape(
            self.parameters.config.horizon, self.parameters.config.n_inputs
        )
        objectives: list[float] = []
        steps: list[float] = []
        statuses: list[str] = []
        for _ in range(self.config.max_iterations):
            u_next, objective, status = self._one_iteration(
                past_u, past_y, y_reference, previous_u, u
            )
            step = float(np.linalg.norm(u_next - u))
            objectives.append(objective)
            steps.append(step)
            statuses.append(status)
            u = u_next
            if step <= self.config.tolerance:
                break
        return ExactCCPResult(u, objectives, steps, statuses)
