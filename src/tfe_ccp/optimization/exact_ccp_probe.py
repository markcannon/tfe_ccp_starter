from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dc_bounds import forward_from_parameters
from ..export import TFEParameters
from .common import MPCBounds, delta_offset, difference_matrix
from .exact_ccp import (
    ExactCCPConfig,
    _require_cvxpy,
    _simplex_vertices,
    _stage1_bounds_cvx,
    _stage2_bounds_cvx,
    _stage3_bounds_cvx,
)


@dataclass(frozen=True)
class PBoxSnapshot:
    iteration: int
    p0: np.ndarray
    p_optimized: np.ndarray
    p_lower: np.ndarray
    p_upper: np.ndarray
    alpha1: np.ndarray
    beta1: float
    gamma1: float
    objective: float
    status: str
    p0_inside_box: bool
    p_optimized_inside_box: bool
    p_optimized_inside_simplex: bool
    p0_box_max_violation: float
    p_optimized_box_max_violation: float
    p_optimized_simplex_lower_max_violation: float
    p_optimized_simplex_sum_violation: float

    @property
    def common_component_width(self) -> float:
        """Width of each coordinate in the simplex bounding box."""

        return self.gamma1

    @property
    def common_scalar_enclosing_width(self) -> float:
        """Width after replacing the componentwise box by one scalar interval."""

        return float(np.max(self.p_upper) - np.min(self.p_lower))

    @property
    def common_scalar_expansion_ratio(self) -> float:
        """How much the common scalar interval expands the common-width box."""

        if self.gamma1 <= 1e-14:
            return 1.0
        return self.common_scalar_enclosing_width / self.gamma1


@dataclass(frozen=True)
class ExactCCPProbeResult:
    future_u: np.ndarray
    objective_history: list[float]
    step_history: list[float]
    status_history: list[str]
    p_box_history: list[PBoxSnapshot]


class ExactConicCCPProbe:
    """Instrumented copy of the exact simplex CCP used only for diagnostics.

    The baseline ``ExactConicCCP`` is intentionally left untouched. This probe
    solves the same convex subproblems, then records the axis-aligned P box
    that encloses the optimized Stage-1 simplex at every CCP iteration.

    For D(alpha1, beta1), with
        gamma1 = beta1 + sum(alpha1),
    the simplex vertices are
        -alpha1
    and
        -alpha1 + gamma1 e_j.
    Hence its exact axis-aligned bounding box is
        P_lower = -alpha1,
        P_upper = -alpha1 + gamma1.
    All entries have the same width gamma1, while their centres can differ.
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

    def _one_iteration(
        self,
        iteration: int,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        linearization_u: np.ndarray,
    ) -> tuple[np.ndarray, float, str, PBoxSnapshot]:
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
            lower_r, upper_r = _stage2_bounds_cvx(cp, p_matrix, current.p, d_key)
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
        problem.solve(
            solver=self.config.solver,
            verbose=self.config.verbose,
            warm_start=True,
        )
        if u.value is None or alpha1.value is None or beta1.value is None:
            raise RuntimeError(
                f"Exact conic CCP probe failed with status {problem.status}"
            )

        alpha_value = np.asarray(alpha1.value, dtype=np.float64)
        beta_value = float(beta1.value)
        gamma_value_raw = beta_value + float(alpha_value.sum())
        # A tiny negative value can occur from solver tolerance despite gamma>=0.
        gamma_value = max(gamma_value_raw, 0.0)
        p_lower = -alpha_value
        p_upper = p_lower + gamma_value
        p0 = np.asarray(current.p, dtype=np.float64)
        u_value = np.asarray(u.value, dtype=np.float64)
        optimized = forward_from_parameters(prm, past_u, past_y, u_value)
        p_optimized = np.asarray(optimized.p, dtype=np.float64)

        # p0 is the CCP linearisation point. It need not lie in the newly
        # optimised Stage-1 simplex/box. By contrast P(U^{k+1}) should satisfy
        # the optimised Stage-1 simplex (up to solver tolerance), hence also
        # its axis-aligned bounding box.
        tolerance = 1e-7

        def box_max_violation(point: np.ndarray) -> float:
            lower_violation = np.maximum(p_lower - point, 0.0)
            upper_violation = np.maximum(point - p_upper, 0.0)
            return float(max(np.max(lower_violation), np.max(upper_violation)))

        p0_box_violation = box_max_violation(p0)
        p_optimized_box_violation = box_max_violation(p_optimized)

        simplex_lower_violation = float(
            np.max(np.maximum(p_lower - p_optimized, 0.0))
        )
        simplex_sum_violation = float(
            max(float(np.sum(p_optimized)) - beta_value, 0.0)
        )
        p_optimized_simplex_violation = max(
            simplex_lower_violation, simplex_sum_violation
        )

        snapshot = PBoxSnapshot(
            iteration=iteration,
            p0=p0.copy(),
            p_optimized=p_optimized.copy(),
            p_lower=p_lower.copy(),
            p_upper=p_upper.copy(),
            alpha1=alpha_value.copy(),
            beta1=beta_value,
            gamma1=gamma_value,
            objective=float(problem.value),
            status=str(problem.status),
            p0_inside_box=p0_box_violation <= tolerance,
            p_optimized_inside_box=p_optimized_box_violation <= tolerance,
            p_optimized_inside_simplex=p_optimized_simplex_violation <= tolerance,
            p0_box_max_violation=p0_box_violation,
            p_optimized_box_max_violation=p_optimized_box_violation,
            p_optimized_simplex_lower_max_violation=simplex_lower_violation,
            p_optimized_simplex_sum_violation=simplex_sum_violation,
        )
        return u_value, float(problem.value), str(problem.status), snapshot

    def solve(
        self,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        initial_u: np.ndarray,
    ) -> ExactCCPProbeResult:
        u = np.asarray(initial_u, dtype=np.float64).reshape(
            self.parameters.config.horizon, self.parameters.config.n_inputs
        )
        objectives: list[float] = []
        steps: list[float] = []
        statuses: list[str] = []
        boxes: list[PBoxSnapshot] = []

        for iteration in range(self.config.max_iterations):
            u_next, objective, status, snapshot = self._one_iteration(
                iteration,
                past_u,
                past_y,
                y_reference,
                previous_u,
                u,
            )
            step = float(np.linalg.norm(u_next - u))
            objectives.append(objective)
            steps.append(step)
            statuses.append(status)
            boxes.append(snapshot)
            u = u_next
            if step <= self.config.tolerance:
                break

        return ExactCCPProbeResult(u, objectives, steps, statuses, boxes)
