from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import numpy as np

from ..dc_bounds import (
    ForwardIntermediates,
    forward_from_parameters,
    softmax_rows,
    stage2_q_box_lower,
    stage2_q_box_linearized_bounds,
    stage2_rough_box_bounds,
    stage2_true_box_extrema,
)
from ..export import TFEParameters
from .common import MPCBounds, delta_offset, difference_matrix


Stage2ControllerMethod = Literal["rough", "q_box_linearized"]


@dataclass(frozen=True)
class BoxCCPConfig:
    tracking_weight: float = 1.0
    control_weight: float = 1.0
    max_iterations: int = 20
    tolerance: float = 1e-4
    solver: str = "CLARABEL"
    verbose: bool = False
    stage2_method: Stage2ControllerMethod = "rough"
    print_diagnostics: bool = True


@dataclass(frozen=True)
class BoxIterationDiagnostics:
    iteration: int
    surrogate_objective: float
    tracking_upper_bound: float
    actual_tracking_cost: float
    control_cost: float
    actual_objective: float
    step_norm: float
    solve_time_seconds: float
    core_iteration_wall_time_seconds: float
    solver_internal_time_seconds: float
    solver_num_iters: int
    num_scalar_variables: int
    num_scalar_eq_constraints: int
    num_scalar_leq_constraints: int
    status: str
    p_mean_width: float
    p_max_width: float
    r_mean_width: float
    r_max_width: float
    s_mean_width: float
    s_max_width: float
    p_containment_violation: float
    r_containment_violation: float
    s_containment_violation: float
    p_ordering_violation: float
    r_ordering_violation: float
    s_ordering_violation: float
    s_expr_mean_width: float
    s_expr_max_width: float
    s_lower_slack_mean: float
    s_upper_slack_mean: float
    s_linearization_nearest_bound_mean: float
    s_linearization_nearest_bound_max: float
    s_linearization_lower_nearest_fraction: float
    s_linearization_upper_nearest_fraction: float
    y_var_mean_width: float
    y_expr_mean_width: float
    y_width_ratio: float
    tracking_upper_from_s_var: float
    tracking_upper_from_s_expr: float
    tracking_upper_difference: float
    active_endpoint_agreement_fraction: float
    active_var_lower_fraction: float
    active_expr_lower_fraction: float
    alpha_sweep_range: float
    componentwise_max_tracking_reduction: float
    componentwise_mean_tracking_reduction: float
    componentwise_argmax_row: int
    componentwise_argmax_col: int
    rough_qbox_mean_abs_difference: float
    rough_qbox_max_abs_difference: float
    rough_mean_gap_to_true_softmax_min: float
    qbox_mean_gap_to_true_softmax_min: float

    # Conditioning diagnostics.
    max_abs_q: float
    max_abs_k: float
    max_abs_p: float
    scaled_p_min: float
    scaled_p_max: float
    scaled_p_range: float
    p_domain_scaled_min: float
    p_domain_scaled_max: float
    p_domain_scaled_range: float
    p_min_positive_width: float
    p_median_width: float
    p_width_dynamic_range: float
    r_min_positive_width: float
    r_median_width: float
    r_width_dynamic_range: float
    s_min_positive_width: float
    s_median_width: float
    s_width_dynamic_range: float


@dataclass(frozen=True)
class BoxCCPResult:
    future_u: np.ndarray
    predicted_y: np.ndarray
    objective_history: list[float]
    step_history: list[float]
    status_history: list[str]
    solve_time_history: list[float]
    core_iteration_wall_time_history: list[float]
    diagnostics: list[BoxIterationDiagnostics]


@dataclass(frozen=True)
class _BoxSolution:
    future_u: np.ndarray
    objective: float
    tracking_upper: float
    control_cost: float
    status: str
    solve_time: float
    solver_internal_time: float
    solver_num_iters: int
    num_scalar_variables: int
    num_scalar_eq_constraints: int
    num_scalar_leq_constraints: int
    p_lower: np.ndarray
    p_upper: np.ndarray
    r_lower: np.ndarray
    r_upper: np.ndarray
    s_lower: np.ndarray
    s_upper: np.ndarray
    s_expr_lower: np.ndarray
    s_expr_upper: np.ndarray
    max_abs_q: float
    max_abs_k: float
    max_abs_p: float
    scaled_p_min: float
    scaled_p_max: float
    scaled_p_range: float
    p_domain_scaled_min: float
    p_domain_scaled_max: float
    p_domain_scaled_range: float


def _require_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "BoxConicCCP requires CVXPY and a conic solver. "
            "Install with: pip install cvxpy clarabel"
        ) from exc
    return cp


def _positive_negative(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    return np.maximum(arr, 0.0), np.minimum(arr, 0.0)


def _affine_interval(
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval for y = x @ matrix + bias over a box in x."""
    
    matrix = np.asarray(matrix, dtype=np.float64)
    pos, neg = _positive_negative(matrix)
    lo = np.asarray(x_lower, dtype=np.float64) @ pos + np.asarray(x_upper, dtype=np.float64) @ neg + bias
    hi = np.asarray(x_upper, dtype=np.float64) @ pos + np.asarray(x_lower, dtype=np.float64) @ neg + bias
    return lo, hi


def _future_projection_affine(
    prm: TFEParameters,
    projection_weight: np.ndarray,
    projection_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical future-u -> projected Q/K/V affine map.

    Returns A,b such that each future row is u_physical @ A + b.
    """

    a_normalized = prm.w_em2.T @ projection_weight.T
    a_physical = np.diag(1.0 / prm.u_scale) @ a_normalized
    b = (
        -(prm.u_mean / prm.u_scale) @ a_normalized
        + prm.b_em2 @ projection_weight.T
        + projection_bias
    )
    return a_physical, b


def _qkv_global_intervals(
    prm: TFEParameters,
    bounds: MPCBounds,
    current: ForwardIntermediates,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed Q/K/V domains induced by physical input bounds.

    Past rows are fixed by the measured history. Future rows are affine in U,
    so their interval is exact for the rectangular input bounds.
    """

    cfg = prm.config
    n, d = cfg.sequence_length, cfg.d_key
    w = cfg.past_window
    q_lo = np.empty((n, d), dtype=np.float64)
    q_hi = np.empty((n, d), dtype=np.float64)
    k_lo = np.empty((n, d), dtype=np.float64)
    k_hi = np.empty((n, d), dtype=np.float64)
    v_lo = np.empty((n, d), dtype=np.float64)
    v_hi = np.empty((n, d), dtype=np.float64)
    q_lo[:w] = q_hi[:w] = current.q[:w]
    k_lo[:w] = k_hi[:w] = current.k[:w]
    v_lo[:w] = v_hi[:w] = current.v[:w]

    u_lo = np.asarray(bounds.u_min, dtype=np.float64)
    u_hi = np.asarray(bounds.u_max, dtype=np.float64)
    for weight, bias, lo_arr, hi_arr in (
        (prm.w_q, prm.b_q, q_lo, q_hi),
        (prm.w_k, prm.b_k, k_lo, k_hi),
        (prm.w_v, prm.b_v, v_lo, v_hi),
    ):
        a, b = _future_projection_affine(prm, weight, bias)
        row_lo, row_hi = _affine_interval(u_lo, u_hi, a, b)
        lo_arr[w:] = row_lo
        hi_arr[w:] = row_hi
    return q_lo, q_hi, k_lo, k_hi, v_lo, v_hi


def _linear_min(coeff: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    pos, neg = _positive_negative(coeff)
    return float(pos @ lower + neg @ upper)


def _stage1_approximation_domain(
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    k_lower: np.ndarray,
    k_upper: np.ndarray,
    q0: np.ndarray,
    k0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Safe fixed domain for the Stage-1 DC lower/upper approximation values.

    These are bounds on the *DC approximation expressions*, not merely on the
    true QK^T product. This matters because P-box decision variables satisfy
    constraints against those approximation expressions.
    """

    n = q0.shape[0]
    lower = np.empty((n, n), dtype=np.float64)
    upper = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            plus_lo = q_lower[i] + k_lower[j]
            plus_hi = q_upper[i] + k_upper[j]
            minus_lo = q_lower[i] - k_upper[j]
            minus_hi = q_upper[i] - k_lower[j]
            plus0 = q0[i] + k0[j]
            minus0 = q0[i] - k0[j]

            tangent_plus_min = (
                0.5 * _linear_min(plus0, plus_lo, plus_hi)
                - 0.25 * float(plus0 @ plus0)
            )
            minus_abs = np.maximum(np.abs(minus_lo), np.abs(minus_hi))
            minus_square_max = 0.25 * float(minus_abs @ minus_abs)
            lower[i, j] = tangent_plus_min - minus_square_max

            plus_abs = np.maximum(np.abs(plus_lo), np.abs(plus_hi))
            plus_square_max = 0.25 * float(plus_abs @ plus_abs)
            tangent_minus_min = (
                0.5 * _linear_min(minus0, minus_lo, minus_hi)
                - 0.25 * float(minus0 @ minus0)
            )
            upper[i, j] = plus_square_max - tangent_minus_min
    return lower, upper


def _stage1_bounds_cvx(
    cp: Any,
    q: Any,
    k: Any,
    q0: np.ndarray,
    k0: np.ndarray,
) -> tuple[list[list[Any]], list[list[Any]]]:
    n = q.shape[0]
    lower: list[list[Any]] = [[None for _ in range(n)] for _ in range(n)]
    upper: list[list[Any]] = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            plus = q[i, :] + k[j, :]
            minus = q[i, :] - k[j, :]
            plus0 = q0[i] + k0[j]
            minus0 = q0[i] - k0[j]
            lower[i][j] = (
                0.25 * float(plus0 @ plus0)
                + 0.5 * plus0 @ (plus - plus0)
                - 0.25 * cp.sum_squares(minus)
            )
            upper[i][j] = (
                0.25 * cp.sum_squares(plus)
                - 0.25 * float(minus0 @ minus0)
                - 0.5 * minus0 @ (minus - minus0)
            )
    return lower, upper


def _stage2_rough_box_bounds_cvx(
    cp: Any,
    p_lower: Any,
    p_upper: Any,
    p0: np.ndarray,
    d_key: int,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Controller-compatible Stage-2 box bounds.

    The lower expression is the term-wise Rough bound from Notes 17 Aug.
    The upper expression evaluates U at its exact critical P-box vertex.
    Both give DCP-safe constraints when P-box endpoints are optimization
    variables.
    """

    scale = math.sqrt(d_key)
    lo = p_lower / scale
    hi = p_upper / scale
    x0 = np.asarray(p0, dtype=np.float64) / scale
    n = x0.shape[0]
    s0 = softmax_rows(x0)
    lower: list[list[Any]] = [[None for _ in range(n)] for _ in range(n)]
    upper: list[list[Any]] = [[None for _ in range(n)] for _ in range(n)]

    for row in range(n):
        s = s0[row]
        maximum = float(np.max(x0[row]))
        lse0 = maximum + math.log(float(np.exp(x0[row] - maximum).sum()))
        constant = float(s @ x0[row] - lse0)
        exp_terms: list[Any] = []
        for j in range(n):
            exponent = (
                constant
                + (1.0 - s[j]) * hi[row, j]
                - cp.sum(cp.multiply(s, lo[row, :]))
                + s[j] * lo[row, j]
            )
            exp_terms.append(cp.exp(exponent))
        for target in range(n):
            others = [exp_terms[j] for j in range(n) if j != target]
            lower[row][target] = 1.0 - cp.sum(cp.hstack(others))
            upper[row][target] = exp_terms[target]
    return lower, upper



def _stage2_qbox_linearized_bounds_cvx(
    cp: Any,
    p_lower: Any,
    p_upper: Any,
    p0: np.ndarray,
    d_key: int,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Convex-compatible Q-box variant using tangent lower-Q endpoints.

    The exact June-20 Q-box lower bound is not DCP-safe with optimization-
    variable P endpoints. Here q_lower = LSE(P_lower,-j) is replaced by the
    affine tangent of LSE at P^k. Since the tangent underestimates q_lower and
    enters the transformed exponent with a negative coefficient, this gives a
    safe (possibly more conservative) Stage-2 lower bound.
    """

    scale = math.sqrt(d_key)
    lo = p_lower / scale
    hi = p_upper / scale
    x0 = np.asarray(p0, dtype=np.float64) / scale
    n = x0.shape[0]
    if n < 2:
        raise ValueError("Q-box transform requires sequence length >= 2.")

    # Reuse the direct critical-P upper bound from the Rough construction.
    _, upper = _stage2_rough_box_bounds_cvx(cp, p_lower, p_upper, p0, d_key)
    lower: list[list[Any]] = [[None for _ in range(n)] for _ in range(n)]
    log_nm1 = math.log(n - 1.0)

    for row in range(n):
        q0 = np.empty(n, dtype=np.float64)
        q_upper: list[Any] = [None for _ in range(n)]
        q_lower_hat: list[Any] = [None for _ in range(n)]
        for j in range(n):
            indices = [m for m in range(n) if m != j]
            x0_sub = x0[row, indices]
            maximum = float(np.max(x0_sub))
            q0[j] = maximum + math.log(float(np.exp(x0_sub - maximum).sum()))
            q_upper[j] = cp.log_sum_exp(cp.hstack([hi[row, m] for m in indices]))
            grad = softmax_rows(x0_sub[None, :])[0]
            q_lower_hat[j] = q0[j] + grad @ (
                cp.hstack([lo[row, m] for m in indices]) - x0_sub
            )

        tq = softmax_rows(q0[None, :])[0]
        max_q0 = float(np.max(q0))
        lse_q0 = max_q0 + math.log(float(np.exp(q0 - max_q0).sum()))
        constant = float(tq @ q0 - lse_q0 + log_nm1)
        for target in range(n):
            exponent = constant + (1.0 - tq[target]) * q_upper[target]
            for j in range(n):
                if j != target:
                    exponent = exponent - tq[j] * q_lower_hat[j]
            lower[row][target] = 1.0 - cp.exp(exponent)
    return lower, upper

def _stage3_point_lower_cvx(cp: Any, r: Any, v: Any, r0: np.ndarray, v0: np.ndarray) -> Any:
    plus = r + v
    minus = r - v
    plus0 = r0 + v0
    return (
        0.25 * float(plus0 @ plus0)
        + 0.5 * plus0 @ (plus - plus0)
        - 0.25 * cp.sum_squares(minus)
    )


def _stage3_point_upper_cvx(cp: Any, r: Any, v: Any, r0: np.ndarray, v0: np.ndarray) -> Any:
    plus = r + v
    minus = r - v
    minus0 = r0 - v0
    return (
        0.25 * cp.sum_squares(plus)
        - 0.25 * float(minus0 @ minus0)
        - 0.5 * minus0 @ (minus - minus0)
    )


def _stage3_monotone_box_bounds_cvx(
    cp: Any,
    r_lower: Any,
    r_upper: Any,
    v: Any,
    r0: np.ndarray,
    v0: np.ndarray,
    r_domain_lower: np.ndarray,
    r_domain_upper: np.ndarray,
    v_domain_lower: np.ndarray,
    v_domain_upper: np.ndarray,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Stage-3 box bounds using constant Jacobian bounds over fixed domains.

    Notes 17 Aug constructs monotone DC components by shifting with Jacobian
    bounds. Here V is kept as the exact affine function of U, so the monotone
    shift is only required in the uncertain R coordinates. No artificial V-box
    is propagated through the optimization.
    """

    n, d_key = v0.shape
    lower: list[list[Any]] = [[None for _ in range(d_key)] for _ in range(n)]
    upper: list[list[Any]] = [[None for _ in range(d_key)] for _ in range(n)]

    for i in range(n):
        for a in range(d_key):
            x0 = r0[i]
            y0 = v0[:, a]
            y = v[:, a]
            x_u = r_upper[i, :]

            # sup d(lower_DC)/dR over the fixed R,V domain. Subtracting
            # gamma makes the concave component non-increasing in R.
            gamma = 0.5 * (
                x0 - r_domain_lower[i] + y0 + v_domain_upper[:, a]
            )
            gamma_pos, gamma_neg = _positive_negative(gamma)
            point_lower = _stage3_point_lower_cvx(cp, x_u, y, x0, y0)
            lower[i][a] = (
                point_lower
                - gamma @ (x_u - x0)
                + gamma_pos @ (r_lower[i, :] - x0)
                + gamma_neg @ (r_upper[i, :] - x0)
            )

            # inf d(upper_DC)/dR. Subtracting delta makes the convex
            # component non-decreasing in R.
            delta = 0.5 * (
                r_domain_lower[i] + v_domain_lower[:, a] - x0 + y0
            )
            delta_pos, delta_neg = _positive_negative(delta)
            point_upper = _stage3_point_upper_cvx(cp, x_u, y, x0, y0)
            upper[i][a] = (
                point_upper
                - delta @ (x_u - x0)
                + delta_pos @ (r_upper[i, :] - x0)
                + delta_neg @ (r_lower[i, :] - x0)
            )
    return lower, upper


def _output_interval_cvx(
    cp: Any,
    prm: TFEParameters,
    z1: Any,
    s_lower: Any,
    s_upper: Any,
) -> tuple[Any, Any]:
    """Axis-aligned output box induced by the S-box.

    This is a scalable outer approximation of the affine readout image. The
    tracking epigraph built from it is conservative because different output
    components can attain their interval endpoints at incompatible S corners.
    """

    w1_pos, w1_neg = _positive_negative(prm.w_l1)
    s_term_lower = s_lower @ w1_pos + s_upper @ w1_neg
    s_term_upper = s_upper @ w1_pos + s_lower @ w1_neg
    z_term = z1 @ prm.w_l1 + prm.b_l1
    h_lower = z_term + s_term_lower
    h_upper = z_term + s_term_upper

    w2_pos, w2_neg = _positive_negative(prm.w_l2)
    y_n_lower_flat = w2_pos @ h_lower + w2_neg @ h_upper + prm.b_l2
    y_n_upper_flat = w2_pos @ h_upper + w2_neg @ h_lower + prm.b_l2

    cfg = prm.config
    y_n_lower = cp.reshape(y_n_lower_flat, (cfg.horizon, cfg.n_outputs), order="C")
    y_n_upper = cp.reshape(y_n_upper_flat, (cfg.horizon, cfg.n_outputs), order="C")
    y_lower = cp.multiply(y_n_lower, prm.y_scale) + prm.y_mean
    y_upper = cp.multiply(y_n_upper, prm.y_scale) + prm.y_mean
    return y_lower, y_upper



def _output_interval_numpy(
    prm: TFEParameters,
    z1: np.ndarray,
    s_lower: np.ndarray,
    s_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical counterpart of _output_interval_cvx for post-solve diagnostics."""

    w1_pos, w1_neg = _positive_negative(prm.w_l1)
    s_term_lower = s_lower @ w1_pos + s_upper @ w1_neg
    s_term_upper = s_upper @ w1_pos + s_lower @ w1_neg

    z_term = z1 @ prm.w_l1 + prm.b_l1
    h_lower = z_term + s_term_lower
    h_upper = z_term + s_term_upper

    w2_pos, w2_neg = _positive_negative(prm.w_l2)
    y_n_lower_flat = w2_pos @ h_lower + w2_neg @ h_upper + prm.b_l2
    y_n_upper_flat = w2_pos @ h_upper + w2_neg @ h_lower + prm.b_l2

    cfg = prm.config
    y_n_lower = np.asarray(y_n_lower_flat, dtype=np.float64).reshape(
        cfg.horizon, cfg.n_outputs
    )
    y_n_upper = np.asarray(y_n_upper_flat, dtype=np.float64).reshape(
        cfg.horizon, cfg.n_outputs
    )

    y_lower = y_n_lower * prm.y_scale + prm.y_mean
    y_upper = y_n_upper * prm.y_scale + prm.y_mean
    return y_lower, y_upper


def _tracking_upper_numpy(
    tracking_weight: float,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Worst-case componentwise squared tracking cost for a numerical Y-box."""
    return float(
        tracking_weight
        * np.sum(
            np.maximum(
                (np.asarray(y_lower) - reference) ** 2,
                (np.asarray(y_upper) - reference) ** 2,
            )
        )
    )


def _containment_violation(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    below = np.maximum(lower - value, 0.0)
    above = np.maximum(value - upper, 0.0)
    return float(max(np.max(below), np.max(above)))


def _ordering_violation(lower: np.ndarray, upper: np.ndarray) -> float:
    """Maximum post-solve violation of lower <= upper.

    This is a numerical diagnostic only. A positive value means the conic solver
    returned at least one endpoint pair with lower > upper by that amount.
    """

    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    return float(max(np.max(lower_arr - upper_arr), 0.0))


def _width_geometry(width: np.ndarray, *, floor: float = 1e-12) -> tuple[float, float, float, float]:
    """Return min positive, median, max, and max/min positive box width."""
    arr = np.asarray(width, dtype=np.float64)
    positive = arr[arr > floor]
    minimum = float(np.min(positive)) if positive.size else 0.0
    median = float(np.median(arr))
    maximum = float(np.max(arr))
    dynamic_range = maximum / minimum if minimum > 0.0 else float("inf")
    return minimum, median, maximum, dynamic_range


def _diagnostic_box(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a numerically ordered copy of a solved box for diagnostics.

    The optimization variables themselves are not changed. If the solver returns
    a tiny lower > upper violation, the endpoints are reordered only for
    post-processing functions that require strict componentwise ordering. A large
    violation raises immediately instead of being hidden.
    """

    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    violation = _ordering_violation(lower_arr, upper_arr)
    if violation > tolerance:
        raise RuntimeError(
            f"{name}-box ordering violation too large: {violation:.3e} "
            f"(diagnostic tolerance {tolerance:.1e})."
        )

    ordered_lower = np.minimum(lower_arr, upper_arr)
    ordered_upper = np.maximum(lower_arr, upper_arr)
    return ordered_lower, ordered_upper, violation


class BoxConicCCP:
    """Hyperrectangle-based CCP controller with Rough Stage-2 bounds.

    Stages 1-3 are all present in the convex subproblem:

        U -> Z1 -> [P_lower,P_upper] -> [R_lower,R_upper]
          -> [S_lower,S_upper] -> output interval -> tracking epigraph.

    Stage 1 uses the paper's DC lower/upper approximation directly. Stage 2
    uses the controller-compatible Rough lower bound and exact critical-vertex
    upper bound. Stage 3 uses the monotone-DC construction from Notes 17 Aug,
    specialized to an uncertain R-box and exact affine V(U).

    The default optimization mode is Rough. An experimental
    ``q_box_linearized`` mode is also provided. It is not identical to the
    exact June-20 numerical Q-box benchmark: lower Q endpoints are linearized
    to preserve DCP. The exact June-20 Q-box is still evaluated after each
    solve as a post-solve diagnostic.
    """

    def __init__(
        self,
        parameters: TFEParameters,
        bounds: MPCBounds,
        config: BoxCCPConfig = BoxCCPConfig(),
    ) -> None:
        if config.stage2_method not in {"rough", "q_box_linearized"}:
            raise ValueError(f"Unsupported Stage-2 controller method: {config.stage2_method}")
        self.parameters = parameters
        self.bounds = bounds
        self.config = config

    def _one_iteration(
        self,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        linearization_u: np.ndarray,
    ) -> _BoxSolution:
        cp = _require_cvxpy()
        prm = self.parameters
        cfg = prm.config
        current = forward_from_parameters(prm, past_u, past_y, linearization_u)
        n, d_key = cfg.sequence_length, cfg.d_key

        u = cp.Variable((cfg.horizon, cfg.n_inputs), name="U")
        p_lower = cp.Variable((n, n), name="P_lower")
        p_upper = cp.Variable((n, n), name="P_upper")
        r_lower = cp.Variable((n, n), name="R_lower")
        r_upper = cp.Variable((n, n), name="R_upper")
        s_lower = cp.Variable((n, d_key), name="S_lower")
        s_upper = cp.Variable((n, d_key), name="S_upper")
        tracking_terms = cp.Variable(cfg.output_size, nonneg=True, name="tracking_terms")

        past_u_n = (np.asarray(past_u) - prm.u_mean) / prm.u_scale
        past_y_n = (np.asarray(past_y) - prm.y_mean) / prm.y_scale
        past_xy_n = np.concatenate((past_u_n, past_y_n), axis=1)
        z_past = past_xy_n @ prm.w_em1.T + prm.b_em1
        u_n = cp.multiply(u - prm.u_mean, 1.0 / prm.u_scale)
        z_future = u_n @ prm.w_em2.T + np.ones((cfg.horizon, 1)) @ prm.b_em2.reshape(1, -1)
        z1 = cp.vstack((z_past, z_future))
        q = z1 @ prm.w_q.T + np.ones((n, 1)) @ prm.b_q.reshape(1, -1)
        k = z1 @ prm.w_k.T + np.ones((n, 1)) @ prm.b_k.reshape(1, -1)
        v = z1 @ prm.w_v.T + np.ones((n, 1)) @ prm.b_v.reshape(1, -1)

        constraints: list[Any] = [p_lower <= p_upper, r_lower <= r_upper, s_lower <= s_upper]

        # Fixed domains used only for safe Jacobian bounds / numerical stability.
        q_dom_lo, q_dom_hi, k_dom_lo, k_dom_hi, v_dom_lo, v_dom_hi = _qkv_global_intervals(
            prm, self.bounds, current
        )
        p_dom_lo_raw, p_dom_hi_raw = _stage1_approximation_domain(
            q_dom_lo, q_dom_hi, k_dom_lo, k_dom_hi, current.q, current.k
        )

        # Numerical guard for a theoretically ordered fixed P-domain.
        # Tiny reversals can occur from floating-point roundoff when the
        # theoretical lower and upper endpoints coincide.
        p_dom_violation = float(np.max(p_dom_lo_raw - p_dom_hi_raw))
        p_dom_scale = float(
            max(
                1.0,
                np.max(np.abs(p_dom_lo_raw)),
                np.max(np.abs(p_dom_hi_raw)),
            )
        )
        p_dom_relative_violation = p_dom_violation / p_dom_scale

        # Use a combined absolute/relative tolerance.  Anything materially
        # larger is treated as a real domain-construction error rather than
        # silently hidden.
        p_dom_tolerance = max(1e-12, 1e-12 * p_dom_scale)
        if p_dom_violation > p_dom_tolerance:
            raise RuntimeError(
                "P-domain ordering violation too large: "
                f"absolute={p_dom_violation:.3e}, "
                f"relative={p_dom_relative_violation:.3e}, "
                f"tolerance={p_dom_tolerance:.3e}"
            )

        p_dom_lo = np.minimum(p_dom_lo_raw, p_dom_hi_raw)
        p_dom_hi = np.maximum(p_dom_lo_raw, p_dom_hi_raw)

        # Conditioning diagnostic A: numerical scale seen by self-attention.
        # Diagnostic only; this does not modify the optimization problem.
        max_abs_q = float(np.max(np.abs(current.q)))
        max_abs_k = float(np.max(np.abs(current.k)))
        max_abs_p = float(np.max(np.abs(current.p)))
        scaled_p = np.asarray(current.p, dtype=np.float64) / math.sqrt(d_key)
        scaled_p_min = float(np.min(scaled_p))
        scaled_p_max = float(np.max(scaled_p))
        scaled_p_range = scaled_p_max - scaled_p_min

        p_domain_scaled_lo = p_dom_lo / math.sqrt(d_key)
        p_domain_scaled_hi = p_dom_hi / math.sqrt(d_key)
        p_domain_scaled_min = float(np.min(p_domain_scaled_lo))
        p_domain_scaled_max = float(np.max(p_domain_scaled_hi))
        p_domain_scaled_range = p_domain_scaled_max - p_domain_scaled_min

        if self.config.print_diagnostics:
            print(
                "Conditioning pre-solve:",
                f"max|Q|={max_abs_q:.3e},",
                f"max|K|={max_abs_k:.3e},",
                f"max|P|={max_abs_p:.3e},",
                f"P/sqrt(dk)=[{scaled_p_min:.3e},{scaled_p_max:.3e}],",
                f"range={scaled_p_range:.3e},",
                f"domain-range={p_domain_scaled_range:.3e}",
            )

        if self.config.print_diagnostics and p_dom_violation > 0.0:
            print(
                "P-domain numerical reorder:",
                f"max violation={p_dom_violation:.3e},",
                f"scale={p_dom_scale:.3e},",
                f"relative={p_dom_relative_violation:.3e}",
            )

        # Stage 1: U -> P-box.
        lower_p, upper_p = _stage1_bounds_cvx(cp, q, k, current.q, current.k)
        for i in range(n):
            for j in range(n):
                constraints.append(p_lower[i, j] <= lower_p[i][j])
                constraints.append(p_upper[i, j] >= upper_p[i][j])
        constraints.extend([p_lower >= p_dom_lo, p_upper <= p_dom_hi])

        # Stage 2: P-box -> R-box.
        if self.config.stage2_method == "rough":
            lower_r, upper_r = _stage2_rough_box_bounds_cvx(
                cp, p_lower, p_upper, current.p, d_key
            )
        else:
            lower_r, upper_r = _stage2_qbox_linearized_bounds_cvx(
                cp, p_lower, p_upper, current.p, d_key
            )
        for i in range(n):
            for j in range(n):
                constraints.append(r_lower[i, j] <= lower_r[i][j])
                constraints.append(r_upper[i, j] >= upper_r[i][j])

        # Safe fixed domain for all Stage-2 endpoint variables. The global
        # P-domain is a bound on the Stage-1 approximation endpoints themselves,
        # so evaluating the monotone Rough formulas on that domain is valid.
        if self.config.stage2_method == "rough":
            r_dom_lo, r_dom_hi = stage2_rough_box_bounds(
                p_dom_lo, p_dom_hi, current.p, d_key
            )
        else:
            r_dom_lo, r_dom_hi = stage2_q_box_linearized_bounds(
                p_dom_lo, p_dom_hi, current.p, d_key
            )
        constraints.extend([r_lower >= r_dom_lo, r_upper <= r_dom_hi])

        # Stage 3: R-box + exact V(U) -> S-box.
        lower_s, upper_s = _stage3_monotone_box_bounds_cvx(
            cp,
            r_lower,
            r_upper,
            v,
            current.r,
            current.v,
            r_dom_lo,
            r_dom_hi,
            v_dom_lo,
            v_dom_hi,
        )
        for i in range(n):
            for a in range(d_key):
                constraints.append(s_lower[i, a] <= lower_s[i][a])
                constraints.append(s_upper[i, a] >= upper_s[i][a])

        # Scalable robust readout. This is an explicit extra relaxation beyond
        # Notes 17 Aug: first enclose the affine image of the S-box by a Y-box,
        # then upper-bound tracking error componentwise.
        y_lower, y_upper = _output_interval_cvx(cp, prm, z1, s_lower, s_upper)
        reference = np.asarray(y_reference, dtype=np.float64).reshape(cfg.horizon, cfg.n_outputs)
        err_lower = cp.reshape(y_lower - reference, (cfg.output_size,), order="C")
        err_upper = cp.reshape(y_upper - reference, (cfg.output_size,), order="C")
        constraints.extend(
            [tracking_terms >= cp.square(err_lower), tracking_terms >= cp.square(err_upper)]
        )
        tracking_upper = self.config.tracking_weight * cp.sum(tracking_terms)

        u_flat = cp.reshape(u, (cfg.horizon * cfg.n_inputs,), order="C")
        dmat = difference_matrix(cfg.horizon, cfg.n_inputs)
        offset = delta_offset(previous_u, cfg.horizon)
        delta_u = dmat @ u_flat - offset
        constraints.extend(
            [
                u >= np.asarray(self.bounds.u_min),
                u <= np.asarray(self.bounds.u_max),
                delta_u >= np.tile(np.asarray(self.bounds.du_min), cfg.horizon),
                delta_u <= np.tile(np.asarray(self.bounds.du_max), cfg.horizon),
            ]
        )
        control_cost = self.config.control_weight * cp.sum_squares(delta_u)
        objective = cp.Minimize(tracking_upper + control_cost)
        problem = cp.Problem(objective, constraints)
        if not problem.is_dcp():
            raise RuntimeError("Internal error: BoxConicCCP subproblem is not DCP.")

        size_metrics = problem.size_metrics
        start = time.perf_counter()
        problem.solve(solver=self.config.solver, verbose=self.config.verbose, warm_start=True)
        elapsed = time.perf_counter() - start
        solver_stats = problem.solver_stats
        solver_internal_time = getattr(solver_stats, "solve_time", None)
        solver_num_iters = getattr(solver_stats, "num_iters", None)
        if u.value is None:
            raise RuntimeError(f"Box CCP subproblem failed with status {problem.status}")

        # Post-solve Stage-3 diagnostic:
        # evaluate the Stage-3 lower/upper expressions at the solved point.
        s_expr_lower = np.array(
            [
                [float(lower_s[i][a].value) for a in range(d_key)]
                for i in range(n)
            ],
            dtype=np.float64,
        )
        s_expr_upper = np.array(
            [
                [float(upper_s[i][a].value) for a in range(d_key)]
                for i in range(n)
            ],
            dtype=np.float64,
        )

        return _BoxSolution(
            future_u=np.asarray(u.value, dtype=np.float64),
            objective=float(problem.value),
            tracking_upper=float(tracking_upper.value),
            control_cost=float(control_cost.value),
            status=str(problem.status),
            solve_time=float(elapsed),
            solver_internal_time=(
                float(solver_internal_time) if solver_internal_time is not None else float("nan")
            ),
            solver_num_iters=int(solver_num_iters) if solver_num_iters is not None else -1,
            num_scalar_variables=int(size_metrics.num_scalar_variables),
            num_scalar_eq_constraints=int(size_metrics.num_scalar_eq_constr),
            num_scalar_leq_constraints=int(size_metrics.num_scalar_leq_constr),
            p_lower=np.asarray(p_lower.value, dtype=np.float64),
            p_upper=np.asarray(p_upper.value, dtype=np.float64),
            r_lower=np.asarray(r_lower.value, dtype=np.float64),
            r_upper=np.asarray(r_upper.value, dtype=np.float64),
            s_lower=np.asarray(s_lower.value, dtype=np.float64),
            s_upper=np.asarray(s_upper.value, dtype=np.float64),
            s_expr_lower=s_expr_lower,
            s_expr_upper=s_expr_upper,
            max_abs_q=max_abs_q,
            max_abs_k=max_abs_k,
            max_abs_p=max_abs_p,
            scaled_p_min=scaled_p_min,
            scaled_p_max=scaled_p_max,
            scaled_p_range=scaled_p_range,
            p_domain_scaled_min=p_domain_scaled_min,
            p_domain_scaled_max=p_domain_scaled_max,
            p_domain_scaled_range=p_domain_scaled_range,
        )

    def solve(
        self,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        initial_u: np.ndarray,
    ) -> BoxCCPResult:
        cfg = self.parameters.config
        u = np.asarray(initial_u, dtype=np.float64).reshape(cfg.horizon, cfg.n_inputs)
        objectives: list[float] = []
        steps: list[float] = []
        statuses: list[str] = []
        solve_times: list[float] = []
        core_iteration_wall_times: list[float] = []
        diagnostics: list[BoxIterationDiagnostics] = []
        reference = np.asarray(y_reference, dtype=np.float64).reshape(cfg.horizon, cfg.n_outputs)
        _diag_print = print if self.config.print_diagnostics else (lambda *args, **kwargs: None)

        for iteration in range(self.config.max_iterations):
            core_iteration_start = time.perf_counter()
            current = forward_from_parameters(self.parameters, past_u, past_y, u)
            solution = self._one_iteration(past_u, past_y, y_reference, previous_u, u)
            u_next = solution.future_u
            step = float(np.linalg.norm(u_next - u))
            actual = forward_from_parameters(self.parameters, past_u, past_y, u_next)

            u_flat = u_next.reshape(-1)
            dmat = difference_matrix(cfg.horizon, cfg.n_inputs)
            offset = delta_offset(previous_u, cfg.horizon)
            delta = dmat @ u_flat - offset
            actual_tracking = self.config.tracking_weight * float(np.sum((actual.y - reference) ** 2))
            actual_control = self.config.control_weight * float(delta @ delta)
            actual_objective = actual_tracking + actual_control
            core_iteration_wall_time = time.perf_counter() - core_iteration_start

            # Post-solve numerical ordering diagnostics. The conic constraints
            # enforce lower <= upper, but an "optimal_inaccurate" solution can
            # violate this by a tiny solver tolerance. Reorder endpoints only for
            # post-processing; the solved optimization variables are left intact.
            p_lo_diag, p_hi_diag, p_ordering_violation = _diagnostic_box(
                solution.p_lower, solution.p_upper, name="P"
            )
            r_lo_diag, r_hi_diag, r_ordering_violation = _diagnostic_box(
                solution.r_lower, solution.r_upper, name="R"
            )
            s_lo_diag, s_hi_diag, s_ordering_violation = _diagnostic_box(
                solution.s_lower, solution.s_upper, name="S"
            )

            _diag_print(
                f"iter {iteration}: ordering violations "
                f"P={p_ordering_violation:.3e}, "
                f"R={r_ordering_violation:.3e}, "
                f"S={s_ordering_violation:.3e}"
            )

            qbox_lower = stage2_q_box_lower(
                p_lo_diag, p_hi_diag, current.p, cfg.d_key
            )
            rough_lower, _ = stage2_rough_box_bounds(
                p_lo_diag, p_hi_diag, current.p, cfg.d_key
            )
            true_r_min, _ = stage2_true_box_extrema(
                p_lo_diag, p_hi_diag, cfg.d_key
            )
            rq_diff = np.abs(rough_lower - qbox_lower)

            p_width = p_hi_diag - p_lo_diag
            r_width = r_hi_diag - r_lo_diag
            s_width = s_hi_diag - s_lo_diag

            # Conditioning diagnostic B: anisotropy of solved box geometry.
            p_min_positive_width, p_median_width, p_max_width_diag, p_width_dynamic_range = _width_geometry(p_width)
            r_min_positive_width, r_median_width, r_max_width_diag, r_width_dynamic_range = _width_geometry(r_width)
            s_min_positive_width, s_median_width, s_max_width_diag, s_width_dynamic_range = _width_geometry(s_width)

            _diag_print(
                f"iter {iteration}: Conditioning box geometry "
                f"P[min+/med/max/ratio]={p_min_positive_width:.3e}/"
                f"{p_median_width:.3e}/{p_max_width_diag:.3e}/{p_width_dynamic_range:.3e}, "
                f"R={r_min_positive_width:.3e}/{r_median_width:.3e}/"
                f"{r_max_width_diag:.3e}/{r_width_dynamic_range:.3e}, "
                f"S={s_min_positive_width:.3e}/{s_median_width:.3e}/"
                f"{s_max_width_diag:.3e}/{s_width_dynamic_range:.3e}"
            )

            # Stage-3 diagnostic: compare the solved S-variable box against
            # the Stage-3 bound expressions themselves.
            s_expr_width = solution.s_expr_upper - solution.s_expr_lower
            s_lower_slack = solution.s_expr_lower - solution.s_lower
            s_upper_slack = solution.s_upper - solution.s_expr_upper

            _diag_print(
                f"iter {iteration}: Stage-3 "
                f"S var width mean={np.mean(s_width):.3e}, "
                f"S expr width mean={np.mean(s_expr_width):.3e}, "
                f"lower slack mean={np.mean(s_lower_slack):.3e}, "
                f"upper slack mean={np.mean(s_upper_slack):.3e}"
            )

            # Fixed-point diagnostic suggested by Mark:
            # compare the Stage-3 value at the current CCP linearisation point
            # with the solved S_lower/S_upper variables. Near a fixed point,
            # current.s should lie on (or close to) at least one of the bounds.
            s_lin = np.asarray(current.s, dtype=np.float64)
            s_dist_lower = np.abs(s_lin - solution.s_lower)
            s_dist_upper = np.abs(solution.s_upper - s_lin)
            s_nearest_dist = np.minimum(s_dist_lower, s_dist_upper)
            s_lower_is_nearest = s_dist_lower <= s_dist_upper
            s_upper_is_nearest = ~s_lower_is_nearest

            s_linearization_nearest_bound_mean = float(np.mean(s_nearest_dist))
            s_linearization_nearest_bound_max = float(np.max(s_nearest_dist))
            s_linearization_lower_nearest_fraction = float(np.mean(s_lower_is_nearest))
            s_linearization_upper_nearest_fraction = float(np.mean(s_upper_is_nearest))

            _diag_print(
                f"iter {iteration}: S-linearisation-to-nearest-bound "
                f"mean={s_linearization_nearest_bound_mean:.3e}, "
                f"max={s_linearization_nearest_bound_max:.3e}, "
                f"lower-nearest={s_linearization_lower_nearest_fraction:.3f}, "
                f"upper-nearest={s_linearization_upper_nearest_fraction:.3f}"
            )

            # Downstream slack diagnostic:
            # compare the output interval produced by the solved slack S-box
            # against the interval produced by the tight Stage-3 expressions,
            # at exactly the same solved U.
            y_var_lower, y_var_upper = _output_interval_numpy(
                self.parameters,
                actual.z1,
                solution.s_lower,
                solution.s_upper,
            )
            y_expr_lower, y_expr_upper = _output_interval_numpy(
                self.parameters,
                actual.z1,
                solution.s_expr_lower,
                solution.s_expr_upper,
            )

            y_var_width = y_var_upper - y_var_lower
            y_expr_width = y_expr_upper - y_expr_lower
            y_var_mean_width = float(np.mean(y_var_width))
            y_expr_mean_width = float(np.mean(y_expr_width))
            y_width_ratio = y_var_mean_width / max(y_expr_mean_width, 1e-15)

            tracking_upper_from_s_var = _tracking_upper_numpy(
                self.config.tracking_weight,
                y_var_lower,
                y_var_upper,
                reference,
            )
            tracking_upper_from_s_expr = _tracking_upper_numpy(
                self.config.tracking_weight,
                y_expr_lower,
                y_expr_upper,
                reference,
            )
            tracking_upper_difference = (
                tracking_upper_from_s_var - tracking_upper_from_s_expr
            )

            _diag_print(
                f"iter {iteration}: Output-box "
                f"Y var width mean={y_var_mean_width:.3e}, "
                f"Y expr width mean={y_expr_mean_width:.3e}, "
                f"ratio={y_width_ratio:.3e}"
            )
            _diag_print(
                f"iter {iteration}: Tracking diagnostic "
                f"current S-box={tracking_upper_from_s_var:.6e}, "
                f"tight expr S-box={tracking_upper_from_s_expr:.6e}, "
                f"difference={tracking_upper_difference:.6e}"
            )

            # Diagnostic 1: which Y endpoint is active in the worst-case
            # tracking error, and whether that active side changes when the
            # S-box is tightened to the Stage-3 expressions.
            var_lower_error_sq = (y_var_lower - reference) ** 2
            var_upper_error_sq = (y_var_upper - reference) ** 2
            expr_lower_error_sq = (y_expr_lower - reference) ** 2
            expr_upper_error_sq = (y_expr_upper - reference) ** 2

            active_var_lower = var_lower_error_sq >= var_upper_error_sq
            active_expr_lower = expr_lower_error_sq >= expr_upper_error_sq

            active_endpoint_agreement_fraction = float(
                np.mean(active_var_lower == active_expr_lower)
            )
            active_var_lower_fraction = float(np.mean(active_var_lower))
            active_expr_lower_fraction = float(np.mean(active_expr_lower))

            _diag_print(
                f"iter {iteration}: Active-endpoint "
                f"agreement={active_endpoint_agreement_fraction:.3f}, "
                f"var lower-active={active_var_lower_fraction:.3f}, "
                f"expr lower-active={active_expr_lower_fraction:.3f}"
            )

            # Diagnostic 2: continuously interpolate between the tight
            # Stage-3 expression box (alpha=0) and the solved slack S-box
            # (alpha=1), and measure the tracking upper bound.
            alpha_values = (0.0, 0.25, 0.50, 0.75, 1.0)
            alpha_tracking_values: list[float] = []

            for alpha in alpha_values:
                s_alpha_lower = (
                    solution.s_expr_lower
                    - alpha * (solution.s_expr_lower - solution.s_lower)
                )
                s_alpha_upper = (
                    solution.s_expr_upper
                    + alpha * (solution.s_upper - solution.s_expr_upper)
                )

                y_alpha_lower, y_alpha_upper = _output_interval_numpy(
                    self.parameters,
                    actual.z1,
                    s_alpha_lower,
                    s_alpha_upper,
                )
                alpha_tracking_values.append(
                    _tracking_upper_numpy(
                        self.config.tracking_weight,
                        y_alpha_lower,
                        y_alpha_upper,
                        reference,
                    )
                )

            alpha_sweep_range = float(
                max(alpha_tracking_values) - min(alpha_tracking_values)
            )
            alpha_text = ", ".join(
                f"a={alpha:.2f}:{value:.6e}"
                for alpha, value in zip(alpha_values, alpha_tracking_values)
            )
            _diag_print(
                f"iter {iteration}: Alpha-tightening "
                f"{alpha_text}, range={alpha_sweep_range:.3e}"
            )

            # Diagnostic 3: starting from the solved slack S-box, tighten one
            # S coordinate at a time to its Stage-3 expression endpoints and
            # measure the resulting reduction in the tracking upper bound.
            componentwise_reductions = np.zeros_like(
                solution.s_lower,
                dtype=np.float64,
            )

            for s_row in range(solution.s_lower.shape[0]):
                for s_col in range(solution.s_lower.shape[1]):
                    s_test_lower = solution.s_lower.copy()
                    s_test_upper = solution.s_upper.copy()
                    s_test_lower[s_row, s_col] = solution.s_expr_lower[s_row, s_col]
                    s_test_upper[s_row, s_col] = solution.s_expr_upper[s_row, s_col]

                    y_test_lower, y_test_upper = _output_interval_numpy(
                        self.parameters,
                        actual.z1,
                        s_test_lower,
                        s_test_upper,
                    )
                    test_tracking = _tracking_upper_numpy(
                        self.config.tracking_weight,
                        y_test_lower,
                        y_test_upper,
                        reference,
                    )
                    componentwise_reductions[s_row, s_col] = (
                        tracking_upper_from_s_var - test_tracking
                    )

            componentwise_max_tracking_reduction = float(
                np.max(componentwise_reductions)
            )
            componentwise_mean_tracking_reduction = float(
                np.mean(componentwise_reductions)
            )
            argmax_flat = int(np.argmax(componentwise_reductions))
            componentwise_argmax_row, componentwise_argmax_col = np.unravel_index(
                argmax_flat,
                componentwise_reductions.shape,
            )

            _diag_print(
                f"iter {iteration}: Componentwise-S-slack "
                f"max reduction={componentwise_max_tracking_reduction:.3e} "
                f"at ({componentwise_argmax_row},{componentwise_argmax_col}), "
                f"mean reduction={componentwise_mean_tracking_reduction:.3e}"
            )

            diagnostics.append(
                BoxIterationDiagnostics(
                    iteration=iteration,
                    surrogate_objective=solution.objective,
                    tracking_upper_bound=solution.tracking_upper,
                    actual_tracking_cost=actual_tracking,
                    control_cost=actual_control,
                    actual_objective=actual_objective,
                    step_norm=step,
                    solve_time_seconds=solution.solve_time,
                    core_iteration_wall_time_seconds=core_iteration_wall_time,
                    solver_internal_time_seconds=solution.solver_internal_time,
                    solver_num_iters=solution.solver_num_iters,
                    num_scalar_variables=solution.num_scalar_variables,
                    num_scalar_eq_constraints=solution.num_scalar_eq_constraints,
                    num_scalar_leq_constraints=solution.num_scalar_leq_constraints,
                    status=solution.status,
                    p_mean_width=float(np.mean(p_width)),
                    p_max_width=float(np.max(p_width)),
                    r_mean_width=float(np.mean(r_width)),
                    r_max_width=float(np.max(r_width)),
                    s_mean_width=float(np.mean(s_width)),
                    s_max_width=float(np.max(s_width)),
                    p_containment_violation=_containment_violation(actual.p, p_lo_diag, p_hi_diag),
                    r_containment_violation=_containment_violation(actual.r, r_lo_diag, r_hi_diag),
                    s_containment_violation=_containment_violation(actual.s, s_lo_diag, s_hi_diag),
                    p_ordering_violation=p_ordering_violation,
                    r_ordering_violation=r_ordering_violation,
                    s_ordering_violation=s_ordering_violation,
                    s_expr_mean_width=float(np.mean(s_expr_width)),
                    s_expr_max_width=float(np.max(s_expr_width)),
                    s_lower_slack_mean=float(np.mean(s_lower_slack)),
                    s_upper_slack_mean=float(np.mean(s_upper_slack)),
                    s_linearization_nearest_bound_mean=s_linearization_nearest_bound_mean,
                    s_linearization_nearest_bound_max=s_linearization_nearest_bound_max,
                    s_linearization_lower_nearest_fraction=s_linearization_lower_nearest_fraction,
                    s_linearization_upper_nearest_fraction=s_linearization_upper_nearest_fraction,
                    y_var_mean_width=y_var_mean_width,
                    y_expr_mean_width=y_expr_mean_width,
                    y_width_ratio=y_width_ratio,
                    tracking_upper_from_s_var=tracking_upper_from_s_var,
                    tracking_upper_from_s_expr=tracking_upper_from_s_expr,
                    tracking_upper_difference=tracking_upper_difference,
                    active_endpoint_agreement_fraction=active_endpoint_agreement_fraction,
                    active_var_lower_fraction=active_var_lower_fraction,
                    active_expr_lower_fraction=active_expr_lower_fraction,
                    alpha_sweep_range=alpha_sweep_range,
                    componentwise_max_tracking_reduction=componentwise_max_tracking_reduction,
                    componentwise_mean_tracking_reduction=componentwise_mean_tracking_reduction,
                    componentwise_argmax_row=int(componentwise_argmax_row),
                    componentwise_argmax_col=int(componentwise_argmax_col),
                    rough_qbox_mean_abs_difference=float(np.mean(rq_diff)),
                    rough_qbox_max_abs_difference=float(np.max(rq_diff)),
                    rough_mean_gap_to_true_softmax_min=float(np.mean(true_r_min - rough_lower)),
                    qbox_mean_gap_to_true_softmax_min=float(np.mean(true_r_min - qbox_lower)),
                    max_abs_q=solution.max_abs_q,
                    max_abs_k=solution.max_abs_k,
                    max_abs_p=solution.max_abs_p,
                    scaled_p_min=solution.scaled_p_min,
                    scaled_p_max=solution.scaled_p_max,
                    scaled_p_range=solution.scaled_p_range,
                    p_domain_scaled_min=solution.p_domain_scaled_min,
                    p_domain_scaled_max=solution.p_domain_scaled_max,
                    p_domain_scaled_range=solution.p_domain_scaled_range,
                    p_min_positive_width=p_min_positive_width,
                    p_median_width=p_median_width,
                    p_width_dynamic_range=p_width_dynamic_range,
                    r_min_positive_width=r_min_positive_width,
                    r_median_width=r_median_width,
                    r_width_dynamic_range=r_width_dynamic_range,
                    s_min_positive_width=s_min_positive_width,
                    s_median_width=s_median_width,
                    s_width_dynamic_range=s_width_dynamic_range,
                )
            )

            objectives.append(solution.objective)
            steps.append(step)
            statuses.append(solution.status)
            solve_times.append(solution.solve_time)
            core_iteration_wall_times.append(core_iteration_wall_time)
            u = u_next
            if step <= self.config.tolerance:
                break

        final_forward = forward_from_parameters(self.parameters, past_u, past_y, u)
        return BoxCCPResult(
            future_u=u,
            predicted_y=final_forward.y,
            objective_history=objectives,
            step_history=steps,
            status_history=statuses,
            solve_time_history=solve_times,
            core_iteration_wall_time_history=core_iteration_wall_times,
            diagnostics=diagnostics,
        )
