from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from .export import TFEParameters


@dataclass(frozen=True)
class ForwardIntermediates:
    z1: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    p: np.ndarray
    logits: np.ndarray
    r: np.ndarray
    s: np.ndarray
    h: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Stage2BoxBounds:
    """Collection of Stage-2 bounds evaluated on one P hyperrectangle.

    All P arguments supplied to the public functions in this module are the
    *unscaled* matrices P = Q K.T. The 1/sqrt(d_key) scaling is applied only
    inside Stage 2, matching the corrected paper and the existing codebase.
    """

    true_lower: np.ndarray
    true_upper: np.ndarray
    p_exact_lower: np.ndarray
    common_lower: np.ndarray
    q_box_lower: np.ndarray
    rough_lower: np.ndarray
    u_upper: np.ndarray


def softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _logsumexp_1d(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _leave_one_out_logsumexp(values: np.ndarray) -> np.ndarray:
    """Return q_i = log(sum_{j != i} exp(values_j)) in O(N)."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("Stage-2 leave-one-out log-sum-exp requires N >= 2.")
    maximum = float(np.max(values))
    exponentials = np.exp(values - maximum)
    total = float(exponentials.sum())
    remainder = total - exponentials
    if np.any(remainder <= 0.0):
        raise FloatingPointError("Invalid leave-one-out log-sum-exp remainder.")
    return maximum + np.log(remainder)


def _stage2_scaled_box_inputs(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_key <= 0:
        raise ValueError("d_key must be positive.")
    lower = np.asarray(p_lower, dtype=np.float64)
    upper = np.asarray(p_upper, dtype=np.float64)
    centre = np.asarray(p0, dtype=np.float64)
    if lower.shape != upper.shape or lower.shape != centre.shape:
        raise ValueError(
            "p_lower, p_upper and p0 must have identical matrix shapes."
        )
    if lower.ndim != 2 or lower.shape[0] != lower.shape[1]:
        raise ValueError("Stage-2 P bounds must be square N x N matrices.")
    if np.any(lower > upper):
        raise ValueError("p_lower must be componentwise <= p_upper.")
    scale = math.sqrt(d_key)
    return lower / scale, upper / scale, centre / scale


def _stage2_row_lower_values(points: np.ndarray, p0_row: np.ndarray) -> np.ndarray:
    """Evaluate the original concave Stage-2 lower function at many points.

    Parameters
    ----------
    points:
        Shape (M, N), already scaled by 1/sqrt(d_key).
    p0_row:
        Shape (N,), already scaled by 1/sqrt(d_key).

    Returns
    -------
    Shape (M, N). Entry [m, i] is L_i(points[m]; p0_row).
    """

    points = np.asarray(points, dtype=np.float64)
    p0_row = np.asarray(p0_row, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != p0_row.size:
        raise ValueError("points must have shape (M, N) matching p0_row.")
    s0 = softmax_rows(p0_row.reshape(1, -1))[0]
    lse0 = _logsumexp_1d(p0_row)
    tangent = lse0 + (points - p0_row) @ s0
    log_terms = points - tangent[:, None]
    with np.errstate(over="ignore"):
        exp_terms = np.exp(log_terms)
    total = exp_terms.sum(axis=1, keepdims=True)
    return 1.0 - (total - exp_terms)


def _stage2_row_upper_values(points: np.ndarray, p0_row: np.ndarray) -> np.ndarray:
    """Evaluate the original convex Stage-2 upper function at many points."""

    points = np.asarray(points, dtype=np.float64)
    p0_row = np.asarray(p0_row, dtype=np.float64).reshape(-1)
    s0 = softmax_rows(p0_row.reshape(1, -1))[0]
    lse0 = _logsumexp_1d(p0_row)
    tangent = lse0 + (points - p0_row) @ s0
    with np.errstate(over="ignore"):
        return np.exp(points - tangent[:, None])


def forward_from_parameters(
    params: TFEParameters,
    past_u: np.ndarray,
    past_y: np.ndarray,
    future_u: np.ndarray,
) -> ForwardIntermediates:
    """Evaluate the exported model in physical units.

    ``p`` is the unscaled matrix P = Q K.T in corrected equation (7), while
    ``logits`` is P / sqrt(d_key), the argument of softmax in equation (3).
    """

    config = params.config
    past_u_n = (np.asarray(past_u) - params.u_mean) / params.u_scale
    past_y_n = (np.asarray(past_y) - params.y_mean) / params.y_scale
    future_u_n = (np.asarray(future_u) - params.u_mean) / params.u_scale
    past_xy = np.concatenate((past_u_n, past_y_n), axis=1)
    z_past = past_xy @ params.w_em1.T + params.b_em1
    z_future = future_u_n @ params.w_em2.T + params.b_em2
    z1 = np.vstack((z_past, z_future))
    q = z1 @ params.w_q.T + params.b_q
    k = z1 @ params.w_k.T + params.b_k
    v = z1 @ params.w_v.T + params.b_v
    p = q @ k.T
    logits = p / math.sqrt(config.d_key)
    r = softmax_rows(logits)
    s = r @ v
    h = (z1 + s) @ params.w_l1 + params.b_l1
    y_n = (params.w_l2 @ h + params.b_l2).reshape(config.horizon, config.n_outputs)
    y = y_n * params.y_scale + params.y_mean
    return ForwardIntermediates(z1, q, k, v, p, logits, r, s, h, y)


def stage1_bounds(
    q: np.ndarray,
    k: np.ndarray,
    q0: np.ndarray,
    k0: np.ndarray,
    d_key: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds for P = Q K.T from corrected equations (11)-(12).

    ``d_key`` is accepted only for source compatibility with version 0.1 and
    is intentionally not used: the corrected paper defines P before the
    ``1/sqrt(d_key)`` attention scaling.
    """

    if d_key is not None and d_key <= 0:
        raise ValueError("d_key must be positive when supplied.")
    length = q.shape[0]
    lower = np.empty((length, length), dtype=np.float64)
    upper = np.empty((length, length), dtype=np.float64)
    for i in range(length):
        for j in range(length):
            plus = q[i] + k[j]
            minus = q[i] - k[j]
            plus0 = q0[i] + k0[j]
            minus0 = q0[i] - k0[j]
            p1_linear = 0.25 * plus0 @ plus0 + 0.5 * plus0 @ (plus - plus0)
            p2 = 0.25 * minus @ minus
            p1 = 0.25 * plus @ plus
            p2_linear = 0.25 * minus0 @ minus0 + 0.5 * minus0 @ (minus - minus0)
            lower[i, j] = p1_linear - p2
            upper[i, j] = p1 - p2_linear
    return lower, upper


def stage2_bounds(
    p: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds for R = softmax(P / sqrt(d_key)), equations (14)-(15).

    Both ``p`` and ``p0`` are the unscaled P = Q K.T matrices used by the
    stage-1 simplex in Algorithm 1. Scaling is applied internally here.
    """

    if d_key <= 0:
        raise ValueError("d_key must be positive.")
    p_scaled = np.asarray(p, dtype=np.float64) / math.sqrt(d_key)
    p0_scaled = np.asarray(p0, dtype=np.float64) / math.sqrt(d_key)
    length = p_scaled.shape[0]
    r0 = softmax_rows(p0_scaled)
    lower = np.empty_like(p_scaled, dtype=np.float64)
    upper = np.empty_like(p_scaled, dtype=np.float64)
    for row in range(length):
        maximum = np.max(p0_scaled[row])
        lse0 = maximum + np.log(np.exp(p0_scaled[row] - maximum).sum())
        tangent_lse = lse0 + r0[row] @ (p_scaled[row] - p0_scaled[row])
        exp_terms = np.exp(p_scaled[row] - tangent_lse)
        upper[row] = exp_terms
        for column in range(length):
            lower[row, column] = 1.0 - (exp_terms.sum() - exp_terms[column])
    return lower, upper


# ---------------------------------------------------------------------------
# Stage-2 hyperrectangle methods used by the new benchmark pipeline.
# ---------------------------------------------------------------------------


def stage2_true_softmax_box_bounds(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    d_key: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact true softmax range over a *fixed* P hyperrectangle.

    This is a diagnostic reference only. It is not, in general, a convenient
    convex/concave constraint when p_lower and p_upper are optimisation
    variables. For each R_ni, softmax is increasing in P_ni and decreasing in
    P_nj (j != i), so its fixed-box extrema are known exactly.

    The implementation uses leave-one-out log-sum-exp values, giving O(N^2)
    work for the full N x N matrix rather than evaluating N separate softmaxes
    for every row.
    """

    lower, upper, _ = _stage2_scaled_box_inputs(
        p_lower, p_upper, np.asarray(p_lower), d_key
    )
    length = lower.shape[0]
    true_lower = np.empty_like(lower)
    true_upper = np.empty_like(lower)
    for row in range(length):
        # min R_i: P_i at its lower endpoint, all other P_j at upper endpoints
        # max R_i: P_i at its upper endpoint, all other P_j at lower endpoints
        upper_loo = _leave_one_out_logsumexp(upper[row])
        lower_loo = _leave_one_out_logsumexp(lower[row])
        log_den_min = np.logaddexp(lower[row], upper_loo)
        log_den_max = np.logaddexp(upper[row], lower_loo)
        true_lower[row] = np.exp(lower[row] - log_den_min)
        true_upper[row] = np.exp(upper[row] - log_den_max)
    return true_lower, true_upper


def stage2_lower_exact_box(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
    *,
    maximum_vertices: int = 1_048_576,
    return_vertices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Exact minimum of the original concave lower function over a P box.

    The original lower function L_i(P; P^k) is concave in each P row, so its
    minimum over a hyperrectangle occurs at a vertex. This routine enumerates
    all 2^N row vertices and is intended only as a small-N ground truth.

    Parameters use unscaled P. The attention scaling is applied internally.

    Returns
    -------
    lower:
        N x N matrix. Entry (n, i) is min L_ni over row-n P box.
    vertices (optional):
        N x N x N array. vertices[n, i] is one minimizing *unscaled* row
        vertex for L_ni.
    """

    lower_s, upper_s, p0_s = _stage2_scaled_box_inputs(
        p_lower, p_upper, p0, d_key
    )
    length = lower_s.shape[0]
    vertex_count = 1 << length
    if vertex_count > maximum_vertices:
        raise ValueError(
            f"Exact Stage-2 row box has 2^{length}={vertex_count} vertices, "
            f"exceeding maximum_vertices={maximum_vertices}."
        )

    bit_matrix = np.array(
        list(itertools.product((0.0, 1.0), repeat=length)), dtype=np.float64
    )
    result = np.empty_like(lower_s)
    arg_vertices_scaled = np.empty((length, length, length), dtype=np.float64)

    for row in range(length):
        vertices = lower_s[row] + bit_matrix * (upper_s[row] - lower_s[row])
        values = _stage2_row_lower_values(vertices, p0_s[row])
        indices = np.argmin(values, axis=0)
        result[row] = values[indices, np.arange(length)]
        arg_vertices_scaled[row] = vertices[indices]

    if return_vertices:
        return result, arg_vertices_scaled * math.sqrt(d_key)
    return result


def _common_row_lower_o_n(
    lower_scalar: float,
    upper_scalar: float,
    p0_row_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact common-scalar-box lower bound for one row in O(N) per target i.

    Returns the lower values, minimizing scaled vertices, and optimal
    cardinalities k*. The O(N) selection step uses numpy.partition.
    """

    p0_row_scaled = np.asarray(p0_row_scaled, dtype=np.float64).reshape(-1)
    length = p0_row_scaled.size
    if length < 2:
        raise ValueError("Stage 2 requires sequence length N >= 2.")

    values = np.empty(length, dtype=np.float64)
    vertices = np.empty((length, length), dtype=np.float64)
    k_stars = np.zeros(length, dtype=np.int64)

    if math.isclose(lower_scalar, upper_scalar, rel_tol=0.0, abs_tol=1e-14):
        point = np.full(length, lower_scalar, dtype=np.float64)
        row_values = _stage2_row_lower_values(point.reshape(1, -1), p0_row_scaled)[0]
        values[:] = row_values
        vertices[:] = point
        return values, vertices, k_stars

    delta = upper_scalar - lower_scalar
    if delta < 0.0:
        raise ValueError("Common lower scalar must be <= upper scalar.")

    s0 = softmax_rows(p0_row_scaled.reshape(1, -1))[0]
    m = length - 1
    expm1_delta = math.expm1(delta)
    all_indices = np.arange(length)

    for target in range(length):
        mask = all_indices != target
        weights = s0[mask]

        # r_j = largest cardinality k for which s_j <= tau_k.
        # This converts the order-statistic condition into an H-index-like
        # bucket-count problem, avoiding a full O(N log N) sort.
        denom = np.expm1(delta * weights)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            threshold_rank = 1.0 + 1.0 / denom - m / expm1_delta
        threshold_rank = np.where(np.isfinite(threshold_rank), threshold_rank, m)
        r = np.floor(threshold_rank + 1e-12).astype(np.int64)
        r = np.clip(r, 0, m)

        counts = np.bincount(r, minlength=m + 1)
        suffix = np.cumsum(counts[::-1])[::-1]
        feasible_k = np.nonzero(suffix[1:] >= np.arange(1, m + 1))[0]
        k_star = int(feasible_k[-1] + 1) if feasible_k.size else 0
        k_stars[target] = k_star

        vertex = np.full(length, lower_scalar, dtype=np.float64)
        if k_star > 0:
            # Linear-time selection in practice (NumPy uses introselect).
            local_positions = np.argpartition(weights, k_star - 1)[:k_star]
            chosen = all_indices[mask][local_positions]
            vertex[chosen] = upper_scalar
        # P_i is always at the lower endpoint for the minimizer of L_i.
        vertex[target] = lower_scalar
        vertices[target] = vertex
        values[target] = _stage2_row_lower_values(
            vertex.reshape(1, -1), p0_row_scaled
        )[0, target]

    return values, vertices, k_stars


def stage2_lower_common_box(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
    *,
    return_vertices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O(N) common-scalar-box method for the original Stage-2 lower bound.

    For each P row, the componentwise box is enclosed in

        [min_j p_lower_j, max_j p_upper_j]^N.

    The minimum of the original L_i over this *enlarged common box* is then
    found exactly using the threshold/bucket method from Notes_21Aug.
    Therefore this is exact for the common enclosing box and conservative for
    the original componentwise box.
    """

    lower_s, upper_s, p0_s = _stage2_scaled_box_inputs(
        p_lower, p_upper, p0, d_key
    )
    length = lower_s.shape[0]
    result = np.empty_like(lower_s)
    arg_vertices_scaled = np.empty((length, length, length), dtype=np.float64)
    k_stars = np.empty((length, length), dtype=np.int64)

    for row in range(length):
        common_lower = float(np.min(lower_s[row]))
        common_upper = float(np.max(upper_s[row]))
        row_values, row_vertices, row_k = _common_row_lower_o_n(
            common_lower, common_upper, p0_s[row]
        )
        result[row] = row_values
        arg_vertices_scaled[row] = row_vertices
        k_stars[row] = row_k

    if return_vertices:
        return result, arg_vertices_scaled * math.sqrt(d_key), k_stars
    return result


def stage2_lower_q_box(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> np.ndarray:
    """Conservative transformed-Q-box lower bound from June_20Aug.

    Define Q_i = log(sum_{j != i} exp(P_j)) in the *scaled* Stage-2
    coordinates. The transformed lower function is decreasing in Q_i and
    increasing in Q_j (j != i), so its minimum over the Q bounding box occurs
    at Q_i = Q_i^upper and Q_j = Q_j^lower for j != i.

    The selected Q vertex need not be reachable from one P vertex, so the
    result can be conservative for N > 2.
    """

    lower_s, upper_s, p0_s = _stage2_scaled_box_inputs(
        p_lower, p_upper, p0, d_key
    )
    length = lower_s.shape[0]
    if length < 2:
        raise ValueError("Q-box transformation requires N >= 2.")

    result = np.empty_like(lower_s)
    log_n_minus_one = math.log(length - 1)

    for row in range(length):
        q0 = _leave_one_out_logsumexp(p0_s[row])
        q_lower = _leave_one_out_logsumexp(lower_s[row])
        q_upper = _leave_one_out_logsumexp(upper_s[row])
        q_softmax0 = softmax_rows(q0.reshape(1, -1))[0]
        q_lse0 = _logsumexp_1d(q0)

        # Start from the all-lower Q point and replace only component i by
        # Q_i^upper. This gives the minimizing Q-box vertex for L_i.
        base_tangent = q_lse0 + q_softmax0 @ (q_lower - q0)
        delta = q_upper - q_lower
        tangent_at_star = base_tangent + q_softmax0 * delta
        log_one_minus_lower = q_upper - tangent_at_star + log_n_minus_one
        with np.errstate(over="ignore"):
            result[row] = 1.0 - np.exp(log_one_minus_lower)

    return result


def stage2_lower_rough_box(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> np.ndarray:
    """Cheap safe closed-form lower bound from Notes_17Aug.

    The quantity 1-L_i is a sum of exponentials. Each exponential term is
    maximized independently over the P box, then the maxima are summed. These
    termwise maxima can correspond to incompatible P vertices, which makes
    the bound conservative but cheap.
    """

    lower_s, upper_s, p0_s = _stage2_scaled_box_inputs(
        p_lower, p_upper, p0, d_key
    )
    length = lower_s.shape[0]
    result = np.empty_like(lower_s)

    for row in range(length):
        s0 = softmax_rows(p0_s[row].reshape(1, -1))[0]
        lse0 = _logsumexp_1d(p0_s[row])
        log_prefactor = -lse0 + float(s0 @ p0_s[row])

        # A_j = max_P [P_j - s0^T P]
        #     = u_j - s0^T l - s0_j (u_j-l_j).
        s_dot_lower = float(s0 @ lower_s[row])
        a = (
            upper_s[row]
            - s_dot_lower
            - s0 * (upper_s[row] - lower_s[row])
        )
        leave_one_out_lse = _leave_one_out_logsumexp(a)
        with np.errstate(over="ignore"):
            result[row] = 1.0 - np.exp(log_prefactor + leave_one_out_lse)

    return result


def stage2_upper_box(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
    *,
    return_vertices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Exact maximum of the original convex U_i over a fixed P box.

    U_i is increasing in P_i and decreasing in every P_j, j != i. Therefore
    the maximizing vertex is known without gradient-magnitude bounds:
    P_i = upper_i and P_j = lower_j for j != i.
    """

    lower_s, upper_s, p0_s = _stage2_scaled_box_inputs(
        p_lower, p_upper, p0, d_key
    )
    length = lower_s.shape[0]
    result = np.empty_like(lower_s)
    arg_vertices_scaled = np.empty((length, length, length), dtype=np.float64)

    for row in range(length):
        s0 = softmax_rows(p0_s[row].reshape(1, -1))[0]
        lse0 = _logsumexp_1d(p0_s[row])
        log_prefactor = -lse0 + float(s0 @ p0_s[row])
        base_dot = float(s0 @ lower_s[row])
        delta = upper_s[row] - lower_s[row]
        log_u = log_prefactor + upper_s[row] - (base_dot + s0 * delta)
        with np.errstate(over="ignore"):
            result[row] = np.exp(log_u)
        for column in range(length):
            vertex = lower_s[row].copy()
            vertex[column] = upper_s[row, column]
            arg_vertices_scaled[row, column] = vertex

    if return_vertices:
        return result, arg_vertices_scaled * math.sqrt(d_key)
    return result


def stage2_box_bounds_all(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
    *,
    maximum_exact_vertices: int = 1_048_576,
) -> Stage2BoxBounds:
    """Convenience evaluator for all Stage-2 benchmark methods."""

    true_lower, true_upper = stage2_true_softmax_box_bounds(
        p_lower, p_upper, d_key
    )
    p_exact_lower = stage2_lower_exact_box(
        p_lower,
        p_upper,
        p0,
        d_key,
        maximum_vertices=maximum_exact_vertices,
    )
    common_lower = stage2_lower_common_box(p_lower, p_upper, p0, d_key)
    q_box_lower = stage2_lower_q_box(p_lower, p_upper, p0, d_key)
    rough_lower = stage2_lower_rough_box(p_lower, p_upper, p0, d_key)
    u_upper = stage2_upper_box(p_lower, p_upper, p0, d_key)
    return Stage2BoxBounds(
        true_lower=true_lower,
        true_upper=true_upper,
        p_exact_lower=p_exact_lower,
        common_lower=common_lower,
        q_box_lower=q_box_lower,
        rough_lower=rough_lower,
        u_upper=u_upper,
    )


def stage3_bounds(
    r: np.ndarray,
    v: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Entrywise bounds for S_ij = R_i.T V_j, equations (17)-(18)."""

    length, d_key = v.shape
    lower = np.empty((length, d_key), dtype=np.float64)
    upper = np.empty((length, d_key), dtype=np.float64)
    for i in range(length):
        for a in range(d_key):
            x, y = r[i], v[:, a]
            x0, y0 = r0[i], v0[:, a]
            plus, minus = x + y, x - y
            plus0, minus0 = x0 + y0, x0 - y0
            p1_linear = 0.25 * plus0 @ plus0 + 0.5 * plus0 @ (plus - plus0)
            p2 = 0.25 * minus @ minus
            p1 = 0.25 * plus @ plus
            p2_linear = 0.25 * minus0 @ minus0 + 0.5 * minus0 @ (minus - minus0)
            lower[i, a] = p1_linear - p2
            upper[i, a] = p1 - p2_linear
    return lower, upper


# ---------------------------------------------------------------------------
# Additional helpers used by the full box-controller pipeline.
# The names below intentionally coexist with the Stage-2 benchmark API above
# so that the previous benchmark scripts keep working after this file replaces
# the repository version.
# ---------------------------------------------------------------------------


def stage2_true_box_extrema(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    d_key: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Alias used by the controller diagnostics."""

    return stage2_true_softmax_box_bounds(p_lower, p_upper, d_key)


def stage2_rough_box_bounds(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Rough lower and direct critical-P upper bounds together."""

    return (
        stage2_lower_rough_box(p_lower, p_upper, p0, d_key),
        stage2_upper_box(p_lower, p_upper, p0, d_key),
    )


def stage2_q_box_lower(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> np.ndarray:
    """Alias for the exact fixed-box June-20 Q-box diagnostic."""

    return stage2_lower_q_box(p_lower, p_upper, p0, d_key)


def stage2_q_box_linearized_bounds(
    p_lower: np.ndarray,
    p_upper: np.ndarray,
    p0: np.ndarray,
    d_key: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Controller-compatible linearized-Q-box bound on a fixed P-box.

    This is NOT the exact June-20 Q-box diagnostic. The lower Q endpoint
    log-sum-exp is replaced by its affine tangent at P^k so that the resulting
    lower-R expression remains convex-compatible when box endpoints are
    optimization variables. It is safe but can be more conservative.
    """

    lower_s, upper_s, p0_s = _stage2_scaled_box_inputs(
        p_lower, p_upper, p0, d_key
    )
    length = lower_s.shape[0]
    if length < 2:
        raise ValueError("Q-box transform requires sequence length >= 2.")

    lower = np.empty_like(lower_s)
    upper = stage2_upper_box(p_lower, p_upper, p0, d_key)
    log_n_minus_one = math.log(length - 1.0)

    for row in range(length):
        q0 = _leave_one_out_logsumexp(p0_s[row])
        q_upper = _leave_one_out_logsumexp(upper_s[row])
        q_lower_hat = np.empty(length, dtype=np.float64)
        for j in range(length):
            indices = np.arange(length) != j
            x0_sub = p0_s[row, indices]
            grad = softmax_rows(x0_sub.reshape(1, -1))[0]
            q_lower_hat[j] = q0[j] + grad @ (
                lower_s[row, indices] - x0_sub
            )

        q_softmax0 = softmax_rows(q0.reshape(1, -1))[0]
        q_lse0 = _logsumexp_1d(q0)
        constant = float(q_softmax0 @ q0 - q_lse0 + log_n_minus_one)
        for target in range(length):
            q_vertex = q_lower_hat.copy()
            q_vertex[target] = q_upper[target]
            exponent = (
                q_vertex[target]
                - float(q_softmax0 @ q_vertex)
                + constant
            )
            with np.errstate(over="ignore"):
                lower[row, target] = 1.0 - math.exp(exponent)
    return lower, upper


def stage3_monotone_box_bounds_numeric(
    r_lower: np.ndarray,
    r_upper: np.ndarray,
    v: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    r_domain_lower: np.ndarray,
    r_domain_upper: np.ndarray,
    v_domain_lower: np.ndarray,
    v_domain_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical counterpart of the Stage-3 monotone box construction.

    ``r_domain_*`` and ``v_domain_*`` are fixed domains used to form constant
    Jacobian bounds. Only monotonicity in R is needed here because V is the
    exact affine function of the current decision U, not an independent box.
    """

    r_lower = np.asarray(r_lower, dtype=np.float64)
    r_upper = np.asarray(r_upper, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    rd_lo = np.asarray(r_domain_lower, dtype=np.float64)
    vd_lo = np.asarray(v_domain_lower, dtype=np.float64)
    vd_hi = np.asarray(v_domain_upper, dtype=np.float64)
    length, d_key = v.shape
    lower = np.empty((length, d_key), dtype=np.float64)
    upper = np.empty((length, d_key), dtype=np.float64)

    for i in range(length):
        for a in range(d_key):
            x_u = r_upper[i]
            y = v[:, a]
            x0 = r0[i]
            y0 = v0[:, a]

            gamma = 0.5 * (x0 - rd_lo[i] + y0 + vd_hi[:, a])
            gamma_pos = np.maximum(gamma, 0.0)
            gamma_neg = np.minimum(gamma, 0.0)
            plus = x_u + y
            minus = x_u - y
            plus0 = x0 + y0
            l_at_upper = float(
                0.25 * (plus0 @ plus0)
                + 0.5 * plus0 @ (plus - plus0)
                - 0.25 * (minus @ minus)
            )
            lower[i, a] = (
                l_at_upper
                - gamma @ (x_u - x0)
                + gamma_pos @ (r_lower[i] - x0)
                + gamma_neg @ (r_upper[i] - x0)
            )

            delta = 0.5 * (rd_lo[i] + vd_lo[:, a] - x0 + y0)
            delta_pos = np.maximum(delta, 0.0)
            delta_neg = np.minimum(delta, 0.0)
            minus0 = x0 - y0
            u_at_upper = float(
                0.25 * (plus @ plus)
                - 0.25 * (minus0 @ minus0)
                - 0.5 * minus0 @ (minus - minus0)
            )
            upper[i, a] = (
                u_at_upper
                - delta @ (x_u - x0)
                + delta_pos @ (r_upper[i] - x0)
                + delta_neg @ (r_lower[i] - x0)
            )
    return lower, upper
