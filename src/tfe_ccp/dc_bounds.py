from __future__ import annotations

from dataclasses import dataclass
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


def softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


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
