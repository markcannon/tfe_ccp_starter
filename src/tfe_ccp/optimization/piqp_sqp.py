from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .common import MPCBounds, delta_offset, difference_matrix


@dataclass(frozen=True)
class QuadraticProgram:
    p: np.ndarray
    c: np.ndarray
    g: np.ndarray | None
    h_l: np.ndarray | None
    h_u: np.ndarray | None
    x_l: np.ndarray
    x_u: np.ndarray


@dataclass(frozen=True)
class PiqpSQPConfig:
    tracking_weight: float = 1.0
    control_weight: float = 1.0
    proximal_weight: float = 1e-3
    trust_radius: float = 1.0
    max_iterations: int = 10
    tolerance: float = 1e-4


def assemble_local_qp(
    y_at_linearization: np.ndarray,
    jacobian: np.ndarray,
    u_linearization: np.ndarray,
    y_reference: np.ndarray,
    previous_u: np.ndarray,
    horizon: int,
    n_inputs: int,
    bounds: MPCBounds,
    config: PiqpSQPConfig,
) -> QuadraticProgram:
    """Assemble the PIQP-compatible local QP.

    The model is approximated by y(U) = y_k + J (U - U_k).
    """
    y_k = np.asarray(y_at_linearization, dtype=np.float64).reshape(-1)
    j = np.asarray(jacobian, dtype=np.float64)
    u_k = np.asarray(u_linearization, dtype=np.float64).reshape(-1)
    y_ref = np.asarray(y_reference, dtype=np.float64).reshape(-1)
    n_variables = horizon * n_inputs
    if u_k.size != n_variables or j.shape[1] != n_variables:
        raise ValueError("Incompatible local model dimensions.")

    q_weight = config.tracking_weight * np.eye(y_k.size)
    d = difference_matrix(horizon, n_inputs)
    offset = delta_offset(previous_u, horizon)
    affine_residual = y_k - j @ u_k - y_ref

    hessian_half = (
        j.T @ q_weight @ j
        + config.control_weight * (d.T @ d)
        + config.proximal_weight * np.eye(n_variables)
    )
    linear = 2.0 * (
        j.T @ q_weight @ affine_residual
        - config.control_weight * d.T @ offset
        - config.proximal_weight * u_k
    )
    p = 2.0 * hessian_half
    p = 0.5 * (p + p.T)

    u_min = np.tile(np.asarray(bounds.u_min, dtype=np.float64), horizon)
    u_max = np.tile(np.asarray(bounds.u_max, dtype=np.float64), horizon)
    x_l = np.maximum(u_min, u_k - config.trust_radius)
    x_u = np.minimum(u_max, u_k + config.trust_radius)

    du_min = np.tile(np.asarray(bounds.du_min, dtype=np.float64), horizon)
    du_max = np.tile(np.asarray(bounds.du_max, dtype=np.float64), horizon)
    h_l = du_min + offset
    h_u = du_max + offset
    return QuadraticProgram(p=p, c=linear, g=d, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)


def solve_with_piqp(qp: QuadraticProgram, verbose: bool = False) -> np.ndarray:
    try:
        import piqp
    except ImportError as exc:
        raise RuntimeError("PIQP is not installed. Run: pip install piqp") from exc

    solver = piqp.DenseSolver()
    solver.settings.verbose = verbose
    solver.settings.compute_timings = True
    solver.setup(qp.p, qp.c, None, None, qp.g, qp.h_l, qp.h_u, qp.x_l, qp.x_u)
    status = solver.solve()
    if int(status) != 1:
        raise RuntimeError(f"PIQP failed with status {status}")
    return np.asarray(solver.result.x, dtype=np.float64).copy()


class PiqpSequentialController:
    """Sequential-QP controller using the exact TFE Jacobian and PIQP.

    This is deliberately named SQP rather than CCP: it is a practical PIQP
    approximation, not the exact simplex-propagating Algorithm 1.
    """

    def __init__(
        self,
        predictor: torch.nn.Module,
        horizon: int,
        n_inputs: int,
        bounds: MPCBounds,
        config: PiqpSQPConfig = PiqpSQPConfig(),
        solver: Callable[[QuadraticProgram], np.ndarray] = solve_with_piqp,
    ) -> None:
        self.predictor = predictor.eval()
        self.horizon = horizon
        self.n_inputs = n_inputs
        self.bounds = bounds
        self.config = config
        self.solver = solver

    def _value_and_jacobian(
        self, past_u: Tensor, past_y: Tensor, future_u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        u_flat = torch.tensor(future_u.reshape(-1), dtype=torch.float32, requires_grad=True)

        def prediction(flat: Tensor) -> Tensor:
            sequence = flat.reshape(1, self.horizon, self.n_inputs)
            return self.predictor(past_u, past_y, sequence).reshape(-1)

        y = prediction(u_flat)
        jacobian = torch.autograd.functional.jacobian(prediction, u_flat, vectorize=True)
        return y.detach().cpu().numpy(), jacobian.detach().cpu().numpy()

    def solve(
        self,
        past_u: np.ndarray,
        past_y: np.ndarray,
        y_reference: np.ndarray,
        previous_u: np.ndarray,
        initial_u: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        past_u_tensor = torch.as_tensor(past_u, dtype=torch.float32).reshape(1, *past_u.shape)
        past_y_tensor = torch.as_tensor(past_y, dtype=torch.float32).reshape(1, *past_y.shape)
        u = np.asarray(initial_u, dtype=np.float64).reshape(self.horizon, self.n_inputs).copy()
        changes: list[float] = []

        for _ in range(self.config.max_iterations):
            y, jacobian = self._value_and_jacobian(past_u_tensor, past_y_tensor, u)
            qp = assemble_local_qp(
                y,
                jacobian,
                u,
                y_reference,
                previous_u,
                self.horizon,
                self.n_inputs,
                self.bounds,
                self.config,
            )
            u_next = self.solver(qp).reshape(self.horizon, self.n_inputs)
            change = float(np.linalg.norm(u_next - u))
            changes.append(change)
            u = u_next
            if change <= self.config.tolerance:
                break
        return u, changes
