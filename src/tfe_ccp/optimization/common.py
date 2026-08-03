from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MPCBounds:
    u_min: np.ndarray
    u_max: np.ndarray
    du_min: np.ndarray
    du_max: np.ndarray

    @classmethod
    def constant(
        cls,
        n_inputs: int,
        u_min: float,
        u_max: float,
        du_min: float,
        du_max: float,
    ) -> "MPCBounds":
        return cls(
            np.full(n_inputs, u_min, dtype=np.float64),
            np.full(n_inputs, u_max, dtype=np.float64),
            np.full(n_inputs, du_min, dtype=np.float64),
            np.full(n_inputs, du_max, dtype=np.float64),
        )


def difference_matrix(horizon: int, n_inputs: int) -> np.ndarray:
    time_difference = np.eye(horizon)
    if horizon > 1:
        time_difference[1:, :-1] -= np.eye(horizon - 1)
    return np.kron(time_difference, np.eye(n_inputs))


def delta_offset(previous_u: np.ndarray, horizon: int) -> np.ndarray:
    previous_u = np.asarray(previous_u, dtype=np.float64).reshape(-1)
    offset = np.zeros(horizon * previous_u.size, dtype=np.float64)
    offset[: previous_u.size] = previous_u
    return offset
