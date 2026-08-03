from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoupledTanks:
    """Coupled-tank benchmark from the paper, with tank 2 measured."""

    dt: float = 1.0
    gravity: float = 981.0
    pump_gain: float = 3.3
    area: float = 15.2
    orifice1: float = 0.135
    orifice2: float = 0.140
    process_noise_std: float = 0.0
    seed: int = 0

    n_inputs: int = 1
    n_outputs: int = 1
    n_states: int = 2

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state = np.array([10.0, 10.0], dtype=np.float64)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        self._state = (
            np.array([10.0, 10.0], dtype=np.float64)
            if state is None
            else np.asarray(state, dtype=np.float64).copy()
        )
        return np.array([self._state[1]])

    def step(self, control: np.ndarray) -> np.ndarray:
        u = float(np.asarray(control).reshape(-1)[0])
        x1, x2 = np.maximum(self._state, 0.0)
        q12 = self.orifice1 * np.sqrt(2.0 * self.gravity * x1)
        q2 = self.orifice2 * np.sqrt(2.0 * self.gravity * x2)
        next_state = np.array(
            [
                x1 + self.dt / self.area * (self.pump_gain * u - q12),
                x2 + self.dt / self.area * (q12 - q2),
            ],
            dtype=np.float64,
        )
        if self.process_noise_std > 0.0:
            next_state += self._rng.normal(0.0, self.process_noise_std, size=2)
        self._state = np.maximum(next_state, 0.0)
        return np.array([self._state[1]])
