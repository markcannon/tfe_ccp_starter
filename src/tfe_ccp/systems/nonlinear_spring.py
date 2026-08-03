from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NonlinearSpringMass:
    """Damped Duffing-type oscillator used as a second SISO benchmark."""

    dt: float = 0.1
    damping: float = 0.35
    stiffness: float = 0.8
    cubic_stiffness: float = 0.12

    n_inputs: int = 1
    n_outputs: int = 1
    n_states: int = 2

    def __post_init__(self) -> None:
        self._state = np.zeros(2, dtype=np.float64)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        self._state = np.zeros(2, dtype=np.float64) if state is None else np.asarray(state, dtype=np.float64).copy()
        return np.array([self._state[0]])

    def step(self, control: np.ndarray) -> np.ndarray:
        u = float(np.asarray(control).reshape(-1)[0])
        position, velocity = self._state
        acceleration = (
            u
            - self.damping * velocity
            - self.stiffness * position
            - self.cubic_stiffness * position**3
        )
        velocity_next = velocity + self.dt * acceleration
        position_next = position + self.dt * velocity_next
        self._state = np.array([position_next, velocity_next], dtype=np.float64)
        return np.array([position_next])
