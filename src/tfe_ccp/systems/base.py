from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DynamicalSystem(Protocol):
    n_inputs: int
    n_outputs: int
    n_states: int

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """Reset internal state and return the current measured output."""

    def step(self, control: np.ndarray) -> np.ndarray:
        """Apply one control sample and return the resulting measured output."""


@dataclass(frozen=True)
class Trajectory:
    u: np.ndarray  # (T, n_inputs)
    y: np.ndarray  # (T, n_outputs), paired with u at the same sample index


def simulate(system: DynamicalSystem, controls: np.ndarray, state: np.ndarray | None = None) -> Trajectory:
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != system.n_inputs:
        raise ValueError("controls must have shape (T, system.n_inputs)")
    system.reset(state)
    outputs = np.empty((controls.shape[0], system.n_outputs), dtype=np.float64)
    for index, control in enumerate(controls):
        outputs[index] = system.step(control)
    return Trajectory(u=controls, y=outputs)


def piecewise_constant_controls(
    length: int,
    n_inputs: int,
    low: float | np.ndarray,
    high: float | np.ndarray,
    hold: int = 8,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low_array = np.broadcast_to(np.asarray(low, dtype=np.float64), (n_inputs,))
    high_array = np.broadcast_to(np.asarray(high, dtype=np.float64), (n_inputs,))
    number_of_blocks = int(np.ceil(length / hold))
    block_values = rng.uniform(low_array, high_array, size=(number_of_blocks, n_inputs))
    return np.repeat(block_values, hold, axis=0)[:length]
