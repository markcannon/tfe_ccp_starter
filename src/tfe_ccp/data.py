from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .scaling import ScalerBundle
from .systems.base import Trajectory


@dataclass(frozen=True)
class WindowArrays:
    past_u: np.ndarray
    past_y: np.ndarray
    future_u: np.ndarray
    target_y: np.ndarray


def make_windows(trajectory: Trajectory, past_window: int, horizon: int) -> WindowArrays:
    """Create windows using the paper's paired sample convention.

    At index k, the past contains samples k-w+1 through k, and the future
    controls/targets contain k+1 through k+p.
    """
    u, y = trajectory.u, trajectory.y
    if len(u) != len(y):
        raise ValueError("Trajectory u and y lengths must match.")
    samples = len(u) - past_window - horizon + 1
    if samples <= 0:
        raise ValueError("Trajectory is too short for the selected windows.")

    past_u, past_y, future_u, target_y = [], [], [], []
    for start in range(samples):
        stop = start + past_window
        future_stop = stop + horizon
        past_u.append(u[start:stop])
        past_y.append(y[start:stop])
        future_u.append(u[stop:future_stop])
        target_y.append(y[stop:future_stop])
    return WindowArrays(
        past_u=np.asarray(past_u),
        past_y=np.asarray(past_y),
        future_u=np.asarray(future_u),
        target_y=np.asarray(target_y),
    )


class TFEDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, windows: WindowArrays, scalers: ScalerBundle) -> None:
        self.past_u = torch.as_tensor(scalers.u.transform(windows.past_u), dtype=torch.float32)
        self.past_y = torch.as_tensor(scalers.y.transform(windows.past_y), dtype=torch.float32)
        self.future_u = torch.as_tensor(scalers.u.transform(windows.future_u), dtype=torch.float32)
        self.target_y = torch.as_tensor(scalers.y.transform(windows.target_y), dtype=torch.float32)

    def __len__(self) -> int:
        return self.past_u.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.past_u[index], self.past_y[index], self.future_u[index], self.target_y[index]


def split_indices(number_of_samples: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(number_of_samples)
    split = int(round(number_of_samples * (1.0 - validation_fraction)))
    return indices[:split], indices[split:]
