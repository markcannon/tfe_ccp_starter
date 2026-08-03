from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-8) -> "StandardScaler":
        array = np.asarray(values, dtype=np.float64)
        mean = array.mean(axis=0)
        scale = array.std(axis=0)
        scale = np.where(scale < eps, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.mean


@dataclass(frozen=True)
class ScalerBundle:
    u: StandardScaler
    y: StandardScaler

    @classmethod
    def fit(cls, u: np.ndarray, y: np.ndarray) -> "ScalerBundle":
        return cls(StandardScaler.fit(u), StandardScaler.fit(y))


class PhysicalPredictor(nn.Module):
    """Differentiable physical-unit wrapper around a normalized TFE."""

    def __init__(self, model: nn.Module, scalers: ScalerBundle) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("u_mean", torch.as_tensor(scalers.u.mean, dtype=torch.float32))
        self.register_buffer("u_scale", torch.as_tensor(scalers.u.scale, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(scalers.y.mean, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(scalers.y.scale, dtype=torch.float32))

    def forward(self, past_u: Tensor, past_y: Tensor, future_u: Tensor) -> Tensor:
        past_u_n = (past_u - self.u_mean) / self.u_scale
        future_u_n = (future_u - self.u_mean) / self.u_scale
        past_y_n = (past_y - self.y_mean) / self.y_scale
        y_n = self.model(past_u_n, past_y_n, future_u_n)
        return y_n * self.y_scale + self.y_mean
