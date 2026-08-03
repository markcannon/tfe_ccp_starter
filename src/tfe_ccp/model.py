from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TFEConfig:
    past_window: int
    horizon: int
    n_inputs: int = 1
    n_outputs: int = 1
    d_key: int = 4

    @property
    def sequence_length(self) -> int:
        return self.past_window + self.horizon

    @property
    def output_size(self) -> int:
        return self.horizon * self.n_outputs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequenceLinearTransformation(nn.Module):
    """Linear transformation in corrected equation (5).

    For X with shape ``(..., sequence_length, d_key)``, this module computes

        h = X @ W_L1 + b_L1,

    with ``W_L1`` in R**d_key and a separate bias for every sequence position,
    ``b_L1`` in R**sequence_length.
    """

    def __init__(self, d_key: int, sequence_length: int) -> None:
        super().__init__()
        self.d_key = d_key
        self.sequence_length = sequence_length
        self.weight = nn.Parameter(torch.empty(d_key))
        self.bias = nn.Parameter(torch.empty(sequence_length))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.d_key)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-2:] != (self.sequence_length, self.d_key):
            raise ValueError(
                "Expected trailing dimensions "
                f"({self.sequence_length}, {self.d_key}), got {tuple(values.shape[-2:])}."
            )
        return torch.matmul(values, self.weight) + self.bias


class TransformerEncoderPredictor(nn.Module):
    """Single-head TFE matching the corrected paper.

    Shapes:
        past_u:   (batch, w, n_inputs)
        past_y:   (batch, w, n_outputs)
        future_u: (batch, p, n_inputs)
        output:   (batch, p, n_outputs)

    Inputs and outputs are expected to be normalized outside this module.
    ``p`` in :meth:`intermediates` is the unscaled matrix P = Q K.T from
    equation (7); ``logits`` is P / sqrt(d_key), the argument of softmax.
    """

    def __init__(self, config: TFEConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_key
        length = config.sequence_length

        self.past_embedding = nn.Linear(config.n_inputs + config.n_outputs, d)
        self.future_embedding = nn.Linear(config.n_inputs, d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.readout1 = SequenceLinearTransformation(d, length)
        self.readout2 = nn.Linear(length, config.output_size)

    def intermediates(
        self, past_u: Tensor, past_y: Tensor, future_u: Tensor
    ) -> dict[str, Tensor]:
        if past_u.ndim != 3 or past_y.ndim != 3 or future_u.ndim != 3:
            raise ValueError("All model inputs must be rank-3 batched tensors.")
        if past_u.shape[1:] != (self.config.past_window, self.config.n_inputs):
            raise ValueError(f"Unexpected past_u shape {tuple(past_u.shape)}")
        if past_y.shape[1:] != (self.config.past_window, self.config.n_outputs):
            raise ValueError(f"Unexpected past_y shape {tuple(past_y.shape)}")
        if future_u.shape[1:] != (self.config.horizon, self.config.n_inputs):
            raise ValueError(f"Unexpected future_u shape {tuple(future_u.shape)}")

        past_xy = torch.cat((past_u, past_y), dim=-1)
        z_past = self.past_embedding(past_xy)
        z_future = self.future_embedding(future_u)
        z1 = torch.cat((z_past, z_future), dim=1)

        q = self.q_proj(z1)
        k = self.k_proj(z1)
        v = self.v_proj(z1)
        p = torch.matmul(q, k.transpose(-1, -2))
        logits = p / math.sqrt(self.config.d_key)
        r = torch.softmax(logits, dim=-1)
        s = torch.matmul(r, v)

        h = self.readout1(z1 + s)
        y_flat = self.readout2(h)
        y = y_flat.reshape(-1, self.config.horizon, self.config.n_outputs)
        return {
            "z1": z1,
            "q": q,
            "k": k,
            "v": v,
            "p": p,
            "logits": logits,
            "r": r,
            "s": s,
            "h": h,
            "y": y,
        }

    def forward(self, past_u: Tensor, past_y: Tensor, future_u: Tensor) -> Tensor:
        return self.intermediates(past_u, past_y, future_u)["y"]
