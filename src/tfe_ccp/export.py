from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .model import TFEConfig, TransformerEncoderPredictor
from .scaling import ScalerBundle, StandardScaler


CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class TFEParameters:
    config: TFEConfig
    w_em1: np.ndarray
    b_em1: np.ndarray
    w_em2: np.ndarray
    b_em2: np.ndarray
    w_q: np.ndarray
    b_q: np.ndarray
    w_k: np.ndarray
    b_k: np.ndarray
    w_v: np.ndarray
    b_v: np.ndarray
    w_l1: np.ndarray
    b_l1: np.ndarray
    w_l2: np.ndarray
    b_l2: np.ndarray
    u_mean: np.ndarray
    u_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=True)


def extract_parameters(model: TransformerEncoderPredictor, scalers: ScalerBundle) -> TFEParameters:
    """Extract matrices and vectors using the corrected paper dimensions.

    In particular, ``w_l1`` has shape ``(d_key,)`` and ``b_l1`` has shape
    ``(past_window + horizon,)``.
    """

    return TFEParameters(
        config=model.config,
        w_em1=_array(model.past_embedding.weight),
        b_em1=_array(model.past_embedding.bias),
        w_em2=_array(model.future_embedding.weight),
        b_em2=_array(model.future_embedding.bias),
        w_q=_array(model.q_proj.weight),
        b_q=_array(model.q_proj.bias),
        w_k=_array(model.k_proj.weight),
        b_k=_array(model.k_proj.bias),
        w_v=_array(model.v_proj.weight),
        b_v=_array(model.v_proj.bias),
        w_l1=_array(model.readout1.weight),
        b_l1=_array(model.readout1.bias),
        w_l2=_array(model.readout2.weight),
        b_l2=_array(model.readout2.bias),
        u_mean=np.asarray(scalers.u.mean, dtype=np.float64),
        u_scale=np.asarray(scalers.u.scale, dtype=np.float64),
        y_mean=np.asarray(scalers.y.mean, dtype=np.float64),
        y_scale=np.asarray(scalers.y.scale, dtype=np.float64),
    )


def save_parameters(path: str | Path, parameters: TFEParameters) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        field.name: getattr(parameters, field.name)
        for field in fields(parameters)
        if field.name != "config"
    }
    arrays["config_values"] = np.array(
        [
            parameters.config.past_window,
            parameters.config.horizon,
            parameters.config.n_inputs,
            parameters.config.n_outputs,
            parameters.config.d_key,
        ],
        dtype=np.int64,
    )
    arrays["format_version"] = np.array([CHECKPOINT_FORMAT_VERSION], dtype=np.int64)
    np.savez(path, **arrays)


def _upgrade_readout_arrays(
    w_l1: np.ndarray,
    b_l1: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Upgrade arrays exported by the first starter version.

    The old code used an ``nn.Linear(d_key, 1)`` readout, so W_L1 was stored as
    ``(1, d_key)`` and b_L1 as a scalar. Repeating that scalar over all sequence
    positions preserves the old model exactly while adopting the corrected
    equation-(5) representation.
    """

    weight = np.asarray(w_l1, dtype=np.float64)
    bias = np.asarray(b_l1, dtype=np.float64)
    if weight.ndim == 2 and weight.shape[0] == 1:
        weight = weight.reshape(-1)
    if weight.ndim != 1:
        raise ValueError(f"Expected w_l1 to be a vector, got shape {weight.shape}.")
    if bias.size == 1:
        bias = np.full(sequence_length, float(bias.reshape(-1)[0]), dtype=np.float64)
    else:
        bias = bias.reshape(-1)
    if bias.shape != (sequence_length,):
        raise ValueError(
            f"Expected b_l1 shape ({sequence_length},), got {bias.shape}."
        )
    return weight, bias


def load_parameters(path: str | Path) -> TFEParameters:
    data = np.load(Path(path), allow_pickle=False)
    w, p, n_u, n_y, d = [int(value) for value in data["config_values"]]
    config = TFEConfig(w, p, n_u, n_y, d)
    kwargs = {
        name: data[name]
        for name in data.files
        if name not in {"config_values", "format_version"}
    }
    kwargs["w_l1"], kwargs["b_l1"] = _upgrade_readout_arrays(
        kwargs["w_l1"], kwargs["b_l1"], config.sequence_length
    )
    return TFEParameters(config=config, **kwargs)


def _upgrade_state_dict(
    state_dict: dict[str, torch.Tensor], config: TFEConfig
) -> dict[str, torch.Tensor]:
    upgraded = {key: value for key, value in state_dict.items()}
    weight_key = "readout1.weight"
    bias_key = "readout1.bias"
    if weight_key in upgraded and upgraded[weight_key].ndim == 2:
        if upgraded[weight_key].shape[0] != 1:
            raise ValueError(
                f"Cannot upgrade {weight_key} with shape {tuple(upgraded[weight_key].shape)}."
            )
        upgraded[weight_key] = upgraded[weight_key].reshape(-1)
    if bias_key in upgraded and upgraded[bias_key].numel() == 1:
        upgraded[bias_key] = upgraded[bias_key].reshape(1).repeat(config.sequence_length)
    return upgraded


def load_checkpoint(path: str | Path) -> tuple[TransformerEncoderPredictor, ScalerBundle]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = TFEConfig(**checkpoint["model_config"])
    model = TransformerEncoderPredictor(config)
    state_dict = _upgrade_state_dict(checkpoint["state_dict"], config)
    model.load_state_dict(state_dict, strict=True)
    scalers = ScalerBundle(
        u=StandardScaler(np.asarray(checkpoint["u_mean"]), np.asarray(checkpoint["u_scale"])),
        y=StandardScaler(np.asarray(checkpoint["y_mean"]), np.asarray(checkpoint["y_scale"])),
    )
    return model.eval(), scalers
