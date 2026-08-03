from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import TFEDataset, WindowArrays, split_indices
from .model import TFEConfig, TransformerEncoderPredictor
from .scaling import ScalerBundle


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 15
    seed: int = 0


def train_model(
    model_config: TFEConfig,
    windows: WindowArrays,
    training_config: TrainingConfig,
    device: str = "cpu",
) -> tuple[TransformerEncoderPredictor, ScalerBundle, dict[str, list[float]]]:
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)

    scalers = ScalerBundle.fit(
        np.concatenate((windows.past_u.reshape(-1, model_config.n_inputs), windows.future_u.reshape(-1, model_config.n_inputs))),
        np.concatenate((windows.past_y.reshape(-1, model_config.n_outputs), windows.target_y.reshape(-1, model_config.n_outputs))),
    )
    dataset = TFEDataset(windows, scalers)
    train_idx, val_idx = split_indices(len(dataset), training_config.validation_fraction, training_config.seed)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=training_config.batch_size, shuffle=False)

    model = TransformerEncoderPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    loss_function = nn.MSELoss()
    history = {"train": [], "validation": []}
    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    stale_epochs = 0

    for _ in range(training_config.epochs):
        model.train()
        train_sum, train_count = 0.0, 0
        for past_u, past_y, future_u, target_y in train_loader:
            past_u, past_y, future_u, target_y = (
                past_u.to(device), past_y.to(device), future_u.to(device), target_y.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(past_u, past_y, future_u)
            loss = loss_function(prediction, target_y)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * past_u.shape[0]
            train_count += past_u.shape[0]

        model.eval()
        validation_sum, validation_count = 0.0, 0
        with torch.no_grad():
            for past_u, past_y, future_u, target_y in val_loader:
                past_u, past_y, future_u, target_y = (
                    past_u.to(device), past_y.to(device), future_u.to(device), target_y.to(device)
                )
                loss = loss_function(model(past_u, past_y, future_u), target_y)
                validation_sum += float(loss) * past_u.shape[0]
                validation_count += past_u.shape[0]

        train_loss = train_sum / max(train_count, 1)
        validation_loss = validation_sum / max(validation_count, 1)
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    model.load_state_dict(best_state)
    return model.cpu(), scalers, history


def save_checkpoint(
    path: str | Path,
    model: TransformerEncoderPredictor,
    scalers: ScalerBundle,
    history: dict[str, list[float]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format_version": 2,
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "u_mean": scalers.u.mean,
            "u_scale": scalers.u.scale,
            "y_mean": scalers.y.mean,
            "y_scale": scalers.y.scale,
            "history": history or {},
        },
        path,
    )
