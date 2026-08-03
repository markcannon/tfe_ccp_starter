from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tfe_ccp.export import load_checkpoint
from tfe_ccp.optimization import MPCBounds, PiqpSQPConfig, PiqpSequentialController
from tfe_ccp.scaling import PhysicalPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    args = parser.parse_args()

    model, scalers = load_checkpoint(args.checkpoint)
    physical_model = PhysicalPredictor(model, scalers)
    cfg = model.config
    bounds = MPCBounds.constant(cfg.n_inputs, -2.0, 2.0, -0.3, 0.3)
    controller = PiqpSequentialController(
        physical_model,
        cfg.horizon,
        cfg.n_inputs,
        bounds,
        PiqpSQPConfig(control_weight=0.5, trust_radius=0.5),
    )

    past_u = np.zeros((cfg.past_window, cfg.n_inputs))
    past_y = np.zeros((cfg.past_window, cfg.n_outputs))
    reference = np.ones((cfg.horizon, cfg.n_outputs))
    initial = np.zeros((cfg.horizon, cfg.n_inputs))
    solution, changes = controller.solve(past_u, past_y, reference, np.zeros(cfg.n_inputs), initial)
    print("First optimized input:", solution[0])
    print("SQP step norms:", changes)


if __name__ == "__main__":
    main()
