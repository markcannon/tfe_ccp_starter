from __future__ import annotations

import argparse
from pathlib import Path

from tfe_ccp.data import make_windows
from tfe_ccp.export import extract_parameters, save_parameters
from tfe_ccp.model import TFEConfig
from tfe_ccp.systems import CoupledTanks, NonlinearSpringMass, piecewise_constant_controls, simulate
from tfe_ccp.training import TrainingConfig, save_checkpoint, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("coupled_tanks", "nonlinear_spring"), default="nonlinear_spring")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--length", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/small_model.pt"))
    args = parser.parse_args()

    if args.system == "coupled_tanks":
        system = CoupledTanks()
        controls = piecewise_constant_controls(args.length, 1, 0.0, 10.0, hold=10, seed=1)
    else:
        system = NonlinearSpringMass()
        controls = piecewise_constant_controls(args.length, 1, -1.5, 1.5, hold=12, seed=1)

    config = TFEConfig(past_window=2, horizon=2, n_inputs=system.n_inputs, n_outputs=system.n_outputs, d_key=2)
    trajectory = simulate(system, controls)
    windows = make_windows(trajectory, config.past_window, config.horizon)
    model, scalers, history = train_model(
        config,
        windows,
        TrainingConfig(epochs=args.epochs, batch_size=128, patience=12),
    )
    save_checkpoint(args.output, model, scalers, history)
    save_parameters(args.output.with_suffix(".npz"), extract_parameters(model, scalers))
    print(f"Saved checkpoint to {args.output}")
    print(f"Final validation MSE: {history['validation'][-1]:.6g}")


if __name__ == "__main__":
    main()
