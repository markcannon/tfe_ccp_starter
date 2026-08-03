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
    parser.add_argument("--past-window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--d-key", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    args = parser.parse_args()

    if args.system == "coupled_tanks":
        system = CoupledTanks(seed=args.seed)
        controls = piecewise_constant_controls(
            args.length, 1, 0.0, 10.0, hold=10, seed=args.seed
        )
    else:
        system = NonlinearSpringMass()
        controls = piecewise_constant_controls(
            args.length, 1, -1.5, 1.5, hold=12, seed=args.seed
        )

    config = TFEConfig(
        past_window=args.past_window,
        horizon=args.horizon,
        n_inputs=system.n_inputs,
        n_outputs=system.n_outputs,
        d_key=args.d_key,
    )
    trajectory = simulate(system, controls)
    windows = make_windows(trajectory, config.past_window, config.horizon)
    model, scalers, history = train_model(
        config,
        windows,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
        ),
    )
    save_checkpoint(args.output, model, scalers, history)
    save_parameters(args.output.with_suffix(".npz"), extract_parameters(model, scalers))
    print(f"Saved checkpoint to {args.output}")
    print(f"Final validation MSE: {history['validation'][-1]:.6g}")


if __name__ == "__main__":
    main()
