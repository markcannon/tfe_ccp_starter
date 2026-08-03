from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tfe_ccp.export import extract_parameters, load_checkpoint
from tfe_ccp.optimization import ExactCCPConfig, ExactConicCCP, MPCBounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    model, scalers = load_checkpoint(args.checkpoint)
    cfg = model.config
    largest_simplex = max(cfg.sequence_length**2 + 1, cfg.sequence_length * cfg.d_key + 1)
    if largest_simplex > 200:
        raise SystemExit(
            "Use a checkpoint trained with small w, p, and d_key for the exact "
            "vertex-enumeration demo."
        )

    solver = ExactConicCCP(
        extract_parameters(model, scalers),
        MPCBounds.constant(cfg.n_inputs, -2.0, 2.0, -0.5, 0.5),
        ExactCCPConfig(max_iterations=20, maximum_vertices=200),
    )
    past_u = np.zeros((cfg.past_window, cfg.n_inputs))
    past_y = np.zeros((cfg.past_window, cfg.n_outputs))
    reference = np.ones((cfg.horizon, cfg.n_outputs))
    initial = np.zeros((cfg.horizon, cfg.n_inputs))
    result = solver.solve(past_u, past_y, reference, np.zeros(cfg.n_inputs), initial)
    print("First optimized input:", result.future_u[0])
    print("Objectives:", result.objective_history)
    print("Steps:", result.step_history)


if __name__ == "__main__":
    main()
