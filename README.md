# TFE-CCP MPC starter

This repository is a research starter for the corrected version of **Successive Convex Optimization for Transformer Encoder Model Predictive Control**.

## Paper-consistency status

Version 0.2 aligns the model and the exact conic reference implementation with the corrected equations:

- `P = Q @ K.T` is kept **unscaled** in self-attention stage 1.
- The factor `1 / sqrt(d_key)` is applied when stage 2 evaluates `R = softmax(P / sqrt(d_key))`.
- The linear transformation implements corrected equation (5):

  ```text
  h = (Z1 + Z2) @ W_L1 + b_L1
  Y = W_L2 @ h + b_L2
  ```

  with `W_L1.shape == (d_key,)` and `b_L1.shape == (w + p,)`.
- Stage 3 uses the corrected entrywise products `S[i, j] = R[i, :] @ V[:, j]`.
- The softmax lower bound uses the corrected summation over `j != i`.
- The matrix simplexes are represented by vectorizing `P`, `R`, and `S`, as specified below Algorithm 1.

See `PAPER_CONSISTENCY.md` for the full audit and migration notes.

## Important solver point

The exact convex subproblem in Algorithm 1 is **not a quadratic program**:

- the softmax upper/lower bounds contain exponentials, so an exponential-cone solver is required;
- the stage-1 and stage-3 DC bounds contain convex quadratic inequalities;
- the worst-case squared tracking epigraph is a convex quadratic/SOC constraint.

PIQP solves QPs with a quadratic objective and linear equalities/inequalities. It therefore cannot solve the paper's exact subproblem without an additional approximation. This project keeps two backends separate:

1. `ExactConicCCP`: a direct, small-scale reference implementation of corrected Algorithm 1 using CVXPY and Clarabel (or SCS). This is the mathematical baseline.
2. `PiqpSequentialController`: a PIQP-compatible sequential QP using the exact trained TFE and its Jacobian at each iterate, input/rate constraints, and a trust region. This is useful for fast experiments, but it is not Algorithm 1 and does not inherit the paper's DC/simplex guarantees.

## Recommended setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[optimization,dev]"
python -m pytest
```

The pytest configuration also adds `src` to the test path, so tests can be collected from a clean checkout. An editable installation is still recommended for scripts, PyCharm, and notebooks.

Train and run the PIQP approximation:

```bash
python scripts/train_demo.py --system nonlinear_spring --epochs 80
python scripts/run_piqp_demo.py --checkpoint artifacts/model.pt
```

For the exact reference, first train a deliberately small model:

```bash
python scripts/train_demo.py --system nonlinear_spring --length 1200 --epochs 40 \
  --past-window 2 --horizon 2 --d-key 2 --output artifacts/small_model.pt
python scripts/run_exact_small.py --checkpoint artifacts/small_model.pt
```

The vertex representation of the propagated simplexes grows rapidly, so `run_exact_small.py` is intentionally restricted to small dimensions.

## Package layout

- `model.py`: explicit single-head TFE matching corrected equations (2)-(5), without layer normalization.
- `data.py`: generic trajectory windows and train/validation splitting.
- `systems/`: plant protocol plus coupled-tank and nonlinear spring benchmarks.
- `training.py`: PyTorch training and checkpointing.
- `export.py`: extraction of every TFE matrix/bias and scaling parameter to NumPy.
- `dc_bounds.py`: NumPy implementations of corrected stage-1, stage-2, and stage-3 bounds.
- `optimization/exact_ccp.py`: CVXPY/Clarabel reference formulation of corrected Algorithm 1.
- `optimization/piqp_sqp.py`: direct PIQP QP assembly and sequential-QP controller.
- `PAPER_CONSISTENCY.md`: equation-by-equation audit against the corrected manuscript.

## Existing version-0.1 checkpoints

`load_checkpoint` and `load_parameters` migrate the first starter version automatically:

- old `W_L1` arrays with shape `(1, d_key)` are squeezed to `(d_key,)`;
- the old shared scalar `b_L1` is repeated over all `w + p` positions.

This preserves the old predictor numerically. It does **not** add the extra degrees of freedom of the corrected architecture, so retraining is recommended for paper-reproduction experiments.

## Extending to another system

Implement the `DynamicalSystem` protocol in `systems/base.py`. The TFE supports arbitrary input and output dimensions. The corrected paper is SISO, while the code extends `W_L2` and `b_L2` to produce `p * n_outputs` values for MIMO benchmarks.

The exact CCP implementation is practical only for very small examples. The PIQP sequential-QP backend scales more naturally to larger horizons and MIMO systems.
