# Project guide: reproducing corrected TFE-based MPC

## 1. What to reproduce first

Treat the work as four separately testable layers:

1. **Data and scaling**: trajectories, sliding windows, and fixed normalization statistics.
2. **TFE prediction**: a deliberately small, explicit single-head network with no layer normalization.
3. **DC-bound verification**: numerical tests showing `lower <= true mapping <= upper` and equality at the linearization point for `P`, `R`, and `S`.
4. **Online optimization**: first the small exact conic reference, then the fast PIQP approximation.

Do not begin with a closed-loop experiment. First make the exported NumPy model match PyTorch to numerical tolerance, then make every DC bound pass randomized tests.

## 2. Tool choices

### Training and model definition

Use PyTorch. Avoid `nn.TransformerEncoder` and `nn.MultiheadAttention` initially because their fused parameters, masks, dropout options, and implementation details make the paper-to-code mapping less transparent. The explicit layers in `model.py` expose exactly the matrices needed by the optimizer.

The corrected first readout is not an ordinary `nn.Linear(d_key, 1)` because its bias has one component per sequence position. `SequenceLinearTransformation` implements this directly.

### Exact Algorithm 1 baseline

Use CVXPY as the modeling layer and Clarabel as the first open-source solver. The softmax bounds are exponential-cone representable, while the quadratic inequalities and robust tracking epigraph are second-order-cone representable. SCS is a useful fallback for debugging but will generally be less accurate.

### PIQP experiments

Use PIQP only after deciding which QP approximation you intend to study. The included backend uses a local affine model of the complete trained TFE and a quadratic tracking/control/trust-region objective. This gives a standard constrained QP and makes direct use of PIQP's Python interface.

### Engineering tools

Use NumPy/SciPy for matrix assembly, pytest for bound and shape tests, and dataclasses for immutable model/controller configuration. Save both a PyTorch checkpoint and an `.npz` export.

## 3. Corrected matrix mapping

| Code | Paper | Shape in SISO paper |
|---|---|---|
| `w_em1`, `b_em1` | `W_EM1`, `b_EM1` | `(d_key, 2)`, `(d_key,)` |
| `w_em2`, `b_em2` | `W_EM2`, `b_EM2` | `(d_key, 1)`, `(d_key,)` |
| `w_q`, `b_q` | `W_Q`, `b_Q` | `(d_key, d_key)`, `(d_key,)` |
| `w_k`, `b_k` | `W_K`, `b_K` | `(d_key, d_key)`, `(d_key,)` |
| `w_v`, `b_v` | `W_V`, `b_V` | `(d_key, d_key)`, `(d_key,)` |
| `w_l1` | `W_L1` | `(d_key,)` |
| `b_l1` | `b_L1` | `(w+p,)` |
| `w_l2`, `b_l2` | `W_L2`, `b_L2` | `(p, w+p)`, `(p,)` |

PyTorch `nn.Linear(in_features, out_features)` stores a matrix with shape `(out_features, in_features)` and evaluates `x @ weight.T + bias`. The custom first readout instead evaluates

```text
h = values @ readout1.weight + readout1.bias
```

so that its tensors match corrected equation (5) directly.

## 4. Corrected forward map

For sequence length `L = w + p`:

```text
Z1 = G(U)                         shape (L, d_key)
Q  = Z1 W_Q.T + b_Q              shape (L, d_key)
K  = Z1 W_K.T + b_K              shape (L, d_key)
V  = Z1 W_V.T + b_V              shape (L, d_key)
P  = Q K.T                       shape (L, L)
R  = softmax(P / sqrt(d_key))    shape (L, L)
S  = R V                         shape (L, d_key)
h  = (Z1 + S) W_L1 + b_L1       shape (L,)
Y  = W_L2 h + b_L2               shape (p,) in SISO
```

The code exposes both `p` and `logits` in model intermediates. This avoids accidentally applying the attention scale in stage 1.

## 5. Exact conic CCP flow

At iteration `k`:

1. Evaluate the exported model at `U_k` to obtain `Z1_k`, `Q_k`, `K_k`, unscaled `P_k`, `R_k`, `V_k`, and `S_k`.
2. Make physical future controls `U` the primary decision variable and normalize them affinely inside the optimization model.
3. Construct `Z1(U)`, `Q(U)`, `K(U)`, and `V(U)` as affine CVXPY expressions.
4. Construct entrywise stage-1 bounds on unscaled `P = Q K.T`.
5. Build `C(alpha1, beta1)`, enumerate its vertices, divide every vertex by `sqrt(d_key)` inside the stage-2 softmax bounds, and construct `C(alpha2, beta2)`.
6. Enumerate the stage-2 simplex vertices and impose the corrected entrywise stage-3 bounds for `S = R V`.
7. Build `C(alpha3, beta3)`, enumerate its vertices, and impose a worst-case tracking epigraph at each output vertex using corrected equation (5).
8. Add input and input-rate constraints, solve, and update `U_k`.

This is intentionally a small-scale reference. With `L = w + p`, stage 2 already uses `L^2 + 1` simplex vertices. Use tiny dimensions to validate mathematics before attempting performance engineering.

## 6. PIQP QP flow

For the current future input vector `U_k`, compute the physical-output Jacobian `J_k` by automatic differentiation:

```text
Y(U) approximately Y(U_k) + J_k (U - U_k).
```

The included code minimizes

```text
||Y_affine(U) - Y_ref||_Q^2
+ lambda ||Delta U||^2
+ rho ||U-U_k||^2
```

subject to input bounds, rate bounds, and a box trust region. This is converted explicitly to PIQP's

```text
0.5 U.T P U + c.T U
```

form. The difference matrix uses time-major flattening, so a MIMO sequence is `[u_0, u_1, ..., u_{p-1}]` with each `u_j` containing all input channels.

The PIQP path automatically uses the corrected predictor architecture because it differentiates the complete PyTorch model. No QP-assembly formula changed in version 0.2.

## 7. Validation milestones

1. `test_model_export.py` passes: corrected tensor dimensions are present and exported and PyTorch predictions agree.
2. `test_dc_bounds.py` passes over many random seeds and perturbation magnitudes.
3. Train a small model with `w=2, p=2, d_key=2` and solve one exact conic subproblem.
4. Compare exact CCP and PIQP-SQP from the same initial sequence on that small model.
5. Add closed-loop simulation and record prediction error, tracking RMSE, solve time, iteration count, and constraint violation.
6. Only then increase to the paper's `w=8, p=20` architecture.

## 8. Checkpoints from starter version 0.1

The loader upgrades old tensor shapes and preserves their predictions. Nevertheless, retrain models for new experiments because the corrected `b_L1` has `w+p` independently trainable components, whereas the old checkpoint contains only one shared scalar.

## 9. Remaining implementation questions

The corrected paper resolves the previous dimensional ambiguities. Questions that remain relevant for a full reproduction include:

- How are the simplex vertex constraints evaluated efficiently at the reported horizon `p=20`?
- What exact normalization, training-data excitation, optimizer settings, and stopping criteria were used for the reported BFR values?
- Were additional input/output constraints used in the closed-loop experiments?
- How was the predictor recursively composed for MPC horizons `N > p` inside the online CCP formulation?
