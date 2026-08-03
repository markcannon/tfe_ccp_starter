# Consistency audit against the corrected paper

This note records the review of starter version 0.1 against the corrected six-page manuscript and the changes made in version 0.2.

## Summary

Two material inconsistencies required code changes:

1. The first linear-transformation bias in corrected equation (5) is a vector in `R^(w+p)`, not a scalar shared by all sequence positions.
2. Corrected stages 1 and 2 distinguish the unscaled matrix `P = Q K.T` from the softmax argument `P / sqrt(d_key)`.

The stage-3 code was already using the corrected entrywise interpretation. The softmax lower-bound code was already summing over `j != i`.

## Equation-by-equation audit

### Equations (2)-(4): embedding and self-attention

The existing embedding, query, key, and value calculations were dimensionally consistent:

```text
Z1_past   = X_p @ W_EM1.T + b_EM1
Z1_future = U   @ W_EM2.T + b_EM2
Q = Z1 @ W_Q.T + b_Q
K = Z1 @ W_K.T + b_K
V = Z1 @ W_V.T + b_V
```

No architectural change was needed here.

The intermediate representation now stores both:

```text
P      = Q @ K.T
logits = P / sqrt(d_key)
R      = softmax(logits)
Z2     = R @ V
```

This makes the exported variables agree with the corrected notation.

### Equation (5): linear transformation

The corrected paper specifies

```text
Y = W_L2 ((Z1 + Z2) W_L1 + b_L1) + b_L2
```

with

```text
W_L1 in R^d_key
b_L1 in R^(w+p)
W_L2 in R^(p x (w+p))
b_L2 in R^p
```

Starter version 0.1 used `nn.Linear(d_key, 1)`, whose bias is one scalar shared across sequence positions. Version 0.2 replaces it with `SequenceLinearTransformation`, which has:

```text
readout1.weight.shape == (d_key,)
readout1.bias.shape   == (w + p,)
```

The NumPy export and exact CCP output expression were changed accordingly.

For `n_outputs > 1`, the code retains the same hidden vector of length `w+p` and generalizes `W_L2` to shape `(p*n_outputs, w+p)`.

### Equations (7)-(12): stage-1 DC bounds

The corrected paper defines

```text
P = Q K.T
```

before attention scaling. Version 0.1 divided the stage-1 values and bounds by `sqrt(d_key)`. Version 0.2 removes that division. Therefore:

```text
stage1_bounds(...)
```

returns lower and upper bounds on unscaled `P`.

The optional `d_key` argument is temporarily accepted for source compatibility but is not used.

### Equations (13)-(16): stage-2 softmax bounds

The corrected paper applies softmax to

```text
P / sqrt(d_key).
```

Version 0.2 changes both NumPy and CVXPY functions so they accept the unscaled stage-1 matrix `P` and apply the scaling internally:

```text
stage2_bounds(P, P_k, d_key)
```

The stage-1 simplex therefore contains the same `P` used in corrected Algorithm 1.

The lower-bound summation was already implemented as `j != i`; no change was needed for that correction.

### Equations (17)-(18): stage-3 bounds

The corrected manuscript gives an entrywise DC decomposition of

```text
S[i, j] = R[i, :] @ V[:, j].
```

Version 0.1 already implemented this entrywise form in NumPy and CVXPY. The implementation was retained and its documentation updated.

### Algorithm 1 and simplex propagation

The code already vectorized each matrix when constructing

```text
C(alpha, beta) = {X: -alpha <= vec(X), 1.T vec(X) <= beta}.
```

The corrected distinction between `P` and `P/sqrt(d_key)` is now respected when propagating the first simplex into the softmax bounds.

The corrected increment indexing is already represented by

```text
Delta U[0] = U[0] - previous_u
Delta U[j] = U[j] - U[j-1],  j > 0.
```

The exact reference adds input and rate bounds as a practical MPC extension; these are not essential to the paper's displayed Algorithm 1.

## Independent correction found during the audit

`make_windows` omitted the final valid sliding window. The number of windows is now

```text
T - w - p + 1
```

rather than `T - w - p`.

## Checkpoint migration

Version-0.1 checkpoints remain loadable. The loader transforms

```text
readout1.weight: (1, d_key) -> (d_key,)
readout1.bias:   (1,)       -> repeat scalar to (w+p,)
```

This gives exactly the same predictions as the old architecture. New models should be retrained to learn the corrected position-dependent `b_L1`.

## Validation performed

The test suite checks that:

- PyTorch and exported NumPy predictions agree;
- `W_L1`, `b_L1`, and `W_L2` have corrected dimensions;
- `P = Q K.T` is unscaled and `logits = P/sqrt(d_key)`;
- all three DC bounds contain the true mapping and are tight at the linearization point;
- an old checkpoint is migrated without changing its predictions;
- generic benchmark-system windows include the final valid sample;
- PIQP QP assembly remains symmetric positive definite and dimensionally valid.
