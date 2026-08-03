# Changelog

## 0.2.0 - corrected-paper alignment

### Changed

- Implemented corrected equation (5) with `W_L1` as a vector of length `d_key` and `b_L1` as a vector of length `w + p`.
- Kept `P = Q K.T` unscaled in stage 1 and moved `1/sqrt(d_key)` scaling into stage 2.
- Updated NumPy parameter evaluation, parameter export, and the exact CVXPY CCP formulation.
- Added `p` and `logits` as distinct model intermediates.
- Added automatic migration for version-0.1 checkpoints and `.npz` parameter files.
- Added command-line options for `past_window`, `horizon`, and `d_key` in `train_demo.py`.
- Added `src` to pytest's configured Python path.

### Fixed

- Included the final valid sliding window in `make_windows`.
- Improved the dimension guard for the exact simplex-enumeration demo.

### Validation

- Added tests for corrected readout dimensions, unscaled/scaled attention variables, DC-bound tightness, and old-checkpoint migration.
