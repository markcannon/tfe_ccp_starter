import numpy as np

from tfe_ccp.dc_bounds import softmax_rows, stage1_bounds, stage2_bounds, stage3_bounds


def test_dc_bounds_contain_true_values_and_are_tight() -> None:
    rng = np.random.default_rng(4)
    length, d_key = 4, 3
    q0, k0 = rng.normal(size=(length, d_key)), rng.normal(size=(length, d_key))
    q = q0 + 0.2 * rng.normal(size=(length, d_key))
    k = k0 + 0.2 * rng.normal(size=(length, d_key))

    # Corrected paper: P = Q K.T is unscaled in stage 1.
    lower_p, upper_p = stage1_bounds(q, k, q0, k0)
    true_p = q @ k.T
    assert np.all(lower_p <= true_p + 1e-10)
    assert np.all(true_p <= upper_p + 1e-10)
    tight_l, tight_u = stage1_bounds(q0, k0, q0, k0)
    np.testing.assert_allclose(tight_l, q0 @ k0.T, atol=1e-12)
    np.testing.assert_allclose(tight_u, q0 @ k0.T, atol=1e-12)

    # Corrected paper: softmax applies the 1/sqrt(d_key) scaling in stage 2.
    p0 = q0 @ k0.T
    p = true_p
    lower_r, upper_r = stage2_bounds(p, p0, d_key)
    true_r = softmax_rows(p / np.sqrt(d_key))
    assert np.all(lower_r <= true_r + 1e-10)
    assert np.all(true_r <= upper_r + 1e-10)
    tight_l, tight_u = stage2_bounds(p0, p0, d_key)
    expected_r0 = softmax_rows(p0 / np.sqrt(d_key))
    np.testing.assert_allclose(tight_l, expected_r0, atol=1e-12)
    np.testing.assert_allclose(tight_u, expected_r0, atol=1e-12)

    r0, r = expected_r0, true_r
    v0 = rng.normal(size=(length, d_key))
    v = v0 + 0.2 * rng.normal(size=(length, d_key))
    lower_s, upper_s = stage3_bounds(r, v, r0, v0)
    true_s = r @ v
    assert np.all(lower_s <= true_s + 1e-10)
    assert np.all(true_s <= upper_s + 1e-10)
    tight_l, tight_u = stage3_bounds(r0, v0, r0, v0)
    np.testing.assert_allclose(tight_l, r0 @ v0, atol=1e-12)
    np.testing.assert_allclose(tight_u, r0 @ v0, atol=1e-12)
