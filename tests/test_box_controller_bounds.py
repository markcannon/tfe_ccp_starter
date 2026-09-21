import numpy as np

from tfe_ccp.dc_bounds import (
    softmax_rows,
    stage1_bounds,
    stage2_q_box_lower,
    stage2_q_box_linearized_bounds,
    stage2_rough_box_bounds,
    stage2_true_box_extrema,
    stage3_monotone_box_bounds_numeric,
)
from tfe_ccp.optimization.box_ccp import _stage1_approximation_domain


def test_stage2_rough_and_qbox_are_valid_on_fixed_boxes() -> None:
    rng = np.random.default_rng(123)
    n, d_key = 4, 3
    for _ in range(30):
        p0 = rng.normal(scale=0.5, size=(n, n))
        half = rng.uniform(1e-4, 0.25, size=(n, n))
        centre = p0 + rng.normal(scale=0.05, size=(n, n))
        p_lower = centre - half
        p_upper = centre + half

        true_min, true_max = stage2_true_box_extrema(p_lower, p_upper, d_key)
        rough_lower, rough_upper = stage2_rough_box_bounds(p_lower, p_upper, p0, d_key)
        q_lower = stage2_q_box_lower(p_lower, p_upper, p0, d_key)
        qlin_lower, qlin_upper = stage2_q_box_linearized_bounds(p_lower, p_upper, p0, d_key)

        assert np.all(rough_lower <= true_min + 1e-10)
        assert np.all(q_lower <= true_min + 1e-10)
        assert np.all(qlin_lower <= true_min + 1e-10)
        assert np.all(rough_upper >= true_max - 1e-10)
        assert np.all(qlin_upper >= true_max - 1e-10)


def test_stage2_bounds_are_tight_when_box_collapses_at_linearization_point() -> None:
    rng = np.random.default_rng(7)
    n, d_key = 4, 2
    p0 = rng.normal(size=(n, n))
    expected = softmax_rows(p0 / np.sqrt(d_key))
    rough_lower, rough_upper = stage2_rough_box_bounds(p0, p0, p0, d_key)
    q_lower = stage2_q_box_lower(p0, p0, p0, d_key)
    qlin_lower, qlin_upper = stage2_q_box_linearized_bounds(p0, p0, p0, d_key)
    np.testing.assert_allclose(rough_lower, expected, atol=1e-12)
    np.testing.assert_allclose(rough_upper, expected, atol=1e-12)
    np.testing.assert_allclose(q_lower, expected, atol=1e-12)
    np.testing.assert_allclose(qlin_lower, expected, atol=1e-12)
    np.testing.assert_allclose(qlin_upper, expected, atol=1e-12)


def test_stage1_approximation_domain_contains_stage1_dc_expressions() -> None:
    rng = np.random.default_rng(9)
    n, d = 4, 3
    q0 = rng.normal(size=(n, d))
    k0 = rng.normal(size=(n, d))
    q_lower = q0 - rng.uniform(0.1, 0.6, size=(n, d))
    q_upper = q0 + rng.uniform(0.1, 0.6, size=(n, d))
    k_lower = k0 - rng.uniform(0.1, 0.6, size=(n, d))
    k_upper = k0 + rng.uniform(0.1, 0.6, size=(n, d))
    dom_lower, dom_upper = _stage1_approximation_domain(
        q_lower, q_upper, k_lower, k_upper, q0, k0
    )
    for _ in range(100):
        q = rng.uniform(q_lower, q_upper)
        k = rng.uniform(k_lower, k_upper)
        lower, upper = stage1_bounds(q, k, q0, k0)
        assert np.all(lower >= dom_lower - 1e-10)
        assert np.all(upper <= dom_upper + 1e-10)


def test_stage3_monotone_box_bounds_contain_product_samples() -> None:
    rng = np.random.default_rng(11)
    n, d = 4, 2
    r0 = rng.uniform(0.05, 0.6, size=(n, n))
    v0 = rng.normal(scale=0.5, size=(n, d))
    r_domain_lower = np.full((n, n), -0.4)
    r_domain_upper = np.full((n, n), 1.4)
    v_domain_lower = np.full((n, d), -2.0)
    v_domain_upper = np.full((n, d), 2.0)

    r_lower = rng.uniform(-0.2, 0.1, size=(n, n))
    r_upper = rng.uniform(0.7, 1.2, size=(n, n))
    v = rng.uniform(v_domain_lower, v_domain_upper)
    lower, upper = stage3_monotone_box_bounds_numeric(
        r_lower,
        r_upper,
        v,
        r0,
        v0,
        r_domain_lower,
        r_domain_upper,
        v_domain_lower,
        v_domain_upper,
    )

    for _ in range(500):
        r = rng.uniform(r_lower, r_upper)
        true_s = r @ v
        assert np.all(lower <= true_s + 1e-10)
        assert np.all(true_s <= upper + 1e-10)


def test_stage3_monotone_box_is_tight_at_collapsed_linearization_box() -> None:
    rng = np.random.default_rng(19)
    n, d = 4, 3
    r0 = rng.uniform(0.0, 1.0, size=(n, n))
    v0 = rng.normal(size=(n, d))
    r_domain_lower = np.full((n, n), -1.0)
    r_domain_upper = np.full((n, n), 2.0)
    v_domain_lower = v0 - 3.0
    v_domain_upper = v0 + 3.0
    lower, upper = stage3_monotone_box_bounds_numeric(
        r0,
        r0,
        v0,
        r0,
        v0,
        r_domain_lower,
        r_domain_upper,
        v_domain_lower,
        v_domain_upper,
    )
    expected = r0 @ v0
    np.testing.assert_allclose(lower, expected, atol=1e-12)
    np.testing.assert_allclose(upper, expected, atol=1e-12)


def test_qkv_global_intervals_contain_forward_values() -> None:
    from tfe_ccp.export import TFEParameters
    from tfe_ccp.model import TFEConfig
    from tfe_ccp.optimization import MPCBounds
    from tfe_ccp.optimization.box_ccp import _qkv_global_intervals
    from tfe_ccp.dc_bounds import forward_from_parameters

    rng = np.random.default_rng(31)
    cfg = TFEConfig(past_window=2, horizon=3, n_inputs=2, n_outputs=1, d_key=3)
    n, d = cfg.sequence_length, cfg.d_key
    params = TFEParameters(
        config=cfg,
        w_em1=rng.normal(size=(d, cfg.n_inputs + cfg.n_outputs)),
        b_em1=rng.normal(size=d),
        w_em2=rng.normal(size=(d, cfg.n_inputs)),
        b_em2=rng.normal(size=d),
        w_q=rng.normal(size=(d, d)),
        b_q=rng.normal(size=d),
        w_k=rng.normal(size=(d, d)),
        b_k=rng.normal(size=d),
        w_v=rng.normal(size=(d, d)),
        b_v=rng.normal(size=d),
        w_l1=rng.normal(size=d),
        b_l1=rng.normal(size=n),
        w_l2=rng.normal(size=(cfg.output_size, n)),
        b_l2=rng.normal(size=cfg.output_size),
        u_mean=np.array([0.2, -0.3]),
        u_scale=np.array([1.4, 0.7]),
        y_mean=np.array([0.1]),
        y_scale=np.array([1.8]),
    )
    bounds = MPCBounds(
        u_min=np.array([-2.0, -1.0]),
        u_max=np.array([1.5, 2.5]),
        du_min=np.array([-1.0, -1.0]),
        du_max=np.array([1.0, 1.0]),
    )
    past_u = rng.normal(size=(cfg.past_window, cfg.n_inputs))
    past_y = rng.normal(size=(cfg.past_window, cfg.n_outputs))
    initial_u = np.zeros((cfg.horizon, cfg.n_inputs))
    current = forward_from_parameters(params, past_u, past_y, initial_u)
    ql, qu, kl, ku, vl, vu = _qkv_global_intervals(params, bounds, current)

    for _ in range(200):
        u = rng.uniform(bounds.u_min, bounds.u_max, size=(cfg.horizon, cfg.n_inputs))
        fwd = forward_from_parameters(params, past_u, past_y, u)
        assert np.all(ql <= fwd.q + 1e-10)
        assert np.all(fwd.q <= qu + 1e-10)
        assert np.all(kl <= fwd.k + 1e-10)
        assert np.all(fwd.k <= ku + 1e-10)
        assert np.all(vl <= fwd.v + 1e-10)
        assert np.all(fwd.v <= vu + 1e-10)
