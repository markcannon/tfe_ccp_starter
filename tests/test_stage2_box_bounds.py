import itertools

import numpy as np

from tfe_ccp.dc_bounds import (
    softmax_rows,
    stage2_bounds,
    stage2_lower_common_box,
    stage2_lower_exact_box,
    stage2_lower_q_box,
    stage2_lower_rough_box,
    stage2_true_softmax_box_bounds,
    stage2_upper_box,
)


def _enumerate_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n = lower.size
    bits = np.array(list(itertools.product((0.0, 1.0), repeat=n)))
    return lower + bits * (upper - lower)


def test_stage2_exact_box_matches_brute_force() -> None:
    rng = np.random.default_rng(11)
    n, d_key = 4, 3
    p0 = rng.normal(size=(n, n))
    p_lower = rng.normal(size=(n, n)) - 0.6
    p_upper = p_lower + rng.uniform(0.1, 1.2, size=(n, n))

    exact = stage2_lower_exact_box(p_lower, p_upper, p0, d_key)
    scale = np.sqrt(d_key)

    brute = np.empty_like(exact)
    for row in range(n):
        vertices = _enumerate_vertices(p_lower[row], p_upper[row])
        lower_values = []
        for vertex in vertices:
            p = p0.copy()
            p[row] = vertex
            lower, _ = stage2_bounds(p, p0, d_key)
            lower_values.append(lower[row])
        brute[row] = np.min(np.asarray(lower_values), axis=0)

    np.testing.assert_allclose(exact, brute, atol=1e-11)


def test_common_o_n_is_exact_for_common_scalar_box() -> None:
    rng = np.random.default_rng(12)
    n, d_key = 5, 2
    p0 = rng.normal(size=(n, n))
    p_lower = np.full((n, n), -0.7)
    p_upper = np.full((n, n), 0.9)

    exact = stage2_lower_exact_box(p_lower, p_upper, p0, d_key)
    common = stage2_lower_common_box(p_lower, p_upper, p0, d_key)
    np.testing.assert_allclose(common, exact, atol=1e-10)


def test_all_lower_methods_are_valid_against_true_softmax_range() -> None:
    rng = np.random.default_rng(13)
    n, d_key = 4, 3
    p0 = rng.normal(size=(n, n))
    p_lower = rng.normal(size=(n, n)) - 0.8
    p_upper = p_lower + rng.uniform(0.05, 1.5, size=(n, n))

    true_lower, true_upper = stage2_true_softmax_box_bounds(
        p_lower, p_upper, d_key
    )
    p_exact = stage2_lower_exact_box(p_lower, p_upper, p0, d_key)
    common = stage2_lower_common_box(p_lower, p_upper, p0, d_key)
    q_box = stage2_lower_q_box(p_lower, p_upper, p0, d_key)
    rough = stage2_lower_rough_box(p_lower, p_upper, p0, d_key)
    u_upper = stage2_upper_box(p_lower, p_upper, p0, d_key)

    assert np.all(p_exact <= true_lower + 1e-10)
    assert np.all(common <= true_lower + 1e-10)
    assert np.all(q_box <= true_lower + 1e-10)
    assert np.all(rough <= true_lower + 1e-10)
    assert np.all(u_upper >= true_upper - 1e-10)

    # Common-box and rough are relaxations of the original P-space lower
    # function, so they must not exceed the exact P-space minimum.
    assert np.all(common <= p_exact + 1e-10)
    assert np.all(rough <= p_exact + 1e-10)


def test_direct_upper_matches_brute_force_maximum_of_U() -> None:
    rng = np.random.default_rng(14)
    n, d_key = 4, 2
    p0 = rng.normal(size=(n, n))
    p_lower = rng.normal(size=(n, n)) - 0.5
    p_upper = p_lower + rng.uniform(0.1, 1.0, size=(n, n))

    direct = stage2_upper_box(p_lower, p_upper, p0, d_key)
    brute = np.empty_like(direct)

    for row in range(n):
        vertices = _enumerate_vertices(p_lower[row], p_upper[row])
        upper_values = []
        for vertex in vertices:
            p = p0.copy()
            p[row] = vertex
            _, upper = stage2_bounds(p, p0, d_key)
            upper_values.append(upper[row])
        brute[row] = np.max(np.asarray(upper_values), axis=0)

    np.testing.assert_allclose(direct, brute, atol=1e-11)


def test_true_softmax_box_bounds_match_vertex_monotonicity() -> None:
    rng = np.random.default_rng(15)
    n, d_key = 4, 3
    p_lower = rng.normal(size=(n, n)) - 0.5
    p_upper = p_lower + rng.uniform(0.1, 1.0, size=(n, n))

    true_lower, true_upper = stage2_true_softmax_box_bounds(
        p_lower, p_upper, d_key
    )
    scale = np.sqrt(d_key)

    brute_lower = np.empty_like(true_lower)
    brute_upper = np.empty_like(true_upper)
    for row in range(n):
        vertices = _enumerate_vertices(p_lower[row], p_upper[row])
        softmax_values = softmax_rows(vertices / scale)
        brute_lower[row] = np.min(softmax_values, axis=0)
        brute_upper[row] = np.max(softmax_values, axis=0)

    np.testing.assert_allclose(true_lower, brute_lower, atol=1e-12)
    np.testing.assert_allclose(true_upper, brute_upper, atol=1e-12)
