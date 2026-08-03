import numpy as np

from tfe_ccp.optimization.common import MPCBounds
from tfe_ccp.optimization.piqp_sqp import PiqpSQPConfig, assemble_local_qp


def test_local_qp_is_symmetric_psd_and_has_expected_constraints() -> None:
    horizon, n_inputs = 3, 1
    jacobian = np.array([[1.0, 0.2, 0.0], [0.3, 1.0, 0.2], [0.0, 0.4, 1.0]])
    qp = assemble_local_qp(
        y_at_linearization=np.zeros(3),
        jacobian=jacobian,
        u_linearization=np.zeros((horizon, n_inputs)),
        y_reference=np.ones(3),
        previous_u=np.array([0.1]),
        horizon=horizon,
        n_inputs=n_inputs,
        bounds=MPCBounds.constant(1, -2.0, 2.0, -0.25, 0.25),
        config=PiqpSQPConfig(trust_radius=0.5),
    )
    np.testing.assert_allclose(qp.p, qp.p.T, atol=1e-12)
    assert np.linalg.eigvalsh(qp.p).min() > 0.0
    assert qp.g.shape == (3, 3)
    np.testing.assert_allclose(qp.x_l, -0.5)
    np.testing.assert_allclose(qp.x_u, 0.5)
    np.testing.assert_allclose(qp.h_l[0], -0.15)
    np.testing.assert_allclose(qp.h_u[0], 0.35)
