import numpy as np

from tfe_ccp.data import make_windows
from tfe_ccp.systems import NonlinearSpringMass, piecewise_constant_controls, simulate


def test_generic_system_and_windows() -> None:
    system = NonlinearSpringMass()
    controls = piecewise_constant_controls(30, 1, -1.0, 1.0, hold=3, seed=2)
    trajectory = simulate(system, controls)
    windows = make_windows(trajectory, past_window=4, horizon=5)
    assert windows.past_u.shape == (22, 4, 1)
    assert windows.future_u.shape == (22, 5, 1)
    assert np.isfinite(windows.target_y).all()
