from pathlib import Path

import numpy as np
import torch

from tfe_ccp.dc_bounds import forward_from_parameters
from tfe_ccp.export import extract_parameters, load_checkpoint
from tfe_ccp.model import TFEConfig, TransformerEncoderPredictor
from tfe_ccp.scaling import ScalerBundle, StandardScaler


def _scalers() -> ScalerBundle:
    return ScalerBundle(
        StandardScaler(np.array([0.2]), np.array([1.3])),
        StandardScaler(np.array([-0.4]), np.array([0.7])),
    )


def test_exported_forward_matches_torch_and_corrected_dimensions() -> None:
    torch.manual_seed(2)
    config = TFEConfig(past_window=3, horizon=2, n_inputs=1, n_outputs=1, d_key=2)
    model = TransformerEncoderPredictor(config).eval()
    scalers = _scalers()
    params = extract_parameters(model, scalers)

    assert params.w_l1.shape == (config.d_key,)
    assert params.b_l1.shape == (config.sequence_length,)
    assert params.w_l2.shape == (config.output_size, config.sequence_length)

    past_u = np.array([[0.1], [0.2], [0.3]])
    past_y = np.array([[1.0], [0.8], [0.6]])
    future_u = np.array([[0.4], [0.5]])

    with torch.no_grad():
        pu = torch.tensor(scalers.u.transform(past_u), dtype=torch.float32)[None]
        py = torch.tensor(scalers.y.transform(past_y), dtype=torch.float32)[None]
        fu = torch.tensor(scalers.u.transform(future_u), dtype=torch.float32)[None]
        torch_values = model.intermediates(pu, py, fu)
        expected_n = torch_values["y"].numpy()[0]
    expected = scalers.y.inverse_transform(expected_n)
    actual_values = forward_from_parameters(params, past_u, past_y, future_u)
    np.testing.assert_allclose(actual_values.y, expected, rtol=2e-6, atol=2e-6)

    # P is unscaled; logits carry the 1/sqrt(d_key) factor.
    np.testing.assert_allclose(
        actual_values.p,
        actual_values.q @ actual_values.k.T,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual_values.logits,
        actual_values.p / np.sqrt(config.d_key),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual_values.h,
        (actual_values.z1 + actual_values.s) @ params.w_l1 + params.b_l1,
        rtol=2e-12,
        atol=2e-12,
    )


def test_version_01_checkpoint_is_upgraded_without_changing_prediction(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = TFEConfig(past_window=2, horizon=2, d_key=3)
    model = TransformerEncoderPredictor(config).eval()
    scalers = _scalers()

    # Version 0.1 used one shared scalar b_L1. Make the corrected model use the
    # same scalar at every position, then serialize the old tensor shapes.
    with torch.no_grad():
        model.readout1.bias.fill_(0.125)
    old_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    old_state["readout1.weight"] = old_state["readout1.weight"].reshape(1, -1)
    old_state["readout1.bias"] = old_state["readout1.bias"][:1]

    checkpoint = tmp_path / "old_model.pt"
    torch.save(
        {
            "model_config": config.to_dict(),
            "state_dict": old_state,
            "u_mean": scalers.u.mean,
            "u_scale": scalers.u.scale,
            "y_mean": scalers.y.mean,
            "y_scale": scalers.y.scale,
        },
        checkpoint,
    )
    loaded, _ = load_checkpoint(checkpoint)
    assert loaded.readout1.weight.shape == (config.d_key,)
    assert loaded.readout1.bias.shape == (config.sequence_length,)
    np.testing.assert_allclose(
        loaded.readout1.bias.detach().numpy(),
        np.full(config.sequence_length, 0.125),
    )

    past_u = torch.zeros((1, config.past_window, config.n_inputs))
    past_y = torch.zeros((1, config.past_window, config.n_outputs))
    future_u = torch.zeros((1, config.horizon, config.n_inputs))
    with torch.no_grad():
        np.testing.assert_allclose(
            loaded(past_u, past_y, future_u).numpy(),
            model(past_u, past_y, future_u).numpy(),
            rtol=1e-7,
            atol=1e-7,
        )
