import numpy as np
import pytest
import torch

from hybrid import SelectiveSSM, VascMambaHybrid, load_per_view_features


def test_density_ordering_preserves_modality_pairs():
    torch.manual_seed(0)
    paired = VascMambaHybrid(order_by_density=True).eval()
    manual = VascMambaHybrid(order_by_density=False).eval()
    manual.load_state_dict(paired.state_dict())

    bmode = torch.randn(3, 4, 512)
    ulm = torch.randn(3, 4, 512)
    density = torch.tensor([
        [0.1, 0.4, 0.2, 0.3],
        [0.8, 0.2, 0.5, 0.1],
        [0.3, 0.6, 0.4, 0.9],
    ])
    valid = torch.tensor([
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
    ], dtype=torch.bool)
    order = density.argsort(dim=1, descending=True)
    feat_order = order.unsqueeze(-1).expand_as(bmode)

    with torch.no_grad():
        actual = paired(bmode, ulm, density, valid)
        expected = manual(
            bmode.gather(1, feat_order),
            ulm.gather(1, feat_order),
            density.gather(1, order),
            valid.gather(1, order),
        )
    torch.testing.assert_close(actual, expected)


def test_all_ssm_parameters_receive_gradients():
    layer = SelectiveSSM(d_model=8, d_state=3, d_conv=2, expand=2)
    layer(torch.randn(2, 5, 8)).square().mean().backward()
    missing = [name for name, parameter in layer.named_parameters()
               if parameter.grad is None]
    assert missing == []


def test_rejects_expanded_session_means(tmp_path):
    rng = np.random.default_rng(0)
    bmode = np.repeat(rng.normal(size=(12, 1, 512)), 4, axis=1).astype("float32")
    ulm = np.repeat(rng.normal(size=(12, 1, 512)), 4, axis=1).astype("float32")
    density = np.repeat(rng.random(size=(12, 1)), 4, axis=1).astype("float32")
    path = tmp_path / "fake.npz"
    np.savez(path, X_bmode=bmode, X_ulm=ulm, density=density,
             y=np.asarray([0, 1] * 6))

    with pytest.raises(ValueError, match="expanded session means"):
        load_per_view_features(path)
