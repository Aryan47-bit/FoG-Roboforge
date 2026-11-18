import pytest

torch = pytest.importorskip("torch")
from src.models.tcn import TCN
from src.models.cnn import ConvPredictor
from src.data.datamodule import WindowDataset
from src.features.windowing import make_windows
from src.data.base_dataset import Recording
from pathlib import Path
import numpy as np


def test_tcn_causality_and_gradient():
    torch.manual_seed(42)
    model = TCN(3, channels=(4, 4))
    x = torch.randn(2, 50, 3)
    changed = x.clone()
    changed[:, 30:] += 10
    torch.testing.assert_close(model.timestep_logits(x)[:, :30], model.timestep_logits(changed)[:, :30])
    output = model(x)
    assert output.shape == (2,)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert TCN(3).receptive_field == 509


def test_lazy_dataset_matches_window_labels_and_values():
    t = np.arange(3000)/100
    record = Recording("fixture", "1", "s", "r", Path("fixture"), 100, t,
                       np.sin(t)[:, None].astype("float32"),
                       ((t >= 15) & (t < 17)).astype("int8"), ("x",), ("g",))
    lazy = WindowDataset([record])
    eager = make_windows(record, window_seconds=5)
    assert len(lazy) == len(eager.targets)
    for index in [0, len(lazy)//2, len(lazy)-1]:
        x, y = lazy[index]
        np.testing.assert_array_equal(x.numpy(), eager.signals[index])
        assert y.item() == eager.targets[index]


def test_cnn_forward_backward_and_checkpoint(tmp_path):
    model = ConvPredictor(9)
    x = torch.randn(2, 500, 9)
    y = model(x)
    assert y.shape == (2,)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / "cnn.pt"
    torch.save(model.state_dict(), path)
    restored = ConvPredictor(9)
    restored.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(x), restored(x))
