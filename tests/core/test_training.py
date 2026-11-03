import pytest
import torch

from runitback.core.training import autocast, cpu_state_dict, resolve_device, to_device


def test_resolve_device_auto_and_explicit():
    dev = resolve_device("auto")
    assert dev.type in {"cuda", "mps", "cpu"}
    assert resolve_device(None) == dev
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_resolve_device_rejects_missing_cuda():
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_to_device_moves_nested_batches():
    batch = {"feats": {"a": torch.zeros(2, 3, dtype=torch.long)}, "valid": torch.ones(2, 3), "n": 3}
    out = to_device(batch, torch.device("cpu"))
    assert out["feats"]["a"].dtype == torch.long and out["valid"].shape == (2, 3) and out["n"] == 3


def test_autocast_is_noop_on_cpu_and_state_dict_is_cpu():
    with autocast(torch.device("cpu"), amp=True):
        y = torch.nn.Linear(4, 4)(torch.randn(2, 4))
    assert y.dtype == torch.float32
    assert all(v.device.type == "cpu" for v in cpu_state_dict(torch.nn.Linear(4, 4)).values())
