"""Sport-agnostic training helpers."""
from __future__ import annotations

import numpy as np


def batches(items: list[dict], batch_size: int, rng: np.random.Generator, shuffle: bool = True,
            length_key: str = "type") -> list[list[int]]:
    """Index batches over per-game items, bucketed by length to reduce padding."""
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    chunks = [order[i: i + batch_size * 20] for i in range(0, len(order), batch_size * 20)]
    out = []
    for c in chunks:
        c = sorted(c, key=lambda i: len(items[i][length_key]))
        out += [c[i: i + batch_size] for i in range(0, len(c), batch_size)]
    if shuffle:
        rng.shuffle(out)
    return out


def resolve_device(device: str | None = "auto") -> "torch.device":
    """Pick the training device. "auto" (or None) uses CUDA if available, then Apple MPS, then CPU;
    anything else ("cpu", "cuda", "cuda:1", "mps") is taken as given and must be available."""
    import torch
    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    if dev.type == "mps" and not (getattr(torch.backends, "mps", None) is not None
                                  and torch.backends.mps.is_available()):
        raise RuntimeError(f"device {device!r} requested but MPS is not available")
    return dev


def to_device(obj, device: "torch.device"):
    """Move every tensor in a (nested) dict/list/tuple batch to `device`."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=device.type == "cuda")
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def autocast(device: "torch.device", amp: bool):
    """bfloat16 autocast on CUDA GPUs that support it when `amp` is set; a no-op context otherwise.
    bf16 keeps fp32's exponent range, so no gradient scaler is needed and the -1e9 logit masks are safe."""
    import contextlib

    import torch
    if amp and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cpu_state_dict(model: "torch.nn.Module") -> dict:
    """The model's weights on the CPU, so checkpoints load the same on any machine."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
