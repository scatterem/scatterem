"""Device selection for tensor consumers, independent of domain objects."""

import torch


def resolve_device(
    device: str | torch.device | None, *tensors: torch.Tensor | None
) -> str | torch.device:
    """Return an explicit device unchanged, else the first tensor's device or CPU.

    Only inspect placement: do not move tensors, allocate storage, or probe CUDA.
    Callers order tensors by precedence and retain their own validation rules.
    """
    if device is not None:
        return device
    return next((t.device for t in tensors if t is not None), torch.device("cpu"))


def resolve_speed_device(device: str | torch.device | None) -> str | torch.device:
    """Return an explicit device unchanged, else CUDA if available, else CPU.

    Only for steps whose result does not depend on the device (a lossless codec
    encodes the same bytes anywhere), where the CUDA probe is a speed choice and
    not a placement rule. Everything else follows :func:`resolve_device`.
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
