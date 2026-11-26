"""Build cached tensors that are never inference tensors.

A cache that is first filled under ``torch.inference_mode`` keeps inference
tensors, and every later autograd call that saves one for backward fails
("Inference tensors cannot be saved for backward"). A cache builds its value
inside :func:`normal_tensors` instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

__all__ = ["normal_tensors"]


@contextmanager
def normal_tensors() -> Iterator[None]:
    """Run the block outside inference mode, in the caller's grad mode.

    ``torch.inference_mode(False)`` alone also turns grad mode on, so a value
    built from a tensor that requires grad would get an autograd graph under
    ``torch.no_grad``. This context sets the caller's grad mode again inside.
    Under ``inference_mode`` grad mode is off inside, also when an inner
    ``torch.enable_grad`` turned it on: inference mode records no graph.

    Yields
    ------
    None
        Tensors that the block creates are normal tensors.
    """
    # read before inference_mode(False) turns grad mode on
    grad = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(grad):
        yield
