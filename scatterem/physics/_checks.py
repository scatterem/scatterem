"""Checks of host scalar inputs for the physics modules.

This private module was written for scatterem. It follows no external
literature; it applies the error rules of the scatterem physics layer.
"""

from __future__ import annotations

import math
import numbers

import torch


def _host_real(name: str, value: object) -> float:
    """Return a host real scalar as a float, or raise ``TypeError``."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise TypeError(f"{name} must be a real number, got bool")
    if isinstance(value, torch.Tensor) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real Python or numpy scalar, got {type(value).__name__}"
        )
    return float(value)


def _checked(name: str, value: object, *, zero_ok: bool) -> float:
    """Return a finite real scalar that is positive (or zero if allowed)."""
    x = _host_real(name, value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    if x < 0.0 or (x == 0.0 and not zero_ok):
        bound = "zero or larger" if zero_ok else "larger than zero"
        raise ValueError(f"{name} must be {bound}, got {x}")
    return x
