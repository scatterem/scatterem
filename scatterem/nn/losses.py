"""Reconstruction loss terms in the amplitude domain.

These are opt-in, model-agnostic callables. They assume the *amplitude*
convention of the ptychography models (both arguments are diffraction
amplitudes, the square roots of the intensities), so they operate in the
amplitude domain directly.
"""

from __future__ import annotations

import torch
from torch import Tensor



def normalized_amplitude_rmse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute the normalized amplitude RMSE between two diffraction stacks.

    Both arguments are amplitudes (see the module docstring), so they are
    compared directly::

        sqrt(mean((pred - target)**2)) / mean(target)

    A 1e-12 epsilon guards the normalizer against an all-zero target. It is
    negligible for real amplitude data.

    Parameters
    ----------
    pred : Tensor
        Predicted diffraction amplitudes. Any shape, real floating point;
        broadcasting is not performed, the two shapes must match.
    target : Tensor
        Measured diffraction amplitudes, same shape, dtype and device as
        ``pred``.

    Returns
    -------
    Tensor
        Scalar (0-dim) loss on the device and dtype of the inputs.
        Differentiable with respect to ``pred``.
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    return mse.sqrt() / (target.mean() + 1e-12)


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Mean of ``values`` over the True pixels of ``valid`` (broadcast).

    Parameters
    ----------
    values : Tensor
        Real values, any shape.
    valid : Tensor
        Bool mask, broadcastable to ``values``.

    Returns
    -------
    Tensor
        Scalar: the sum over the valid pixels divided by their number (at
        least 1).
    """
    valid = valid.expand_as(values)
    total = torch.where(valid, values, torch.zeros_like(values)).sum()
    return total / valid.sum().clamp_min(1).to(values.dtype)


def masked_normalized_amplitude_rmse_loss(
    pred: Tensor, target: Tensor, valid: Tensor
) -> Tensor:
    """:func:`normalized_amplitude_rmse_loss` over the valid detector pixels only.

    ``sqrt(mse) / mean(target)``, both means over the True pixels of
    ``valid``. An all-True mask gives :func:`normalized_amplitude_rmse_loss`.
    A masked pixel has no value and no gradient. A batch with no valid pixel
    gives 0 with a zero gradient (``sqrt'(0)`` is infinite, so its mse is
    replaced before the root).

    Parameters
    ----------
    pred, target : Tensor
        Predicted and measured amplitudes, same shape.
    valid : Tensor
        Bool mask, broadcastable to ``pred``; True for a pixel in the loss.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    any_valid = valid.expand_as(target).any()
    mse = masked_mean((pred - target) ** 2, valid)
    mse = torch.where(any_valid, mse, torch.ones_like(mse))
    value = mse.sqrt() / (masked_mean(target, valid) + 1e-12)
    return torch.where(any_valid, value, torch.zeros_like(value))










__all__ = [
    "masked_mean",
    "masked_normalized_amplitude_rmse_loss",
    "normalized_amplitude_rmse_loss",
]
