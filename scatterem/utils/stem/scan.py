"""Raster scan position generation."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def advanced_raster_scan(
    ny: int = 10,
    nx: int = 10,
    fast_axis: int = 1,
    mirror: Sequence[int] = (1, 1),
    theta: float = 0,
    dy: float = 1,
    dx: float = 1,
    device: "str | torch.device | None" = None,
    dtype=torch.float32,
) -> Tensor:
    """Generate a raster scan.

    Parameters
    ----------
    ny, nx : int
        Number of steps in *y* (vertical) and *x* (horizontal) direction.
    fast_axis : int
        Which axis is the fast axis (1 for x, 0 for y).
    mirror : List[int]
        Mirror factors for x and y axes.
    theta : float
        Rotation angle in degrees.
    dy, dx : float
        Step size (grid spacing) in *y* and *x*.
    device : torch.device
        Device for the output tensor.
    dtype : torch.dtype, optional
        Dtype of the output tensor. Default is ``torch.float32``.

    Returns
    -------
    Tensor
        A (N,2)-tensor of positions.
    """
    device = torch.device("cpu") if device is None else device
    iiy, iix = np.indices((ny, nx), dtype=np.float32)

    if fast_axis != 1:
        iix, iiy = iiy, iix

    # Create positions directly without list comprehension
    positions = np.stack([dy * iiy.ravel(), dx * iix.ravel()], axis=1)

    # Center the positions
    center = positions.mean(axis=0)
    positions -= center

    # Apply mirroring
    positions[:, 0] *= mirror[0]
    positions[:, 1] *= mirror[1]

    # Apply rotation. Kept in numpy: the
    # float64 rotation of the float32 positions is rounded once to ``dtype`` (default
    # float32), so a torch cos/sin spelling would risk those bits.
    if theta != 0:
        theta_rad = np.radians(theta)
        R = np.array(
            [
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad), np.cos(theta_rad)],
            ]
        )
        positions = positions @ R

    # Translate to start from origin
    positions -= positions.min(axis=0)

    return torch.as_tensor(positions, device=device, dtype=dtype)
