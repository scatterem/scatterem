"""Axis-angle rotations: one builder of rotation matrices.

A rotation by ``angle`` (radians, right-handed) about ``axis`` is built
through its quaternion, ``quaternion.to_matrix(quaternion.from_axis_angle(
axis, angle))``. The rotation is active and acts on column vectors
(``R @ v``). A caller that applies the matrix to row vectors (``r @ R``)
turns its points by ``-angle``.

NumPy callers convert at their boundary and name the dtype of both inputs,
for example ``to_matrix(torch.as_tensor(axis, dtype=torch.float64),
torch.as_tensor(theta, dtype=torch.float64)).numpy()``. Without ``dtype``,
torch makes a list of Python floats float32 and a list of Python ints int64,
which :func:`to_matrix` refuses.
"""

from __future__ import annotations

from torch import Tensor

from scatterem.rotation import quaternion

__all__ = ["to_matrix"]


def to_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    """Return the matrices of right-handed turns by `angle` about `axis`.

    Parameters
    ----------
    axis : Tensor
        Rotation axes with shape ``(..., 3)``. They need not be unit vectors,
        but must not be zero.
    angle : Tensor
        Rotation angles in radians with shape ``(...)``, broadcastable with
        the batch shape of `axis`.

    Returns
    -------
    Tensor
        Rotation matrices with shape ``(..., 3, 3)``. ``R @ v`` turns a
        column vector ``v`` counter-clockwise about `axis` when seen from its
        tip.

    Notes
    -----
    Three dtype rules apply, as in
    :func:`scatterem.rotation.quaternion.from_axis_angle`:

    - The output dtype is the type promotion of `axis` and `angle`.
    - ``sin`` and ``cos`` of the half angle run in the dtype of `angle` (the
      default float dtype for an integer `angle`). A float32 `angle` with a
      float64 `axis` gives a float64 matrix with float32 accuracy.
    - An integer `axis` is refused: ``torch.linalg.vector_norm`` raises
      ``RuntimeError``.

    The quaternion route agrees with Rodrigues' formula
    ``cos(angle) I + sin(angle) K + (1 - cos(angle)) u u^T`` to a few ulp,
    but not bit for bit.

    Examples
    --------
    >>> import math, torch
    >>> from scatterem.rotation.axis_angle import to_matrix
    >>> R = to_matrix(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    >>> torch.allclose(R @ torch.tensor([1.0, 0.0, 0.0]),
    ...                torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    True
    """
    return quaternion.to_matrix(quaternion.from_axis_angle(axis, angle))
