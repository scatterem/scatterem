"""Quaternions: conversion to and from rotation matrices and axis-angle, the product.

Quaternions are ``(w, x, y, z)``, scalar first, with the Hamilton product.
``q`` and ``-q`` are the same rotation; :func:`from_matrix` returns ``w >= 0``.
Rotations are active and act on column vectors, so :func:`to_matrix` gives the
``R`` with ``q * v * conj(q) == R @ v``.
"""

from __future__ import annotations

import torch

__all__ = [
    "from_axis_angle",
    "from_matrix",
    "from_rotation_vector",
    "multiply",
    "to_matrix",
]

# The nine entries of the matrix, in row-major order, all have the shape
#   m_k = beta_k + alpha_k * (q[ia_k]*q[ib_k] + s_k * q[ic_k]*q[id_k])
# with alpha in {+-2}, s in {+-1}, and beta = 1 on the diagonal.  Spelling them
# that way lets ONE gather + ONE multiply + TWO addcmuls replace ~40 tiny
# elementwise launches, which is what dominates on CUDA at these batch sizes.
#
# It is bit-exact to the expanded form, by IEEE and not by luck:
#   * s_k and alpha_k are +-1 / +-2, so both products are EXACT and any fma
#     contraction of `a + b*s` / `beta + t*alpha` rounds exactly once, as the
#     expanded form does;
#   * beta_k is -0.0 off the diagonal, so `-0.0 + 2*t` reproduces `2*t` for
#     every input INCLUDING the signed zeros that `0.0 + (-0.0) = +0.0` would
#     otherwise flip.
# Keep those constants if you touch this: a sha256-gated benchmark depends on them,
# and its round() makes it blind to a one-ULP change here.
_Q2M_IABCD = (
    [2, 1, 1, 1, 1, 2, 1, 2, 1]  # ia
    + [3, 0, 0, 0, 3, 0, 0, 0, 2]  # ic
    + [2, 2, 3, 2, 1, 3, 3, 3, 1]  # ib
    + [3, 3, 2, 3, 3, 1, 2, 1, 2]  # id
)
_Q2M_S = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
_Q2M_ALPHA = [-2.0, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0, 2.0, -2.0]
_Q2M_BETA = [1.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0]
_Q2M_CACHE: dict = {}
_Q2M_CACHE_MAX = 8


def _q2m_consts(device, dtype):
    key = (device, dtype)
    c = _Q2M_CACHE.get(key)
    if c is None:
        if len(_Q2M_CACHE) >= _Q2M_CACHE_MAX:
            _Q2M_CACHE.clear()
        c = (
            torch.tensor(_Q2M_IABCD, dtype=torch.long, device=device),
            torch.tensor(_Q2M_S, dtype=dtype, device=device).unsqueeze(0),
            torch.tensor(_Q2M_ALPHA, dtype=dtype, device=device).unsqueeze(0),
            torch.tensor(_Q2M_BETA, dtype=dtype, device=device).unsqueeze(0),
        )
        _Q2M_CACHE[key] = c
    return c


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Parameters
    ----------
    q : torch.Tensor
        Quaternions ``(w, x, y, z)`` with shape ``(..., 4)``. Not required to
        be unit norm; normalized internally.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if q.is_cuda and q.is_floating_point():
        # CUDA only: the stacked form trades access regularity for ~6x fewer
        # dispatches, which only pays where a dispatch costs something.  Measured
        # 3.2-3.4x here and 0.94-1.12x (non-monotone) on CPU, so CPU keeps the
        # expanded body below.
        qf = q.reshape(-1, 4)
        i36, s, alpha, beta = _q2m_consts(q.device, q.dtype)
        g = qf[:, i36]
        p = g[:, :18] * g[:, 18:]
        t = torch.addcmul(p[:, :9], p[:, 9:], s)
        m = torch.addcmul(beta, t, alpha)
        return m.reshape(q.shape[:-1] + (3, 3))
    w, x, y, z = q.unbind(-1)
    m = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Build the quaternions of right-handed turns about axes.

    The quaternion is ``(cos(angle / 2), sin(angle / 2) * axis / |axis|)``.
    The function multiplies by the sine first and divides by the axis norm
    after, so a single turn about a coordinate axis has exact bits.

    Parameters
    ----------
    axis : torch.Tensor
        Rotation axes with shape ``(..., 3)``. They need not be unit vectors,
        but must not be zero.
    angle : torch.Tensor
        Rotation angles in radians with shape ``(...)``, broadcastable with
        the batch shape of `axis`.

    Returns
    -------
    torch.Tensor
        Unit quaternions ``(w, x, y, z)`` with shape ``(..., 4)``. ``w`` is
        negative for ``|angle| > pi``; the function does not fold the sign.

    Notes
    -----
    Three dtype rules apply:

    - The output dtype is the type promotion of `axis` and `angle`.
    - ``sin`` and ``cos`` of the half angle run in the dtype of `angle` (the
      default float dtype for an integer `angle`). A float32 `angle` with a
      float64 `axis` gives a float64 result with float32 accuracy.
    - An integer `axis` is refused: ``torch.linalg.vector_norm`` raises
      ``RuntimeError``.
    """
    half = angle / 2.0
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    vector = torch.sin(half)[..., None] * axis / norm
    w = torch.cos(half)[..., None].expand(vector.shape[:-1] + (1,))
    return torch.cat([w, vector], dim=-1)


def from_rotation_vector(w: torch.Tensor) -> torch.Tensor:
    """Build the unit quaternions of rotation vectors.

    The rotation is ``exp([w]_x)``: the angle is ``|w|`` in radians and the
    axis is ``w``. The quaternion is ``(cos(|w| / 2), sin(|w| / 2) w / |w|)``.
    The function writes ``sin(|w| / 2) / |w|`` as ``sinc(|w| / 2 pi) / 2``,
    so it is smooth at ``w = 0`` and has a finite gradient there. That is
    where every gradient refinement of an orientation starts. Unlike
    :func:`from_axis_angle`, it accepts the zero vector, which gives the
    identity ``(1, 0, 0, 0)``.

    Parameters
    ----------
    w : torch.Tensor
        Rotation vectors with shape ``(..., 3)``, in radians.

    Returns
    -------
    torch.Tensor
        Unit quaternions ``(w, x, y, z)`` with shape ``(..., 4)``, in the
        dtype of `w`. ``to_matrix`` of the result is the
        rotation matrix ``exp([w]_x)``.
    """
    angle = w.norm(dim=-1, keepdim=True)
    half = 0.5 * torch.sinc(angle / (2.0 * torch.pi))  # sin(angle/2) / angle
    return torch.cat([torch.cos(0.5 * angle), half * w], dim=-1)


def from_matrix(m: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions.

    Parameters
    ----------
    m : torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.

    Returns
    -------
    torch.Tensor
        Unit quaternions ``(w, x, y, z)`` with shape ``(..., 4)``, with
        ``w >= 0``.
    """
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    qw2 = (1 + m00 + m11 + m22).clamp_min(0.0)
    qx2 = (1 + m00 - m11 - m22).clamp_min(0.0)
    qy2 = (1 - m00 + m11 - m22).clamp_min(0.0)
    qz2 = (1 - m00 - m11 + m22).clamp_min(0.0)
    pivot = torch.stack([qw2, qx2, qy2, qz2], dim=-1).argmax(dim=-1)
    s_w = qw2.sqrt().clamp_min(1e-12) * 2
    s_x = qx2.sqrt().clamp_min(1e-12) * 2
    s_y = qy2.sqrt().clamp_min(1e-12) * 2
    s_z = qz2.sqrt().clamp_min(1e-12) * 2
    cand = torch.stack(
        [
            torch.stack(
                [s_w / 4, (m21 - m12) / s_w, (m02 - m20) / s_w, (m10 - m01) / s_w], -1
            ),
            torch.stack(
                [(m21 - m12) / s_x, s_x / 4, (m01 + m10) / s_x, (m02 + m20) / s_x], -1
            ),
            torch.stack(
                [(m02 - m20) / s_y, (m01 + m10) / s_y, s_y / 4, (m12 + m21) / s_y], -1
            ),
            torch.stack(
                [(m10 - m01) / s_z, (m02 + m20) / s_z, (m12 + m21) / s_z, s_z / 4], -1
            ),
        ],
        dim=1,
    )  # (B, 4 pivots, 4)
    q = cand[torch.arange(m.shape[0], device=m.device), pivot]
    q = q * torch.where(
        q[:, :1] < 0, -torch.ones_like(q[:, :1]), torch.ones_like(q[:, :1])
    )
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q.reshape(batch + (4,))


def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute the Hamilton product ``q1 ⊗ q2``.

    The result applies rotation ``q2`` followed by ``q1``, matching the
    composition ``R1 @ R2`` of the corresponding rotation matrices.

    Parameters
    ----------
    q1 : torch.Tensor
        Quaternions ``(w, x, y, z)`` with shape ``(..., 4)``.
    q2 : torch.Tensor
        Quaternions ``(w, x, y, z)`` with shape ``(..., 4)``, broadcastable
        with `q1`.

    Returns
    -------
    torch.Tensor
        Hamilton product quaternions with shape ``(..., 4)``.
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
