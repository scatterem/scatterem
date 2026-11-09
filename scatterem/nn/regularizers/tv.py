"""Dense total-variation losses and their analytic gradients.

:func:`total_variation_3d` is the differentiable loss: a mean over forward
differences, on the common interior for isotropic TV. The two ``*_subgradient``
functions are its exact gradients (a subgradient at the kinks of anisotropic
TV), evaluated analytically without autograd so they can be injected into
``.grad``. Inputs use voxel differences, without physical voxel-spacing
normalization. Real floating tensors retain their dtype and device.

"""

import torch
from torch import Tensor

__all__ = [
    "anisotropic_tv_subgradient",
    "isotropic_tv_subgradient",
    "mean_abs_forward_diff",
    "total_variation_2d",
    "total_variation_3d",
]


def mean_abs_forward_diff(vol: Tensor, dim: int) -> Tensor:
    """Mean absolute forward difference of `vol` along `dim`.

    Shared anisotropic-TV building block: ``mean(|vol[n+1] - vol[n]|)`` on
    axis `dim`. Used by :func:`total_variation_3d` (anisotropic).

    Parameters
    ----------
    vol : Tensor
        Real floating tensor of any shape.
    dim : int
        Axis along which to take the forward difference. The axis length must
        be at least 2 or the mean is NaN.

    Returns
    -------
    Tensor
        Scalar mean absolute difference on the dtype and device of `vol`.
    """
    return torch.diff(vol, dim=dim).abs().mean()


@torch.no_grad()
def anisotropic_tv_subgradient(volume: Tensor) -> Tensor:
    """Exact (sub)gradient of the mean anisotropic TV loss.

    The loss is ``sum_axes mean_n |v[n+1] - v[n]|`` over the forward differences
    of each axis: :func:`total_variation_3d` with ``isotropic=False`` on a
    ``[D, H, W]`` volume. Its derivative is the discrete
    divergence of ``sign(grad v)`` scaled by ``1 / numel(diff)`` of that axis:
    ``sign`` of each forward difference is subtracted at the left voxel and
    added at the right voxel. Away from ties this equals autograd of the loss;
    at a kink (a zero difference) the subgradient 0 is chosen, as autograd's
    ``abs`` does. Pure tensor math (no autograd graph, no backward), so it can
    be *injected* straight into ``object.grad`` -- cheaper than building a TV
    loss term and backpropagating it.

    Parameters
    ----------
    volume : Tensor
        Real floating dense volume, any number of dimensions. An axis with
        fewer than two voxels has no differences and contributes nothing
        (where the mean loss would be NaN).

    Returns
    -------
    Tensor
        Subgradient of the mean anisotropic TV, same shape, dtype and device
        as `volume`.
    """
    grad = torch.zeros_like(volume)
    for axis in range(volume.dim()):
        n = volume.shape[axis]
        if n < 2:
            continue
        d = torch.diff(volume, dim=axis)  # forward differences, len n-1
        s = torch.sign(d) / d.numel()  # d mean|d| / d d
        grad.narrow(axis, 0, n - 1).sub_(s)  # -sign at the left voxel
        grad.narrow(axis, 1, n - 1).add_(s)  # +sign at the right voxel
    return grad


@torch.no_grad()
def isotropic_tv_subgradient(volume: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Exact gradient of the mean isotropic TV loss of :func:`total_variation_3d`.

    The loss is ``mean_i sqrt(dz^2 + dy^2 + dx^2 + eps)`` over the common
    interior ``i in [0, D-1) x [0, H-1) x [0, W-1)``, with forward differences
    ``d.[i] = v[i + e.] - v[i]``. It couples the three axes through the
    per-voxel gradient norm, so it smooths more uniformly than the
    axis-separable anisotropic version. With ``eps > 0`` the loss is smooth, so
    this is its gradient everywhere, computed analytically without autograd.

    Parameters
    ----------
    volume : Tensor
        Real floating dense ``[D, H, W]`` volume. When an axis has fewer than
        two voxels the interior is empty and the result is all zeros (where
        the mean loss would be NaN).
    eps : float, optional
        Smoothing term inside the gradient-norm square root, by default
        1.0e-8, the default of :func:`total_variation_3d`.

    Returns
    -------
    Tensor
        Gradient of the mean isotropic TV, shape ``[D, H, W]``, on the dtype
        and device of `volume`.
    """
    grad = torch.zeros_like(volume)
    base = volume[:-1, :-1, :-1]
    if base.numel() == 0:
        return grad
    dz = volume[1:, :-1, :-1] - base
    dy = volume[:-1, 1:, :-1] - base
    dx = volume[:-1, :-1, 1:] - base
    w = 1.0 / (torch.sqrt(dz * dz + dy * dy + dx * dx + eps) * base.numel())
    grad[:-1, :-1, :-1] -= (dz + dy + dx) * w  # v[i] enters each difference as -1
    grad[1:, :-1, :-1] += dz * w  # v[i + ez] enters dz[i] as +1
    grad[:-1, 1:, :-1] += dy * w
    grad[:-1, :-1, 1:] += dx * w
    return grad


def total_variation_2d(image: Tensor, *, per_item: bool = False) -> Tensor:
    """Mean anisotropic total variation over the last two axes of an image.

    Forward differences: ``mean(|dH|) + mean(|dW|)``. When `per_item` is False
    the means cover the whole tensor; when True they reduce only the last two
    axes, returning one value per leading index. Matches the ``"tv"`` branch of
    the direct-ptychography image-sharpness metric.

    Parameters
    ----------
    image : Tensor
        Real floating image ``[..., H, W]``.
    per_item : bool, optional
        Reduce only over ``H`` and ``W`` when True. Default False.

    Returns
    -------
    Tensor
        Scalar mean TV, or a tensor of shape ``image.shape[:-2]`` when
        `per_item` is True.
    """
    dims = (-2, -1) if per_item else None

    def _mean(t: Tensor) -> Tensor:
        return t.mean(dim=dims) if dims is not None else t.mean()

    dy = _mean((image[..., 1:, :] - image[..., :-1, :]).abs())
    dx = _mean((image[..., :, 1:] - image[..., :, :-1]).abs())
    return dy + dx


def total_variation_3d(
    vol: Tensor, isotropic: bool = True, eps: float = 1.0e-8
) -> Tensor:
    """Mean 3D total variation of a dense ``[D, H, W]`` volume (differentiable).

    Isotropic: mean of ``sqrt(dz^2 + dy^2 + dx^2 + eps)`` over the common
    interior (edge-preserving, rotation-invariant). Anisotropic: sum of the
    mean absolute forward differences along each axis (cheaper, axis-aligned).
    Scale on this data: a converged 170^3 volume has TV ~1.6e-2 vs inner MSE
    ~2e-3, so ``tv_weight`` in 0.01-0.1 spans a gentle-to-strong prior.

    Parameters
    ----------
    vol : Tensor
        Real floating volume of shape ``(D, H, W)``. Differences are per voxel,
        without physical voxel-spacing normalization. Each axis must contain
        at least two voxels for a finite mean; an empty difference yields NaN.
    isotropic : bool, optional
        Couple the three differences on the common interior when True;
        otherwise sum their separate mean absolute values. Default True.
    eps : float, optional
        Squared-difference smoothing term for isotropic TV, default 1.0e-8.
        Ignored for anisotropic TV. Inputs and eps are not validated.

    Returns
    -------
    Tensor
        Differentiable scalar mean TV on the input dtype and device.
    """
    if isotropic:
        dz = vol[1:] - vol[:-1]
        dy = vol[:, 1:] - vol[:, :-1]
        dx = vol[:, :, 1:] - vol[:, :, :-1]
        mag = torch.sqrt(
            dz[:, :-1, :-1] ** 2 + dy[:-1, :, :-1] ** 2 + dx[:-1, :-1, :] ** 2 + eps
        )
        return mag.mean()
    return (
        mean_abs_forward_diff(vol, 0)
        + mean_abs_forward_diff(vol, 1)
        + mean_abs_forward_diff(vol, 2)
    )
