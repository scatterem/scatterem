"""Measure the diffraction centre of mass across the scan."""

from __future__ import annotations

import torch

from ._common import _to_tensor
from ._origin import fit_origin

#: Byte budget for the masked product in :func:`_masked_marginals`.  A cube larger
#: than this is masked one scan-row block at a time instead of all at once.
_COM_MASK_BLOCK_BYTES = 256 * 2**20


def _masked_marginals(
    intensities: torch.Tensor,
    dp_mask: torch.Tensor,
    block_bytes: int = _COM_MASK_BLOCK_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(I * m).sum(-1)`` and ``(I * m).sum(-2)`` without a cube-sized product.

    The masked cube exists only to be reduced twice, so the same product and the
    same two sums are applied to scan-row blocks of ``intensities`` (C-contiguous
    ``(Rx, Ry, Qx, Qy)``), written through one reused block buffer into
    preallocated marginals.  Every element sees the same ops in the same order,
    so the marginals are bit-identical to the whole-cube spelling; the peak drops
    from one cube to one block.
    """
    Rx, Ry, Qx, Qy = intensities.shape
    dtype = torch.result_type(intensities, dp_mask)
    sum_dtype = dtype if (dtype.is_floating_point or dtype.is_complex) else torch.int64
    rows = max(1, min(Rx, int(block_bytes) // (Ry * Qx * Qy * dtype.itemsize)))
    dev = intensities.device
    buf = torch.empty((rows, Ry, Qx, Qy), dtype=dtype, device=dev)
    marginal_x = torch.empty((Rx, Ry, Qx), dtype=sum_dtype, device=dev)
    marginal_y = torch.empty((Rx, Ry, Qy), dtype=sum_dtype, device=dev)
    for s in range(0, Rx, rows):
        e = min(s + rows, Rx)
        blk = torch.mul(intensities[s:e], dp_mask, out=buf[: e - s])
        torch.sum(blk, dim=-1, out=marginal_x[s:e])
        torch.sum(blk, dim=-2, out=marginal_y[s:e])
    return marginal_x, marginal_y


def _fused_marginals(
    intensities: torch.Tensor, dp_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The one-pass Triton marginals, or ``None`` when declined."""
    try:
        from ._com_marginals_triton import com_marginals
    except ImportError:  # no Triton
        return None
    return com_marginals(intensities, dp_mask)


def _as_cube(intensities, device: str | torch.device | None) -> torch.Tensor:
    """Place the 4-D cube on ``device`` as float32, whatever its input dtype."""
    return _to_tensor(intensities, dtype=torch.float32, device=device)


def _measure_com_from_intensities(
    intensities: torch.Tensor,
    dp_mask: torch.Tensor | None,
    vectorized_calculation: bool,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First moment of every diffraction pattern, in detector pixel indices.

    With weights ``w = I * dp_mask`` (``I`` alone when there is no mask), the
    centre of mass at each scan position is ``sum(w * qx) / sum(w)`` along
    detector axis 0 and ``sum(w * qy) / sum(w)`` along axis 1, ``qx`` and
    ``qy`` being the pixel indices ``0, 1, ...``.

    The vectorized form is separable: sum the weights over one detector axis to
    get a 1-D marginal, then take the moment of that marginal. The looped form
    does the same arithmetic one pattern at a time, into float32 outputs.
    """
    cube = _as_cube(intensities, device)
    n_rx, n_ry, n_qx, n_qy = cube.shape
    weights_mask = None
    if dp_mask is not None:
        weights_mask = _to_tensor(dp_mask, dtype=torch.float32, device=cube.device)
        if tuple(weights_mask.shape) != (n_qx, n_qy):
            raise ValueError(
                f"dp_mask must match the detector shape {(n_qx, n_qy)}, "
                f"got {tuple(weights_mask.shape)}"
            )
    index_x = torch.arange(n_qx, dtype=torch.float32, device=cube.device)
    index_y = torch.arange(n_qy, dtype=torch.float32, device=cube.device)

    if not vectorized_calculation:
        out_x = torch.zeros((n_rx, n_ry), dtype=torch.float32, device=cube.device)
        out_y = torch.zeros_like(out_x)
        # Index planes broadcast over one pattern: row index, column index.
        plane_x, plane_y = index_x[:, None], index_y[None, :]
        for i in range(n_rx):
            for j in range(n_ry):
                pattern = cube[i, j]
                if weights_mask is not None:
                    pattern = pattern * weights_mask
                total = pattern.sum()
                out_x[i, j] = (pattern * plane_x).sum() / total
                out_y[i, j] = (pattern * plane_y).sum() / total
        return out_x, out_y

    marginals = _fused_marginals(cube, weights_mask)
    if marginals is None:
        if weights_mask is None:
            marginals = (cube.sum(-1), cube.sum(-2))
        else:
            product_bytes = (
                cube.numel() * torch.result_type(cube, weights_mask).itemsize
            )
            if cube.is_contiguous() and product_bytes > _COM_MASK_BLOCK_BYTES:
                marginals = _masked_marginals(
                    cube, weights_mask, block_bytes=_COM_MASK_BLOCK_BYTES
                )
            else:
                weighted = cube * weights_mask
                marginals = (weighted.sum(-1), weighted.sum(-2))
    along_x, along_y = marginals
    # Both marginals carry the same total weight; take it from the first.
    total = along_x.sum(-1)
    com_x = (along_x * index_x).sum(-1) / total
    com_y = (along_y * index_y).sum(-1) / total
    return com_x, com_y


def _calculate_intensities_center_of_mass(
    intensities: torch.Tensor,
    reciprocal_sampling: tuple[float, float],
    dp_mask: torch.Tensor | None = None,
    fit_function: str | None = "plane",
    com_shifts: tuple[torch.Tensor, torch.Tensor] | None = None,
    com_measured: tuple[torch.Tensor, torch.Tensor] | None = None,
    vectorized_calculation: bool = True,
    device: str | torch.device | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute and fit diffraction-pattern centre-of-mass.

    Parameters
    ----------
    intensities : Tensor, shape (Rx, Ry, Qx, Qy)
    reciprocal_sampling : (dx, dy)
    dp_mask : Tensor, shape (Qx, Qy), optional
    fit_function : str or None
    com_shifts : (Tensor, Tensor), optional — pre-computed fitted CoM
    com_measured : (Tensor, Tensor), optional — pre-computed measured CoM
    device : str, torch.device or None
        Default: the device of ``intensities`` if it is a tensor, else CPU.

    Returns
    -------
    com_measured_x, com_measured_y,
    com_fitted_x, com_fitted_y,
    com_normalized_x, com_normalized_y

    Notes
    -----
    Clean-room implementation from a behavioural specification, with the
    interface kept from py4DSTEM: the name, the parameters and their defaults,
    and the order of the six returned arrays (see :mod:`._surfaces`). The
    method is that of Savitzky et al. (2021) and Ophus (2019). The measured CoM
    is the first moment of each pattern; the fitted CoM is a smooth surface through it
    (:func:`fit_origin`), which models the descan; the normalized CoM is the
    difference of the two, with NaN set to zero (``torch.nan_to_num``, so
    +-inf becomes the largest finite float32), scaled to reciprocal units.
    """
    intensities = _as_cube(intensities, device)
    device = intensities.device

    if com_measured is not None:
        measured_x = _to_tensor(com_measured[0], dtype=torch.float32, device=device)
        measured_y = _to_tensor(com_measured[1], dtype=torch.float32, device=device)
    else:
        measured_x, measured_y = _measure_com_from_intensities(
            intensities, dp_mask, vectorized_calculation, device
        )

    if com_shifts is not None:
        fitted_x = _to_tensor(com_shifts[0], dtype=torch.float32, device=device)
        fitted_y = _to_tensor(com_shifts[1], dtype=torch.float32, device=device)
    elif fit_function is not None:
        surfaces = fit_origin(
            (measured_x, measured_y),
            mask=torch.isfinite(measured_x),
            fit_function=fit_function,
            device=device,
        )
        fitted_x = surfaces[0].to(torch.float32)
        fitted_y = surfaces[1].to(torch.float32)
    else:
        fitted_x, fitted_y = measured_x.clone(), measured_y.clone()

    normalized_x = torch.nan_to_num(measured_x - fitted_x) * reciprocal_sampling[0]
    normalized_y = torch.nan_to_num(measured_y - fitted_y) * reciprocal_sampling[1]
    return (
        measured_x,
        measured_y,
        fitted_x,
        fitted_y,
        normalized_x,
        normalized_y,
    )
