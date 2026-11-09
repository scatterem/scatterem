"""Bright-field pixel sets of a virtual bright field.

``_bf_mask_geometry`` builds the bright-field mask and its index and ordering
tensors from a mean CBED. ``_VbfSubsets`` gives
:class:`~scatterem.data.datasets.DatasetVirtualBrightField4DStem` its derived
providers: a flat-fielded copy, a subset of the bright-field pixels and two
detector-block half sets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .virtual_bright_field import DatasetVirtualBrightField4DStem

#: The cap acts when the mean-pattern maximum is above this factor times the
#: maximum of the 3 x 3-median-filtered mean pattern. See
#: :func:`_bf_reference_level`.
_BF_REFERENCE_CAP = 1.5

#: The cap acts only when the filtered maximum is above this factor times the
#: 5 % quantile of the mean pattern (the background). Below it, the median
#: filter has removed the disk itself.
_BF_DISK_OVER_BACKGROUND = 2.0


def _median3x3(image: Tensor) -> Tensor:
    """3 x 3 median filter of a 2D tensor, edges replicated."""
    padded = torch.nn.functional.pad(image[None, None], (1, 1, 1, 1), mode="replicate")
    windows = padded[0, 0].unfold(0, 3, 1).unfold(1, 3, 1)
    return windows.reshape(*image.shape, 9).median(-1).values


def _bf_reference_level(diff_mean: Tensor) -> Tensor:
    """The level that a bright-field mask threshold is a fraction of.

    The level is the maximum of the mean pattern (the peak), or the maximum
    of its 3 x 3-median-filtered copy (the filtered peak) when the cap acts.
    The cap acts when both of these hold:

    1. The peak is above ``_BF_REFERENCE_CAP`` (1.5) times the filtered peak.
    2. The filtered peak is above ``_BF_DISK_OVER_BACKGROUND`` (2) times the
       5 % quantile of the mean pattern.

    Rule 1 finds a few hot detector pixels above the disk. On one measured scan
    a detector row at 2.2 times the disk level set the peak: the 0.5
    threshold sat above the disk, and the vBF had 200 instead of ~1000
    pixels. The capped level is the disk level, so a threshold up to about 1
    keeps the disk. Rule 2 keeps the peak when the median filter removes the
    disk itself: a disk of about 2 x 2 pixels, or a disk on a zero
    background.

    When the cap does not act, the level is the peak, bit for bit, and the
    mask is the mask of the plain maximum. A hot pixel below 1.5 times the
    filtered peak does not make the cap act. It still sets the level.

    Parameters
    ----------
    diff_mean : Tensor
        The ``(K_y, K_x)`` mean pattern.

    Returns
    -------
    Tensor
        0-dim tensor on the device of ``diff_mean``.
    """
    peak = diff_mean.max()
    smooth = _median3x3(diff_mean).max()
    background = torch.quantile(diff_mean.flatten().double(), 0.05)
    disk_kept = smooth.double() > _BF_DISK_OVER_BACKGROUND * background
    hot = peak > _BF_REFERENCE_CAP * smooth
    return torch.where(hot & disk_kept & (smooth > 0), smooth, peak)


def _normalized_mean_pattern(diff_mean: Tensor) -> Tensor:
    """``diff_mean`` divided by :func:`_bf_reference_level` (a new tensor)."""
    return diff_mean / _bf_reference_level(diff_mean)


def _bf_mask_geometry(
    diff_mean: torch.Tensor, bright_field_mask_threshold: float
) -> dict[str, torch.Tensor]:
    """Bright-field mask + index/ordering tensors from a (max-normalized) mean
    CBED. Single source of truth shared by the eager vBF constructor and the
    out-of-core streaming provider — the two MUST produce identical masks,
    orderings and k-vectors for their reconstructions to be interchangeable.
    """
    bright_field_mask = diff_mean > bright_field_mask_threshold
    bright_field_inds = torch.argwhere(bright_field_mask)
    bright_field_inds_centered = (
        bright_field_inds.float() - torch.mean(bright_field_inds.float(), dim=0)[None]
    )
    bright_field_inds_radial_order = torch.argsort(
        torch.sum(bright_field_inds_centered**2, dim=1)
    )
    return {
        "bright_field_mask": bright_field_mask,
        "bright_field_inds": bright_field_inds,
        "bright_field_inds_centered": bright_field_inds_centered,
        "bright_field_inds_radial_order": bright_field_inds_radial_order,
        "bright_field_inds_ordered_by_radius": bright_field_inds[
            bright_field_inds_radial_order
        ],
        "bright_field_inds_centered_ordered_by_radius": bright_field_inds_centered[
            bright_field_inds_radial_order
        ],
    }


def _detector_block_parity(inds: Tensor, block: int) -> Tensor:
    """Checkerboard colour of detector pixels in ``block x block`` tiles.

    Parameters
    ----------
    inds : Tensor
        ``(N, 2)`` integer ``(row, col)`` detector-pixel indices.
    block : int
        Tile edge in detector pixels; ``1`` is the plain pixel checkerboard.

    Returns
    -------
    Tensor
        Boolean ``(N,)``: ``True`` where ``row // block + col // block`` is odd.

    Raises
    ------
    ValueError
        If ``block`` is not a positive integer.
    """
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError(f"block must be a positive integer, got {block!r}")
    tiles = torch.div(inds.long(), block, rounding_mode="floor")
    return (tiles.sum(1) % 2).bool()


def _subset_indices(idx: Tensor | Sequence[int], n: int) -> Tensor:
    """Sorted, unique, in-range CPU ``long`` indices from an index list or a mask."""
    idx = torch.as_tensor(idx)
    if idx.dtype == torch.bool:
        if tuple(idx.shape) != (n,):
            raise ValueError(
                f"a boolean idx needs shape ({n},), got {tuple(idx.shape)}"
            )
        idx = torch.nonzero(idx).flatten()
    idx = idx.long().flatten().cpu()
    if idx.numel() == 0:
        raise ValueError("idx selects no bright-field pixel")
    lo, hi = int(idx.min()), int(idx.max())
    if lo < 0 or hi >= n:
        raise ValueError(f"idx must lie in [0, {n}), got values in [{lo}, {hi}]")
    out = torch.unique(idx)  # sorted: the subset keeps the radius order
    if out.numel() != idx.numel():
        raise ValueError("idx holds the same bright-field pixel more than once")
    return out


def _derived(
    vbf: DatasetVirtualBrightField4DStem, array: Tensor, name: str, **fields: Any
) -> DatasetVirtualBrightField4DStem:
    """A new provider like ``vbf`` with ``array`` and the given fields replaced.

    The new provider lives on the compute device of ``vbf``. For a streaming
    vBF, this is ``vbf._compute_device``, not the CPU of its stack: the
    per-pixel fields (``k``, the q grids, the mask) are already there. The
    function does not move the aberrations of the shared ``meta``.
    """
    from .virtual_bright_field import DatasetVirtualBrightField4DStem as Vbf

    device = getattr(vbf, "_compute_device", None) or array.device
    aberrations = getattr(vbf.meta, "aberrations", None)
    placed = getattr(aberrations, "array", None)
    kw: dict[str, Any] = {
        "diffraction_pattern_mean_normalized": (
            vbf.diffraction_pattern_mean_normalized
        ),
        "bright_field_mask": vbf.bright_field_mask,
        "bright_field_inds": vbf.bright_field_inds,
        "bright_field_inds_centered": vbf.bright_field_inds_centered,
        "bright_field_inds_radial_order": vbf.bright_field_inds_radial_order,
        "bright_field_inds_ordered_by_radius": (
            vbf.bright_field_inds_ordered_by_radius
        ),
        "bright_field_inds_centered_ordered_by_radius": (
            vbf.bright_field_inds_centered_ordered_by_radius
        ),
        "k": vbf.k,
        "qx_1d": vbf.qx_1d,
        "qy_1d": vbf.qy_1d,
    }
    kw.update(fields)
    out = Vbf(
        array=array.to(device),
        name=name,
        origin=vbf.origin.copy(),
        sampling=vbf.sampling.copy(),
        units=list(vbf.units),
        parent_dataset=vbf.parent_dataset,
        signal_units=vbf.signal_units,
        meta=vbf.meta,
        astype_float32=False,
        fourier_shift_dim=None,
        probe_index=vbf.probe_index,
        device=device,
        clip_neg_values=False,
        _token=Vbf._token,
        **kw,
    )
    # Vbf.__init__ places the aberrations of the shared meta on the new
    # device. Put back the parent's tensor, so the parent's runs still see it.
    if aberrations is not None and placed is not None:
        aberrations.array = placed
    return out


class _VbfSubsets:
    """Derived providers of a :class:`DatasetVirtualBrightField4DStem`.

    Each method returns a new in-core provider and leaves ``self`` unchanged.
    The new provider starts with empty caches: no ``G``, no alignment-shift
    overlay, no total intensity and no direct-ptychography images. It keeps
    ``meta`` (the same object, as ``from_4dstem_dataset`` does) and
    ``parent_dataset``. It lives on the compute device of ``self``: the device
    of ``self.array`` for an in-core vBF. For a streaming vBF, it is the
    compute device, so the new provider holds its pixel columns of the stack
    in the memory of that device.
    """

    def flat_fielded(
        self,
    ) -> DatasetVirtualBrightField4DStem:
        """Return a copy with each bright-field pixel divided by its scan mean.

        The division removes the detector gain and the probe-intensity profile
        across the bright field: every pixel of the result has a scan mean of
        1. The mean CBED of the result reads 1 at every bright-field pixel, so
        consumers that divide by it (``normalized_bright_field``, the
        bright-field-shift self-calibration) see the flat-fielded values.

        Returns
        -------
        DatasetVirtualBrightField4DStem
            A new provider with the same pixels and geometry; ``self`` is
            unchanged.

        Raises
        ------
        ValueError
            If a bright-field pixel has a scan mean that is not positive.
        """
        vbf = cast("DatasetVirtualBrightField4DStem", self)
        mean = vbf.array.mean((0, 1), keepdim=True)
        n_bad = int((mean <= 0).sum())
        if n_bad:
            raise ValueError(
                f"{n_bad} bright-field pixel(s) have a scan mean <= 0; "
                "remove them with subset() before flat_fielded()"
            )
        inds = vbf.bright_field_inds_ordered_by_radius
        dpm = vbf.diffraction_pattern_mean_normalized.clone()
        dpm[inds[:, 0], inds[:, 1]] = 1.0
        return _derived(
            vbf,
            vbf.array / mean,
            f"{vbf.name} (flat-fielded)",
            diffraction_pattern_mean_normalized=dpm,
        )

    def subset(self, idx: Tensor | Sequence[int]) -> DatasetVirtualBrightField4DStem:
        """Return a provider over some of the bright-field pixels.

        Every per-pixel field follows ``idx``: the array columns, ``k``, the
        mask, all index tensors and the radial order. The centred indices
        keep the centre of the full bright field, so the ``k`` of a pixel
        does not change. The mean CBED stays the full detector map.

        Parameters
        ----------
        idx : Tensor or sequence of int
            Positions in the radius order (the last array axis), or a boolean
            mask of length ``n_bright_field``. The order of ``idx`` does not
            matter: the subset keeps the radius order.

        Returns
        -------
        DatasetVirtualBrightField4DStem
            A new provider with one column per selected pixel; ``self`` is
            unchanged.

        Raises
        ------
        ValueError
            If ``idx`` is empty, out of range or holds duplicates, or if it
            is a mask of the wrong length.
        """
        vbf = cast("DatasetVirtualBrightField4DStem", self)
        sel = _subset_indices(idx, vbf.n_bright_field)
        ordered = vbf.bright_field_inds_ordered_by_radius
        ordered = ordered[sel.to(ordered.device)]
        centered_ordered = vbf.bright_field_inds_centered_ordered_by_radius
        centered_ordered = centered_ordered[sel.to(centered_ordered.device)]
        mask = torch.zeros_like(vbf.bright_field_mask)
        mask[ordered[:, 0], ordered[:, 1]] = True
        inds = torch.argwhere(mask)
        width = mask.shape[-1]
        # Raster position of each radius-ordered pixel: inds[order] == ordered.
        order = torch.searchsorted(
            inds[:, 0] * width + inds[:, 1], ordered[:, 0] * width + ordered[:, 1]
        )
        centered = torch.empty_like(centered_ordered)
        centered[order.to(centered.device)] = centered_ordered
        return _derived(
            vbf,
            vbf.array[..., sel.to(vbf.array.device)].contiguous(),
            f"{vbf.name} (subset of {sel.numel()})",
            bright_field_mask=mask,
            bright_field_inds=inds,
            bright_field_inds_centered=centered,
            bright_field_inds_radial_order=order,
            bright_field_inds_ordered_by_radius=ordered,
            bright_field_inds_centered_ordered_by_radius=centered_ordered,
            k=vbf.k[sel.to(vbf.k.device)],
        )

    def split_halves(
        self, block: int = 4
    ) -> tuple[DatasetVirtualBrightField4DStem, DatasetVirtualBrightField4DStem]:
        """Split the bright field into two half sets on a detector checkerboard.

        Tiles of ``block x block`` detector pixels take alternate colours.
        The two pixel sets are disjoint, but their noise is not fully
        independent. Charge sharing correlates the noise of neighbouring
        pixels, and edge-adjacent tiles have opposite colours. The pixels at
        a tile edge therefore share charge with pixels of the other half. At
        ``block=4``, 12 of the 16 pixels of an inner tile touch the other
        half; a larger ``block`` lowers this fraction (about ``4 / block``).
        The residual correlation biases half-set statistics (FRC, SSNR)
        high. The bias is common to all reconstructions from the same split,
        so it shifts their scores together but can also flatten a score
        curve.

        Parameters
        ----------
        block : int, optional
            Tile edge in detector pixels. Default is 4.

        Returns
        -------
        tuple of DatasetVirtualBrightField4DStem
            ``(even, odd)``: the pixels where ``row // block + col // block``
            is even, then odd. The two pixel sets are disjoint, and their
            union is the bright field of ``self``.

        Raises
        ------
        ValueError
            If ``block`` is not a positive integer, or if one colour holds no
            bright-field pixel (``block`` larger than the bright field).
        """
        vbf = cast("DatasetVirtualBrightField4DStem", self)
        odd = _detector_block_parity(vbf.bright_field_inds_ordered_by_radius, block)
        return vbf.subset(~odd), vbf.subset(odd)
