"""Virtual detectors on a 4D-STEM dataset.

``_VirtualDetectors`` gives :class:`~scatterem.data.datasets.Dataset4DStem`
its detector masks, its virtual images and its dark-field half sets.
``_detector_group_means_contracted`` is the one-pass form of the per-group
detector mean that tilt-corrected dark field reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from torch import Tensor

from ._vbf_subsets import _detector_block_parity, _normalized_mean_pattern

if TYPE_CHECKING:
    from .dataset_4dstem import Dataset4DStem

#: Units of the radii of :meth:`_VirtualDetectors.annular_detector_mask`.
AnnularUnits = Literal["mrad", "pixels", "inv_angstrom", "bf_radius"]
_ANNULAR_UNITS = ("mrad", "pixels", "inv_angstrom", "bf_radius")

#: Upper bound on the bytes of one block of scan rows that
#: :meth:`_VirtualDetectors.virtual_image` reads at a time (float32).
_VIRTUAL_IMAGE_BLOCK_BYTES = 128 * 1024**2


def _float_block(dataset: Dataset4DStem, block: Tensor | np.ndarray) -> Tensor:
    """A scan block as float32 torch data, in the normalised units of the dataset.

    ``Dataset4DStem.iter_scan_blocks`` yields blocks that are already
    transformed, for resident and source-backed datasets alike.
    """
    del dataset
    if isinstance(block, np.ndarray):
        return torch.from_numpy(block.astype(np.float32))
    return block.to(torch.float32)


class _VirtualDetectors:
    """Detector masks, virtual images and dark-field half sets of a 4D dataset.

    The masks are boolean ``(K_y, K_x)`` tensors on the detector grid of
    the dataset. The images are float32 ``(R_y, R_x)`` tensors on
    ``self.device``.
    """

    def bright_field_mask(self, bright_field_mask_threshold: float = 0.3) -> Tensor:
        """Return the bright-field detector mask that tcDF uses.

        A pixel is in the bright field where the mean CBED
        (:attr:`mean_cbed_tcdf`), divided by its reference level, is above
        the threshold. The reference level is the maximum, capped at 1.5
        times the maximum of the 3 x 3-median-filtered pattern, so a few hot
        pixels cannot lift it above the disk. Tilt-corrected dark field builds its mask the same way, so
        the two masks agree for the same threshold. The direct-ptychography
        vBF (:meth:`get_vbf`) averages a smaller scan window, so its mask can
        differ at the disk edge.

        Parameters
        ----------
        bright_field_mask_threshold : float, optional
            Fraction of the reference level of the mean CBED. Default is 0.3.

        Returns
        -------
        Tensor
            Boolean ``(K_y, K_x)`` mask on ``self.device``.
        """
        ds = cast("Dataset4DStem", self)
        diff_mean = ds.mean_cbed_tcdf.to(ds.device)
        return _normalized_mean_pattern(diff_mean) > bright_field_mask_threshold

    def annular_detector_mask(
        self,
        inner: float,
        outer: float,
        *,
        units: AnnularUnits = "mrad",
        center: Sequence[float] | None = None,
    ) -> Tensor:
        """Return the mask of an annular detector on the detector grid.

        Parameters
        ----------
        inner : float
            Inner radius, included.
        outer : float
            Outer radius, excluded.
        units : {"mrad", "pixels", "inv_angstrom", "bf_radius"}, optional
            Units of ``inner`` and ``outer``. ``"mrad"`` uses the detector
            sampling (``sampling[-2:]``, 1/Angstrom per pixel) and
            ``meta.wavelength``. ``"inv_angstrom"`` uses the detector sampling
            only. ``"bf_radius"`` uses the fitted bright-field radius in pixels
            from :meth:`bright_field_radius_and_center` as one unit, so BF is
            ``(0, 0.5)``, ABF ``(0.5, 1)`` and ADF for example ``(1.05, 1.25)``.
            It does not use ``dk``; ``pad_detector(unit="alpha")`` uses the
            nominal ``meta.semiconvergence_angle`` and ``dk`` instead.
            Default is ``"mrad"``.
        center : sequence of float, optional
            ``(row, col)`` centre in detector pixels. Default is the centroid
            of :meth:`bright_field_mask` at its default threshold, the centre
            that the vBF uses for ``k``.

        Returns
        -------
        Tensor
            Boolean ``(K_y, K_x)`` mask on ``self.device``, ``True`` where
            ``inner <= r < outer``.

        Raises
        ------
        ValueError
            If ``units`` is not one of the four above, or if ``inner > outer``.
        """
        ds = cast("Dataset4DStem", self)
        if units not in _ANNULAR_UNITS:
            raise ValueError(f"units must be one of {_ANNULAR_UNITS}, got {units!r}")
        if inner > outer:
            raise ValueError(f"inner ({inner}) must not exceed outer ({outer})")
        if center is None:
            inds = torch.argwhere(ds.bright_field_mask()).to(torch.float64)
            center = inds.mean(0).tolist()
        ky, kx = (int(s) for s in ds.shape[-2:])
        dy = torch.arange(ky, dtype=torch.float64) - float(center[0])
        dx = torch.arange(kx, dtype=torch.float64) - float(center[1])
        if units in ("mrad", "inv_angstrom"):
            sy, sx = (float(s) for s in ds.sampling[-2:])
            scale = ds._require_meta().wavelength * 1e3 if units == "mrad" else 1.0
            dy, dx = dy * sy * scale, dx * sx * scale
        elif units == "bf_radius":
            radius = ds.bright_field_radius_and_center().rim_radius
            dy, dx = dy / radius, dx / radius
        r = torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
        return ((r >= inner) & (r < outer)).to(ds.device)

    def virtual_image(self, mask: Tensor, *, block_rows: int | None = None) -> Tensor:
        """Sum the detector signal under a mask at every scan position.

        The method reads the data in blocks of scan rows
        (``iter_scan_blocks``), so it never builds a second 4D array. It also
        works on out-of-core datasets.

        Parameters
        ----------
        mask : Tensor
            ``(K_y, K_x)`` or ``(n, K_y, K_x)`` detector weights, boolean or
            real.
        block_rows : int, optional
            Scan rows per block. Default: as many as fit in 128 MiB.

        Returns
        -------
        Tensor
            Float32 ``(R_y, R_x)`` image, or ``(n, R_y, R_x)`` for a stack of
            masks, on ``self.device``.

        Raises
        ------
        ValueError
            If the mask does not end in the detector shape.
        """
        ds = cast("Dataset4DStem", self)
        det = tuple(int(s) for s in ds.shape[-2:])
        w = torch.as_tensor(mask)
        if w.ndim not in (2, 3) or tuple(w.shape[-2:]) != det:
            raise ValueError(
                f"mask of shape {tuple(w.shape)} does not end in the detector "
                f"shape {det}"
            )
        weights = w.reshape(-1, *det).to(torch.float32)
        ny, nx = (int(s) for s in ds.shape[:2])
        if block_rows is None:
            row_bytes = nx * det[0] * det[1] * 4
            block_rows = max(1, _VIRTUAL_IMAGE_BLOCK_BYTES // row_bytes)
        out = torch.empty(
            (weights.shape[0], ny, nx), dtype=torch.float32, device=ds.device
        )
        for s, e, raw in ds.iter_scan_blocks(block_rows):
            block = _float_block(ds, raw)
            wb = weights.to(block.device)
            for j in range(wb.shape[0]):
                out[j, s:e] = (block * wb[j]).sum((-2, -1)).to(ds.device)
        return out[0] if w.ndim == 2 else out

    def virtual_bright_field(self, bright_field_mask_threshold: float = 0.3) -> Tensor:
        """Return the virtual bright-field image.

        Parameters
        ----------
        bright_field_mask_threshold : float, optional
            Threshold of :meth:`bright_field_mask`. Default is 0.3.

        Returns
        -------
        Tensor
            Float32 ``(R_y, R_x)`` image on ``self.device``.
        """
        ds = cast("Dataset4DStem", self)
        return ds.virtual_image(ds.bright_field_mask(bright_field_mask_threshold))

    def virtual_annular(
        self,
        inner: float,
        outer: float,
        *,
        units: AnnularUnits = "mrad",
        center: Sequence[float] | None = None,
    ) -> Tensor:
        """Return the image of an annular detector (ABF, ADF, HAADF).

        Parameters
        ----------
        inner : float
            Inner radius, included.
        outer : float
            Outer radius, excluded.
        units : {"mrad", "pixels", "inv_angstrom", "bf_radius"}, optional
            Units of the radii; see :meth:`annular_detector_mask`.
        center : sequence of float, optional
            ``(row, col)`` centre; see :meth:`annular_detector_mask`.

        Returns
        -------
        Tensor
            Float32 ``(R_y, R_x)`` image on ``self.device``.
        """
        ds = cast("Dataset4DStem", self)
        mask = ds.annular_detector_mask(inner, outer, units=units, center=center)
        return ds.virtual_image(mask)

    def split_dark_field_halves(
        self, bright_field_mask_threshold: float = 0.3, block: int = 4
    ) -> tuple[Dataset4DStem, Dataset4DStem]:
        """Split the dark field into two half sets for tilt-corrected dark field.

        Each half keeps the whole bright field and one colour of a
        checkerboard of ``block x block`` detector tiles in the dark field.
        The other dark-field pixels are set to 0. The colours are those of
        :meth:`DatasetVirtualBrightField4DStem.split_halves`.

        Memory: each half is a full in-core copy of the array, so the two
        halves need twice the memory of ``self``. The tcDF path has no
        detector-mask input that would let the halves share the data.

        Run tcDF on each half with the same ``bright_field_mask_threshold``.
        The bright-field mask of each half is then that of ``self``. With
        ``dark_field_scaling=True`` each half rescales by its own recorded
        dark field, which holds about half the counts.

        Parameters
        ----------
        bright_field_mask_threshold : float, optional
            Threshold of :meth:`bright_field_mask`. Default is 0.3, the tcDF
            default.
        block : int, optional
            Tile edge in detector pixels. Default is 4.

        Returns
        -------
        tuple of Dataset4DStem
            ``(even, odd)`` copies of ``self``. Their dark-field pixel sets are
            disjoint, and their union is the dark field of ``self``. ``meta``
            is copied; the dose fields stay those of the full acquisition.

        Raises
        ------
        TypeError
            If ``self`` is an out-of-core dataset.
        ValueError
            If ``block`` is not a positive integer.
        """
        ds = cast("Dataset4DStem", self)
        if not ds.source.is_resident:
            raise TypeError(
                "split_dark_field_halves needs an in-core dataset: each half is "
                "a full copy of the array"
            )
        ky, kx = (int(s) for s in ds.shape[-2:])
        grid = torch.cartesian_prod(torch.arange(ky), torch.arange(kx))
        odd = _detector_block_parity(grid, block).reshape(ky, kx).to(ds.device)
        bf = ds.bright_field_mask(bright_field_mask_threshold)
        halves = []
        for i, colour in enumerate((~odd, odd)):
            half = ds.copy()
            half.array.mul_((bf | colour).to(half.array.dtype))
            half.name = f"{ds.name} (dark-field half {i})"
            halves.append(half)
        return halves[0], halves[1]


#: Coverage floor for the one-pass route in
#: ``_detector_group_means_contracted``.  A RESOURCE ratio derived from the
#: two routes' compulsory traffic, not a fitted constant: the per-group gather
#: moves about ``3 * P * D * 4`` bytes (it writes a ``(ny, nx, M_i)`` float
#: intermediate per group and reads it straight back for the ``mean``, on top of
#: the gathered read), while the contraction moves ``P * C * 4`` and reads every
#: detector column exactly once.  So the contraction is the cheaper route while
#: ``C < 3 * D`` -- i.e. while the groups together cover more than a third of the
#: detector.  Written as a ratio of the two traffics so that it degrades safely
#: when either the detector or the segmentation changes shape.
_GROUP_MEANS_COVERAGE_FLOOR = 1.0 / 3.0


def _detector_group_means_contracted(
    arr: torch.Tensor, groups: list[torch.Tensor]
) -> list[torch.Tensor] | None:
    """One-pass form of ``[arr[:, :, g[:, 0], g[:, 1]].mean(-1) for g in groups]``,
    or ``None`` when it does not apply and the caller must use that gather.

    Why.  On the FF-STEM figure
    path the per-group gather is the single largest object in the row: at
    ``r14_ffstem/gd2o3`` the 32 azimuthal dark-field segments cost 45.7 ms, of
    which 35.8 ms is the gather, against 9.5 ms for reading the gathered columns
    once.  The defect is not the gather's arithmetic -- it has none -- but that
    every group MATERIALISES a ``(ny, nx, M_i)`` intermediate, 6.30 GiB of
    columns written and read back to produce 33 MB of images, and that each of
    the 32 gathers re-reads its own scattered subset of the same detector block.

    A per-group mean over member pixels is a contraction of the detector axis
    with a ``(C, S)`` weight matrix carrying ``1 / M_j`` on group ``j``'s
    members, so ONE matmul computes all groups while reading each detector
    column exactly once, perfectly coalesced, with no intermediate at all.

    NOT bit-exact, and not claimed to be: the reduction order changes from
    torch's per-row tree to cuBLAS's ``K``-blocked accumulation.  It is however
    measurably *more* accurate -- against an independent float64 reference over
    the same data the shipped chain reads max|d| 2.503e-08 and this route
    1.795e-08 -- because a matmul accumulates a length-``C`` row in wide blocks
    where ``mean`` reduces only the ``M_i`` gathered values.

    ``fp32_precision = "ieee"`` is load-bearing and not a precaution.
    A call to ``scatterem.configure()`` sets float32 matmul
    precision to ``"medium"``, so without this the contraction would run in TF32
    and truncate the detector counts to a 10-bit mantissa.  The new ``fp32_precision``
    attribute is used rather than the legacy ``allow_tf32`` flag because setting
    the latter alongside ``set_float32_matmul_precision`` puts torch in a mixed
    state in which ``get_float32_matmul_precision()`` itself raises.
    """
    if not groups:
        return None
    # Everything here is host-known, so the predicate costs no device sync.
    if (
        arr.device.type != "cuda"
        or arr.dtype != torch.float32
        or arr.dim() != 4
        or not arr.is_contiguous()
    ):
        # A float64 array would contract at 1/32 rate and lose outright; a
        # half-precision one would accumulate the detector axis in half; a
        # non-contiguous array cannot be reshaped to (P, C) without a copy of
        # the whole cube; and the CPU route has no coalescing to win back.
        return None
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return None

    ny, nx, my, mx = (int(s) for s in arr.shape)
    n_col = my * mx
    lens = [int(g.shape[0]) for g in groups]
    if sum(lens) < _GROUP_MEANS_COVERAGE_FLOOR * n_col:
        return None

    n_grp = len(groups)
    device, dtype = arr.device, arr.dtype
    # The group axis and the 1/M_j weights are built on the HOST, from ``lens``,
    # and crossed once.  ``torch.repeat_interleave`` with a CUDA repeat tensor
    # cannot be used here: its output shape is data-dependent, so it SYNCS, and
    # at the small figure size this path is ~99.9 % host-bound, where a
    # sync costs more than the whole device saving is worth.  Measured: two such
    # syncs turned ``carbon/tcdf`` into a 1.3 % loss.
    lens_np = np.asarray(lens)
    inv = np.zeros(len(lens), dtype=np.float64)
    np.divide(1.0, lens_np.astype(np.float64), out=inv, where=lens_np > 0)
    grp = torch.from_numpy(np.repeat(np.arange(n_grp), lens)).to(device)
    val = torch.from_numpy(np.repeat(inv, lens)).to(device=device, dtype=dtype)
    # One ``cat`` over the group list, not one index expression per group.
    allg = torch.cat(groups, 0)
    flat = allg[:, 0].long() * mx + allg[:, 1].long()
    # ``index_add_`` rather than an assignment so that a group listing the same
    # detector pixel twice weighs it twice, exactly as the gather would.
    weight = torch.zeros(n_col * n_grp, device=device, dtype=dtype)
    weight.index_add_(0, flat * n_grp + grp, val)

    prev = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        res = arr.reshape(ny * nx, n_col) @ weight.reshape(n_col, n_grp)
    finally:
        torch.backends.cuda.matmul.fp32_precision = prev

    images = list(res.t().contiguous().reshape(n_grp, ny, nx))
    for j, m in enumerate(lens):
        if m == 0:
            # Parity with ``mean`` over an empty gather, which the contraction
            # would otherwise report as a plain 0.0.
            images[j] = torch.full((ny, nx), float("nan"), device=device, dtype=dtype)
    return images
