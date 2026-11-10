import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from scatterem.data import validation
from scatterem.data.aberrations import Aberrations
from scatterem.data.acquisition import AcquisitionMetadata, _DetectorReach
from scatterem.data.pupil_gauge import PupilEstimate, measure_pupil
from scatterem.data.validation import ensure_valid_array, validate_ndinfo

from . import _detector_ops, _direct, _init_passes
from ._blocks import _SourceBacked
from ._meta import copy_meta, meta_sampling, sync_meta_sampling, sync_meta_shape
from ._parent_link import MetaDropsVbfCache
from ._virtual_detectors import _VirtualDetectors
from .base import Dataset, _place_aberrations_on_device
from .virtual_bright_field import DatasetVirtualBrightField4DStem

if TYPE_CHECKING:
    from typing import Self


logger = logging.getLogger(__name__)


class _BrightFieldCropBox(NamedTuple):
    crop_slice: tuple
    radius: int  # crop half-width, clamped to the detector
    center: NDArray  # [y, x] bright-field center
    y0: int
    x0: int
    radius_ceil: float  # ceil of the detected radius, before clamping


class _PhysicsMetadata(NamedTuple):
    meta: AcquisitionMetadata
    sampling: tuple[float, float, float, float]
    origin: NDArray


def _metadata4dstem_from_physics(
    array_shape: Sequence[int],
    energy: float,
    semiconvergence_angle: float | None,
    scan_step: float | tuple | list | NDArray | None,
    reciprocal_step: float | tuple | list | NDArray | None = None,
    rotation: float = 0.0,
    aberrations: Aberrations | None = None,
) -> _PhysicsMetadata:
    """Build an ``AcquisitionMetadata`` from acquisition physics.

    The "physics convenience path" used by ``Dataset4DStem.from_array``; also
    shared with the downloadable public datasets (``data/public/base.py``).

    ``scan_step`` and ``reciprocal_step`` each accept a scalar or a length-2
    pair; ``reciprocal_step`` defaults to the ``(1.0, 1.0)`` placeholder that a
    later calibration (e.g. ``calibrate_reciprocal_from_bright_field``)
    overwrites.

    Returns:
        A ``_PhysicsMetadata(meta, sampling, origin)`` where ``sampling`` is
        ``(dr0, dr1, dk0, dk1)`` and ``origin`` is the array center.
    """
    shape = np.array(array_shape)

    def _as_pair(v, default):
        v = default if v is None else v
        if hasattr(v, "__len__"):
            return float(v[0]), float(v[1])
        return float(v), float(v)

    dr0, dr1 = _as_pair(scan_step, 1.0)
    dk0, dk1 = _as_pair(reciprocal_step, 1.0)
    sampling = (dr0, dr1, dk0, dk1)
    meta = AcquisitionMetadata(
        energy=energy,
        semiconvergence_angle=float(semiconvergence_angle or 0.0),
        sampling=sampling,
        shape=shape,
        rotation=rotation,
        aberrations=aberrations,
    )
    return _PhysicsMetadata(meta=meta, sampling=sampling, origin=shape / 2.0)


class Dataset4DStem(
    _VirtualDetectors, _SourceBacked, _detector_ops.DetectorOps, Dataset, _DetectorReach
):
    """A 4D STEM dataset with metadata for electron diffraction patterns.

    Parameters
    ----------
    array : Tensor
        The 4D diffraction data, shaped ``(scan_y, scan_x, det_y, det_x)``.
    name : str
        The name of the dataset.
    origin : NDArray or tuple or list or float or int
        The per-axis origin, i.e. the pixel offset of the array's zero index.
    sampling : NDArray or tuple or list or float or int
        The per-axis physical sampling (scan step and detector pixel size).
    units : list of str or tuple or list
        The per-axis physical units.
    signal_units : str
        The units of the recorded signal.
    meta : AcquisitionMetadata, optional
        Acquisition metadata (energy, semiconvergence angle, aberrations,
        and related quantities).
    transform_to_amplitudes : bool
        Whether to take the square root of the array in place, converting
        recorded intensities to amplitudes.
    astype_float32 : bool
        Whether to cast the array to ``float32``.
    fourier_shift_dim : tuple, optional
        Detector dims to ``fftshift`` on construction: only to centre corner data.
    probe_index : int
        Index of the probe this dataset is associated with.
    device : torch.device or str or None, optional
        ``None`` keeps a torch tensor's device; other inputs become CPU.
    normalize : bool
        Whether to divide the array in place by the maximum mean diffraction
        pattern. Default is ``True``. **This rescales the stored data**: pass
        ``normalize=False`` (and ``clip_neg_values=False``) whenever raw
        counts must survive byte-exact, e.g. for dose accounting or
        comparisons against another reader.
    clip_neg_values : bool
        Whether to clamp negative values to zero in place. Default is
        ``True``; see the note on ``normalize``.
    copy : bool
        Whether to clone the array before the in-place operations above run,
        so that the caller's array is never mutated.
    _token : object, optional
        Sentinel guarding direct construction; use :meth:`from_array` instead.

    Attributes
    ----------
    bright_field_mask_threshold : float
        Default bright-field mask level, 0.3. ``FusedFullFieldConfig`` reads it
        when the caller omits the threshold; the other configs have their own
        default. The examples pass it to every config; set it per dataset.
    """

    SERIAL_ASSIGNED = ("transform_to_amplitudes", "fourier_shift_dim", "_is_cropped")
    SERIAL_ASSIGNED += ("_is_bright_field", "bright_field_mask_threshold")
    SERIAL_ASSIGNED += ("dose_per_probe_unnormalized", "_total_intensity")
    SERIAL_ASSIGNED += _detector_ops.DetectorOps.SERIAL_OPTIONAL
    SERIAL_FIELDS = (*Dataset.SERIAL_FIELDS, "meta", "probe_index", *SERIAL_ASSIGNED)
    SERIAL_NESTED: ClassVar[dict] = {"meta": AcquisitionMetadata}
    SERIAL_INIT_KWARGS: ClassVar = {**Dataset.SERIAL_INIT_KWARGS, "normalize": False}
    SERIAL_INIT_KWARGS |= {"astype_float32": False, "clip_neg_values": False}
    meta = MetaDropsVbfCache()
    _is_cropped: bool = False
    _is_bright_field: bool = False
    dose_per_probe_normalized: float | None = None

    def _require_meta(self) -> "AcquisitionMetadata":
        if self.meta is None:
            raise ValueError("Dataset4DStem.meta is not set; load metadata first")
        return self.meta

    def __init__(
        self,
        array: Tensor | np.ndarray,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        meta: "AcquisitionMetadata | None" = None,
        transform_to_amplitudes: bool = False,
        astype_float32: bool = True,
        fourier_shift_dim: tuple | None = None,
        probe_index: int = 0,
        device: torch.device | str | None = None,
        normalize: bool = True,
        clip_neg_values: bool = True,
        copy: bool = True,
        _token: object | None = None,
    ) -> None:
        super().__init__(
            array=array,
            name=name,
            signal_units=signal_units,
            origin=origin,
            sampling=sampling,
            units=units,
            device=device,
            _token=_token,
        )

        self.meta = meta
        _place_aberrations_on_device(meta, self._array.device)
        self.transform_to_amplitudes = transform_to_amplitudes
        self.fourier_shift_dim = fourier_shift_dim
        self.probe_index = probe_index
        self.bright_field_mask_threshold = 0.3  # see the class docstring

        self._shape = self._array.shape
        # Track whether we already hold a private buffer. `.to(float32)` copies only
        # when the dtype actually changes, and fftshift always returns a new tensor.
        _private = self._setter_copied  # the _array setter made a contiguous copy
        if astype_float32:
            _private = _private or self._array.dtype != torch.float32
            self._array = self._array.to(torch.float32)
        self._array, _row_sums = _init_passes.shift_and_clamp(
            self._array,
            fourier_shift_dim,
            clip_neg_values,
            transform_to_amplitudes,
            normalize,
        )
        _private = _private or fourier_shift_dim is not None
        # Everything from here on mutates self._array IN PLACE (the clip, sqrt_()
        # and the normalization below). If it still shares storage with the
        # caller's buffer, those writes corrupt the caller's array (a float32 numpy
        # cube came back divided by normalization_const). Take one copy here.
        if (
            copy
            and not _private
            and (clip_neg_values or transform_to_amplitudes or normalize)
        ):
            self._array = self._array.clone()

        # Total intensity is a property of the DATA: capture it HERE, before the
        # normalization below, or `fluence` is off by 1/normalization_const. One
        # float64 sum per outer slice, one host sync; the clamp and sqrt_()
        # run in place first, unless a fused pass made the sums.
        _total_unnormalized = _init_passes.clamp_sqrt_total(
            self._array, clip_neg_values, transform_to_amplitudes, _row_sums
        )
        self.dose_per_probe_unnormalized = _total_unnormalized / (
            self._shape[0] * self._shape[1]
        )

        if _init_passes.normalize_pending(self._array, normalize):
            # Not when a fused pass has divided already.
            self._array /= _init_passes.normalization_const(self._array, self._shape)

        # Seeded with the pre-normalization float64 sum computed above -- NOT None. A
        # lazy sum here would read the already-normalized array.
        self._total_intensity = _total_unnormalized

    #: Byte budget for the ONE demoted vBF :meth:`get_vbf` may keep alive
    #: besides ``self.vBF`` (see there).  Same value, and the same reasoning, as
    #: ``_TV_CHUNK_CACHE_BUDGET_BYTES`` in ``reconstruction/ptychography/direct``:
    #: a retained tensor is affordable when it is small next to the working set
    #: and must not be retained when it is not.  Written in terms of a RESOURCE
    #: rather than an entry count on purpose -- a resource-shaped gate degrades
    #: safely as configurations grow, a count-shaped one silently stops being
    #: the decision it was measured as.
    _VBF_SPARE_BUDGET_BYTES = 4 * 1024**3

    @staticmethod
    def _vbf_retained_bytes(vbf: "DatasetVirtualBrightField4DStem") -> int:
        """Upper bound on what keeping ``vbf`` alive costs.

        The bright-field stack it owns, plus the ``G`` of the same shape its
        lazy property will materialise (complex64, 8 bytes/element) -- counted
        whether or not ``G`` exists yet, so the budget cannot be walked past by
        caching a provider before its transform is taken.
        """
        n = vbf.array.numel()
        return n * vbf.array.element_size() + n * 8

    def get_vbf(
        self, bright_field_mask_threshold: float
    ) -> "DatasetVirtualBrightField4DStem":
        """Cached virtual-bright-field provider, keyed on the mask threshold.

        Keeps ``self.vBF`` (and its cached ~17 GiB G for full-scan data) alive
        per dataset across the aberration fit, the reconstruction, the empirical
        SSNR and the depth section, plus AT MOST ONE demoted provider while that
        one fits in ``_VBF_SPARE_BUDGET_BYTES``. Subclasses override
        ``_build_vbf`` to pick their provider flavor.

        WHY THE SECOND SLOT EXISTS.  One slot is one slot only if there is one
        threshold, and on the published fusion path there are two: the
        reconstruction asks for the caller's ``bright_field_mask_threshold``
        while ``phase_contrast_transfer_function`` asks for its own default,
        because ``direct_ptychography_ssnr`` does not forward it.  A one-slot
        cache asked alternately for two keys hits NEVER -- measured on
        ``r14_ffstem/carbon/ptycho``: ``_build_vbf`` ran **twice per
        call, every call**, 1.879 ms of a 16.96 ms row, forever.  Holding the
        evicted provider fixes the thrash without changing which vBF any caller
        receives: every caller still gets exactly the provider
        ``_build_vbf(threshold)`` returns for the threshold it asked for.

        Making the two callers agree on a threshold would ALSO fix it and would
        cost no memory -- but it changes the bright-field mask the PCTF is built
        from, hence the SSNR weights, hence the published fused image. That is a
        physics decision and is deliberately NOT taken here.

        ``self.vBF`` and ``self._vbf_threshold`` keep exactly their previous
        meaning (the most recently requested provider and its key). When a
        new object is bound to ``self.meta``, all three slots are dropped
        (``_parent_link.MetaDropsVbfCache``). No provider outlives its metadata.

        Parameters
        ----------
        bright_field_mask_threshold : float
            The mask threshold to key the cached provider on.

        Returns
        -------
        DatasetVirtualBrightField4DStem
            The provider for this threshold (freshly built, or reused from
            ``self.vBF`` / the demoted spare).
        """
        vbf = getattr(self, "vBF", None)
        threshold = getattr(self, "_vbf_threshold", None)
        if vbf is not None and threshold == bright_field_mask_threshold:
            return vbf
        spare = getattr(self, "_vbf_spare", None)
        if spare is not None and spare[0] == bright_field_mask_threshold:
            wanted = spare[1]
        else:
            wanted = self._build_vbf(bright_field_mask_threshold)
        # Demote the outgoing provider, budget permitting. Over budget the spare
        # is dropped and this method is byte for byte the one-slot cache it was.
        if (
            vbf is not None
            and self._vbf_retained_bytes(vbf) <= self._VBF_SPARE_BUDGET_BYTES
        ):
            self._vbf_spare = (threshold, vbf)
        else:
            self._vbf_spare = None
        self.vBF = wanted
        self._vbf_threshold = bright_field_mask_threshold
        return wanted

    def _build_vbf(
        self, bright_field_mask_threshold: float
    ) -> "DatasetVirtualBrightField4DStem":
        return _direct.build_vbf(self, bright_field_mask_threshold)

    @property
    def mean_cbed_tcdf(self) -> torch.Tensor:
        """Mean CBED in the tcDF variant ``array[:n, :n].mean((0, 1))``.

        ``n`` is the number of scan rows (numpy clamps the column slice when
        ``nx < n``).
        """
        return _direct.mean_cbed_tcdf(self)

    def gather_detector_group_means(
        self, groups: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Compute the per-scan-position mean over each group of detector pixels.

        A dataset that reads in blocks takes one sequential pass over its
        source instead. The eager per-group gather is the reference route and
        stays the fallback; ``_detector_group_means_contracted`` is the
        one-pass form for the case the tcDF figure path actually runs, and
        returns ``None`` whenever it does not apply.

        Parameters
        ----------
        groups : list of Tensor
            One ``(M_i, 2)`` integer index tensor per group, each row a
            ``[row, col]`` detector-pixel index.

        Returns
        -------
        list of Tensor
            One ``(ny, nx)`` image per group, on ``self.device`` (NaN for an
            empty group, matching ``mean`` over an empty gather).
        """
        return _direct.detector_group_means(self, groups)


    @property
    def radius_bright_field(self):
        """
        Returns the radius of the bright field (BF) region if available.
        """
        return getattr(self, "_radius_bright_field", None)

    def _bright_field_crop_box(
        self: "Dataset4DStem",
        estimator: Literal["integral", "rayfit"],
        edge_reach: float | None = None,
    ) -> _BrightFieldCropBox:
        r, c = self.bright_field_radius_and_center(estimator, edge_reach=edge_reach)
        r_ceil = np.ceil(r)
        r_int = int(np.ceil(r_ceil)) + 1
        y0_int = int(np.round(c[0]))
        x0_int = int(np.round(c[1]))
        # self.shape (not self._array.shape) so a dataset that reads in blocks
        # (no in-memory array) can reuse this box computation.
        rmax = self.shape[-1] // 2
        r = min(r_int, rmax)
        crop_slice = np.s_[
            :, :, y0_int - r : y0_int + r + 1, x0_int - r : x0_int + r + 1
        ]
        return _BrightFieldCropBox(crop_slice, r, c, y0_int, x0_int, r_ceil)

    def crop_brightfield_(
        self: "Dataset4DStem",
        estimator: Literal["integral", "rayfit"] = "integral",
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4DStem":
        """Crop the dataset to the bright-field region without cloning.

        ``normalize``/``clip_neg_values`` are forwarded to :meth:`crop_` --
        see its docstring; pass ``False`` for both when the crop is a
        measurement-only scratch object (e.g. inside
        ``calibrate_reciprocal_from_bright_field``) that must not mutate this
        dataset's array through the unclonded view.

        Parameters
        ----------
        estimator : {"integral", "rayfit"}
            The disk estimator of :meth:`bright_field_radius_and_center`
            that locates the disk.
        normalize : bool
            Forwarded to :meth:`crop_`.
        clip_neg_values : bool
            Forwarded to :meth:`crop_`.

        Returns
        -------
        Dataset4DStem
            The cropped bright-field dataset, with ``is_bright_field=True``
            and its origin shifted into the cropped coordinate system.
        """
        return _direct.crop_brightfield(self, estimator, normalize, clip_neg_values)





    def _averaged_diffraction_pattern(self) -> Tensor:
        return _direct.averaged_pattern(self)

    def bright_field_radius_and_center(
        self,
        estimator: Literal["integral", "rayfit"] = "integral",
        *,
        edge_reach: float | None = None,
        arc_floor: float | None = None,
    ) -> PupilEstimate:
        """Return the radius and centre ``(y, x)`` of the bright-field disk [pixel].

        Calls :func:`scatterem.data.pupil_gauge.measure_pupil` on the mean
        diffraction pattern. The largest plausible radius is the default of
        that function (0.45 of the shorter detector side), or 1.0 for a
        dataset that is cropped to the bright field (``is_bright_field``).

        Parameters
        ----------
        estimator : {"integral", "rayfit"}
            ``"integral"`` (default) for a complete disk. ``"rayfit"`` for a
            disk with a missing sector or with a part outside the detector.
        edge_reach : float, optional
            Half-width of the edge band [pixel]; None: 2 (integral) or 10 (rayfit).
            A wide band reads a sloped disk top 4 % small (Gd2O3). Raise for soft edges.
        arc_floor : float, optional
            Only for ``"rayfit"``: the smallest fraction of the full circle
            that the accepted edge points must cover.

        Returns
        -------
        PupilEstimate
            ``(rim_radius, rim_center)``: the radius [pixel] and the centre
            ``(y, x)`` [pixel] as a NumPy array of shape ``(2,)``.

        Raises
        ------
        ValueError
            If the mean pattern has no clear disk; see
            :func:`~scatterem.data.pupil_gauge.measure_pupil`.
        """
        return measure_pupil(
            self._averaged_diffraction_pattern(),
            estimator,
            radius_cap=1.0 if self.is_bright_field else 0.45,
            edge_reach=edge_reach,
            arc_floor=arc_floor,
        )

    def bright_field_radius_and_center_(
        self,
        estimator: Literal["integral", "rayfit"] = "integral",
        *,
        edge_reach: float | None = None,
        arc_floor: float | None = None,
    ) -> PupilEstimate:
        """In-place variant: caches the radius as ``self._radius_bright_field``.

        Parameters
        ----------
        estimator : {"integral", "rayfit"}
            See :meth:`bright_field_radius_and_center`.
        edge_reach : float, optional
            See :meth:`bright_field_radius_and_center`.
        arc_floor : float, optional
            See :meth:`bright_field_radius_and_center`.

        Returns
        -------
        PupilEstimate
            ``(rim_radius, rim_center)`` in pixels; the radius is also cached
            as ``self._radius_bright_field``.
        """
        estimate = self.bright_field_radius_and_center(
            estimator, edge_reach=edge_reach, arc_floor=arc_floor
        )
        self._radius_bright_field = estimate.rim_radius
        return estimate

    def calibrate_reciprocal_from_bright_field(
        self,
        semiconvergence_angle: float | None = None,
        estimator: Literal["integral", "rayfit"] = "integral",
        *,
        edge_reach: float | None = None,
    ) -> float:
        """Calibrate the reciprocal-space pixel size ``dk`` from the bright field.

        Sets ``dk = alpha / rBF / wavelength`` and ``self.sampling = (dr, dr,
        dk, dk)`` in place (the setter updates ``self.meta.sampling``). ``rBF``
        comes from the mean pattern of the whole detector
        (:meth:`bright_field_radius_and_center`). A scratch crop (as
        ``crop_brightfield_``, same ``edge_reach``, no normalisation, so the array
        does not change) recentres the detector entries of ``self.origin``, sets
        ``self._radius_bright_field`` and refuses a disk near the detector edge.

        Parameters
        ----------
        semiconvergence_angle : float, optional
            Beam convergence semi-angle in radians. Defaults to
            ``self.meta.semiconvergence_angle``.
        estimator : {"integral", "rayfit"}
            The disk estimator; see :meth:`bright_field_radius_and_center`.
        edge_reach : float, optional
            See :meth:`bright_field_radius_and_center`.

        Returns
        -------
        float
            The calibrated reciprocal-space sampling ``dk``.
        """
        meta = self._require_meta()
        if semiconvergence_angle is None:
            semiconvergence_angle = meta.semiconvergence_angle

        # Corner-origin data give a wrong radius, so a wrong dk, with no error.
        # Check first; `mean_cbed_tcdf` is the mean-CBED accessor (a scan subset
        # is enough: the disk position does not depend on the averaged scans).
        from scatterem.data.origin_check import (
            corner_origin_margin,
            is_corner_origin,
        )

        _mean_cbed = self.mean_cbed_tcdf
        if is_corner_origin(_mean_cbed):
            raise ValueError(
                "the diffraction patterns use the CORNER-origin convention, but "
                "reciprocal calibration (and the whole FF-STEM path) assumes a CENTERED "
                "origin. Left uncaught this fails silently: the bright-field disk "
                "straddles the array corners, so the fitted radius and therefore dk are "
                "wrong, and direct ptychography returns an essentially flat image. Fix "
                "with dataset.recentre_origin(), or fftshift the detector axes first: "
                "torch.fft.fftshift(cube, dim=(-2, -1)) "
                f"(corner-origin margin {corner_origin_margin(_mean_cbed):.1f} px)."
            )

        # The radius comes from the whole detector: a crop at the disk edge has
        # too few pixels outside the disk for the outside level.
        disk = self.bright_field_radius_and_center(estimator, edge_reach=edge_reach)
        rBF = disk.rim_radius
        _direct.crop_brightfield(self, estimator, False, False, edge_reach=edge_reach)

        dk = (semiconvergence_angle / rBF) / meta.wavelength
        dr = self.sampling[0]
        self.sampling = (dr, dr, dk, dk)
        return dk

    @property
    def total_intensity(self) -> float:
        """
        Total intensity of the probe over the dataset.
        """
        if self._total_intensity is None:
            self._total_intensity = self._lazy_total()
        return self._total_intensity

    @total_intensity.setter
    def total_intensity(self, value: float) -> None:
        self._total_intensity = value

    @property
    def mean_probe_intensity(self) -> torch.Tensor:
        """
        Mean intensity of the probe over the dataset.
        """
        return self._array.sum(dim=(-2, -1)).mean()

    @property
    def max_probe_intensity(self) -> torch.Tensor:
        """
        Max intensity of the probe over the dataset.
        """
        return self._array.sum(dim=(-2, -1)).max()

    @property
    def fluence(self) -> float:
        """Calculate total electron fluence (electrons per square Angstrom) from total intensity.

        Returns:
            float: Total electron fluence in electrons per square Angstrom.
        """
        scan_area = float(np.prod(self.sampling[:2] * np.array(self._shape[:2])))
        return self.total_intensity / scan_area

    @property
    def fluence_per_probe(self) -> float | None:
        """Calculate the average electron dose per probe position.

        Returns:
            float: Average dose per probe (unnormalized).
        """
        return self.dose_per_probe_unnormalized



    def crop(  # pyright: ignore[reportIncompatibleMethodOverride]  # by design
        self,
        index: tuple[slice, ...],
        clone: bool = True,
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4DStem":
        """
        Cut a 4D sub-block out of the cube, as a new dataset.

        The ``origin`` moves by the slice starts. ``meta`` is copied (:func:`~scatterem.data.datasets._meta.copy_meta`)
        with the crop's own shape written in: a cropped detector has its own
        ``k_max``, so it cannot share geometry with its source. Fit aberrations
        on the dataset you reconstruct -- a fit on the crop stays on the crop.

        Parameters
        ----------
        index : tuple[slice, ...]
            One slice per axis (scan y, scan x, detector y, detector x).
        clone : bool
            If True, the array is cloned before returning.
        normalize, clip_neg_values : bool
            Forwarded to ``Dataset4DStem.from_array``. When ``clone=False``
            the child's array is an unclonded VIEW into ``self._array`` --
            ``normalize=True``/``clip_neg_values=True`` then mutate the
            parent's storage in place (``self._array /= ...`` / masked
            zeroing runs through the view). Pass ``False`` for a
            measurement-only scratch crop that must not touch the parent.

        Returns
        -------
        Dataset4DStem
            The sub-block, with ``is_cropped`` set.
        """
        array_view = _init_passes.crop_array(  # one Warp pass on CUDA
            self.array, index, clone, normalize, clip_neg_values
        )
        ndim = array_view.ndim

        # The new origin: the old one minus the start of each slice (0 if none).
        starts = [getattr(index[i], "start", None) for i in range(4)]
        new_origin = np.array(self.origin) - np.array(
            [0 if start is None else start for start in starts]
        )

        if ndim != 4:
            raise ValueError("only 4D slices are supported.")
        new = Dataset4DStem.from_array(
            array=array_view,
            name=self.name + str(index),
            origin=new_origin,
            sampling=self.sampling,
            units=self.units,
            signal_units=self.signal_units,
            device=self.device,
            meta=copy_meta(self.meta, shape=array_view.shape, index=index),
            is_cropped=True,
            normalize=normalize,
            clip_neg_values=clip_neg_values,
            copy=not clone,  # a cloned slice is fresh; a view keeps its clone
        )
        return self._carry_record(new)

    def crop_(
        self,
        index: tuple[slice, ...],
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4DStem":
        """
        :meth:`crop` with ``clone=False``: the result shares memory with this cube.

        Parameters
        ----------
        index : tuple[slice, ...]
            One slice per axis, as for :meth:`crop`.
        normalize, clip_neg_values : bool
            Forwarded to :meth:`crop` (which is always called with
            ``clone=False`` here, i.e. an unclonded view -- see its docstring
            for why these default to ``True`` but must be ``False`` for a
            measurement-only scratch crop).

        Returns
        -------
        Dataset4DStem
            The sub-block as a view, with ``is_cropped`` set.
        """
        return self.crop(
            index, clone=False, normalize=normalize, clip_neg_values=clip_neg_values
        )

    @property
    def detector_shape(self) -> NDArray:
        """Shape of the detector (last two) dimensions."""
        return np.array(self._shape[-2:])

    @property
    def dk(self) -> float:
        """Calculate reciprocal space sampling of the detector from a bright field radius estimation.

        Returns:
            float: Reciprocal space sampling of the detector in inverse Angstroms.
        """
        rbf, _ = self.bright_field_radius_and_center()
        meta = self._require_meta()
        return float(meta.semiconvergence_angle / rbf / meta.wavelength)

    @classmethod
    def from_array(
        cls,
        array: Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        meta: "AcquisitionMetadata | None" = None,
        transform_to_amplitudes: bool = False,
        fourier_shift_dim: tuple | None = None,
        normalize: bool = True,
        clip_neg_values: bool = True,
        copy: bool = True,
        *,
        device: torch.device | str | None = None,
        is_cropped: bool = False,
        out_of_core: bool = False,
        ooc_row_block: int = 8,
        g_residency: str = "auto",
        energy: float | None = None,
        semiconvergence_angle: float | None = None,
        scan_step: float | tuple | list | NDArray | None = None,
        reciprocal_step: float | tuple | list | NDArray | None = None,
        rotation: float = 0.0,
        aberrations: Aberrations | None = None,
    ) -> "Dataset4DStem":
        """Build a 4D-STEM dataset (scan y, scan x, detector y, detector x).

        Parameters
        ----------
        array : Any
            The 4D cube: numpy array, tensor, h5py dataset or memory map.
        name : str, optional
            Label. Default: ``"4d dataset"``.
        origin : NDArray or tuple or list or float or int, optional
            Coordinate of index zero per axis. Default: 0, or the array centre
            on the physics path (see ``energy``).
        sampling : NDArray or tuple or list or float or int, optional
            Pitch per axis. Defaults to what ``meta`` describes
            (``meta.sampling``), else ``1.0`` per axis.
        units : list of str or tuple or list, optional
            Unit per axis. Default: the units of ``meta``, else ``"pixels"``.
        signal_units : str
            Unit of the values.
        meta : AcquisitionMetadata, optional
            Acquisition metadata. Built internally from the physics-path
            parameters below when ``None`` and ``energy`` is given.
        transform_to_amplitudes : bool
            Whether to take the square root of the array in place, converting
            recorded intensities to amplitudes.
        fourier_shift_dim : tuple, optional
            Detector dimensions to ``fftshift`` in place on construction.
        normalize : bool
            Whether to divide the array in place by the maximum mean
            diffraction pattern. Default is ``True``. **This rescales the
            stored data**: pass ``normalize=False`` (and
            ``clip_neg_values=False``) whenever raw counts must survive
            byte-exact, e.g. for dose accounting or comparisons against
            another reader.
        clip_neg_values : bool
            Whether to clamp negative values to zero in place. Default is
            ``True``; see the note on ``normalize``.
        copy : bool
            Whether to clone the array before any in-place operations run,
            so that the caller's array is never mutated.
        device : torch.device or str or None, optional
            ``None`` keeps a torch tensor's device; other inputs become CPU.
            Out-of-core: compute device for streamed slabs (``None`` → CPU).
        is_cropped : bool
            Whether the array is the result of a crop.
        out_of_core : bool
            Keep the cube where it is and read it in pieces (a ``MemmapSource``);
            every read converts like the eager path. uint8, uint16 and float16
            cubes stream in a fit.
        ooc_row_block : int
            Scan rows per read of pass 0 and ``iter_scan_blocks`` (out-of-core only).
        g_residency : str
            ``"gpu"`` | ``"vbf_gpu"`` | ``"cpu_pinned"`` | ``"auto"`` -- where
            the streaming vBF provider keeps G (out-of-core only).
        energy : float, optional
            Beam energy in eV. Triggers the physics convenience path: when
            ``meta`` is ``None`` and ``energy`` is given, an ``AcquisitionMetadata``
            is built internally (shape derived from ``array``,
            ``sampling = (dr, dr, dk, dk)`` with ``dr`` from ``scan_step`` and
            ``dk`` from ``reciprocal_step`` -- both scalar or a length-2 pair,
            ``dk`` defaulting to a ``(1.0, 1.0)`` placeholder to be calibrated
            later, e.g. via ``calibrate_reciprocal_from_bright_field``).
            ``origin`` defaults to the array center in this path when not
            given explicitly.
        semiconvergence_angle : float, optional
            Beam convergence semi-angle in radians, used by the physics path.
        scan_step : float or tuple or list or NDArray, optional
            Real-space scan step, used by the physics path.
        reciprocal_step : float or tuple or list or NDArray, optional
            Reciprocal-space pixel size, used by the physics path.
        rotation : float
            Scan rotation in degrees, used by the physics path.
        aberrations : Aberrations, optional
            Aberration coefficients, used by the physics path.

        Returns
        -------
        Dataset4DStem
            The container, with any unset axis metadata filled in from the
            array's rank.
        """
        # np.ndim reads the rank of a tensor, an array or a memory map without
        # a copy; both the eager and the out-of-core path use it.
        _ndim = np.ndim(array)
        _name = f"{_ndim}d dataset" if name is None else name

        if meta is None and energy is not None:
            meta, _built_sampling, _built_origin = _metadata4dstem_from_physics(
                array.shape,
                energy=energy,
                semiconvergence_angle=semiconvergence_angle,
                scan_step=scan_step,
                reciprocal_step=reciprocal_step,
                rotation=rotation,
                aberrations=aberrations,
            )
            sampling = _built_sampling if sampling is None else sampling
            if origin is None:
                origin = _built_origin

        _origin = np.zeros(_ndim) if origin is None else origin
        _s = meta_sampling(meta) if sampling is None else sampling
        _sampling = np.ones(_ndim) if _s is None else _s
        _u = getattr(meta, "units", None) if units is None else units
        _units = ["pixels"] * _ndim if _u is None else _u

        if out_of_core:
            return _direct.out_of_core_dataset(
                cls, array, meta, device=device, normalize=normalize,
                transform_to_amplitudes=transform_to_amplitudes,
                fourier_shift_dim=fourier_shift_dim, clip_neg_values=clip_neg_values,
                row_block=ooc_row_block, g_residency=g_residency,
                labels=(_name, _origin, _sampling, _units, signal_units),
            )  # fmt: skip

        validated, owned = validation.ensure_valid_array_owned(array, device=device)
        return cls(
            array=validated,
            name=_name,
            signal_units=signal_units,
            origin=_origin,
            sampling=_sampling,
            units=_units,
            _token=cls._token,
            meta=meta,
            device=device,
            transform_to_amplitudes=transform_to_amplitudes,
            fourier_shift_dim=fourier_shift_dim,
            normalize=normalize,
            copy=copy and not owned,  # an h5py read is fresh
            clip_neg_values=clip_neg_values,
        )

    def copy(self) -> "Self":
        """Deep-copy the dataset without re-running the construction transforms.

        The inherited :meth:`Dataset.copy` rebuilds through ``from_array``
        with its defaults, which would renormalize/clip the already-processed
        data and drop ``meta`` -- so this override passes the "already
        applied" flags explicitly and carries the bookkeeping over.

        ``meta`` is copied (:func:`~scatterem.data.datasets._meta.copy_meta`,
        a detached snapshot) so geometry updates on the copy never reach the
        parent. Cached derived images (direct-ptychography/tcDF
        results) are NOT copied. Only supported for in-core datasets:
        the public download classes have
        constructor signatures ``from_array`` cannot satisfy.

        Returns
        -------
        Self
            A new dataset of the same class with a cloned array and independent
            ``meta``.
        """
        new = type(self).from_array(
            array=self.array.clone(),
            name=self.name,
            signal_units=self.signal_units,
            origin=self.origin.copy(),
            sampling=self.sampling.copy(),
            units=list(self.units),
            meta=copy_meta(self.meta),
            transform_to_amplitudes=False,  # already applied to the data
            normalize=False,  # data already carries its final scaling
            clip_neg_values=False,
            device=self.device,
        )
        new.transform_to_amplitudes = self.transform_to_amplitudes
        new.probe_index = self.probe_index
        new.fourier_shift_dim = self.fourier_shift_dim
        new.bright_field_mask_threshold = self.bright_field_mask_threshold
        new.is_bright_field = self.is_bright_field
        new.is_cropped = self.is_cropped
        new.dose_per_probe_unnormalized = self.dose_per_probe_unnormalized
        new.dose_per_probe_normalized = self.dose_per_probe_normalized
        new._total_intensity = self._total_intensity
        new.processing_steps = self.processing_steps
        # from_array is typed on the base because it may build an out-of-core
        # dataset; with the in-core arguments above it always builds ``cls``.
        return cast("Self", new)


    @property
    def array(self) -> Tensor:
        """The 4D cube as a tensor; the setter keeps dtype and rank."""
        return self._array

    @array.setter
    def array(self, value: Tensor) -> None:
        new = ensure_valid_array(
            value, dtype=self.dtype, ndim=self.ndim, device=value.device
        )
        sync_meta_shape(self.meta, new.shape, old_shape=getattr(self, "_shape", None))
        # After the sync, which refuses before any change: _shape follows pad/bin.
        self._array, self._shape = new, new.shape

    @property
    def origin(self) -> NDArray:
        """The per-axis origin, i.e. the pixel offset of the array's zero index."""
        return self._origin

    @origin.setter
    def origin(self, value: NDArray | tuple | list | float | int) -> None:
        self._origin = validate_ndinfo(value, self.ndim, "origin")

    @property
    def sampling(self) -> NDArray:
        """The per-axis physical sampling (scan step and detector pixel size)."""
        return self._sampling

    @sampling.setter
    def sampling(self, value: NDArray | tuple | list | float | int) -> None:
        self._sampling = validate_ndinfo(value, self.ndim, "sampling")
        sync_meta_sampling(self.meta, self._sampling)

    @property
    def is_bright_field(self) -> bool:
        """Whether the dataset contains only the bright field region."""
        return self._is_bright_field if hasattr(self, "_is_bright_field") else False

    @is_bright_field.setter
    def is_bright_field(self, value: bool) -> None:
        self._is_bright_field = bool(value)

    @property
    def is_cropped(self) -> bool:
        """Whether the dataset has been cropped."""
        return self._is_cropped if hasattr(self, "_is_cropped") else False

    @is_cropped.setter
    def is_cropped(self, value: bool) -> None:
        self._is_cropped = bool(value)

    @property
    def device(self) -> torch.device:
        """The device the data (or, for a source on disk, its reads) is on."""
        return self._data_device()

    @device.setter
    def device(self, value: torch.device) -> None:
        """Move the data (or redirect reads of a source on disk) to ``value``."""
        self._move_to(value)
        if isinstance(self.meta, AcquisitionMetadata):
            self.meta.device = value

    def _summary_rows(self) -> dict[str, Any]:
        rows = {
            "name": self.name,
            "scan": f"{self.shape[0]} x {self.shape[1]}",
            "detector": f"{self.shape[2]} x {self.shape[3]}",
            "dtype": self.dtype,
            "device": self.device,
            "amplitudes": self.transform_to_amplitudes,
            "dose/probe": "deferred" if self._dose_deferred else self._dose,  # no read
        }
        if self.meta is not None:
            rows["energy (eV)"] = getattr(self.meta, "energy", None)
            rows["semiconv (rad)"] = getattr(self.meta, "semiconvergence_angle", None)
        return rows
