"""Detector operations on a loaded ``Dataset4DStem``: origin and padding.

``Dataset4DStem`` takes these methods from :class:`DetectorOps`:

* :meth:`~DetectorOps.recentre_origin` moves corner-origin (FFT-shifted)
  patterns to the centred convention that the rest of scatterem expects.
* :meth:`~DetectorOps.pad_detector` zero-pads a centred detector to a size or
  a k-range. The reciprocal sampling ``dk`` does not change.

Each operation adds one entry to ``processing_steps``, so a dataset tells
which of them it went through. ``bin_detector`` adds one entry too. ``copy()``,
``crop``, ``bin_detector`` and the pad-and-taper operations keep
``processing_steps`` and ``fourier_shift_dim``. ``Dataset4DStem.save`` writes
the steps as UTF-8 JSON (the field ``_processing_steps_json``). JSON has no
tuples, so a tuple in a step comes back from ``load`` as a list. A file
written before the field existed loads with no steps. ``recentre_origin`` also
sets ``fourier_shift_dim``, which ``save`` writes.
"""

import json
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch
from torch import Tensor

from scatterem.data.origin_check import corner_origin_margin, is_clearly_corner_origin

from ._meta import KEEP, copy_meta, moved_warp_centre

if TYPE_CHECKING:
    from .dataset_4dstem import Dataset4DStem

__all__ = ["DetectorOps"]

logger = logging.getLogger(__name__)


def _pair(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, Sequence | np.ndarray):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def _jsonable(value: Any) -> Any:
    """Turn a numpy or torch value in a step record into a JSON value."""
    if isinstance(value, Tensor | np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"a processing step holds a {type(value).__name__}")


class DetectorOps:
    """Mixin for ``Dataset4DStem``: operations on the detector axes."""

    #: The operations applied to the data, oldest first. Each entry is a dict
    #: with the key ``"op"`` and the parameters of that operation.
    processing_steps: tuple[dict[str, Any], ...] = ()
    #: ``Dataset4DStem`` saves the steps in this field. Older files lack it.
    SERIAL_OPTIONAL: ClassVar[tuple[str, ...]] = ("_processing_steps_json",)

    @property
    def _processing_steps_json(self) -> np.ndarray:
        """``processing_steps`` as UTF-8 JSON bytes, the form ``save`` writes.

        A uint8 array, not a string attribute: HDF5 limits an attribute to
        64 kB, and a step can hold a long list, for example of scan positions.
        """
        text = json.dumps(list(self.processing_steps), default=_jsonable)
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()

    @_processing_steps_json.setter
    def _processing_steps_json(self, value: np.ndarray) -> None:
        text = np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8")
        self.processing_steps = tuple(json.loads(text))

    def _carry_record(
        self: Any, new: Any, step: dict | None = None, *, dose: bool = False
    ) -> Any:
        """Give ``new`` the ``fourier_shift_dim`` and ``processing_steps`` of this one.

        Parameters
        ----------
        new : Dataset4DStem
            The derived dataset. This method changes it in place.
        step : dict, optional
            A step to add after the steps of this dataset.
        dose : bool, optional
            Also copy the dose values of this dataset. ``from_array`` calculates
            them again from data that can be normalised already, which gives
            wrong values. Use it when the operation keeps the counts per probe.

        Returns
        -------
        Dataset4DStem
            ``new``.
        """
        new.fourier_shift_dim = self.fourier_shift_dim
        new.processing_steps = (*self.processing_steps, *([step] if step else []))
        if dose:
            new.dose_per_probe_unnormalized = self.dose_per_probe_unnormalized
            new.dose_per_probe_normalized = self.dose_per_probe_normalized
            new._total_intensity = self._total_intensity
        return new

    def _derived(
        self: Any,
        array: Tensor,
        origin: np.ndarray,
        step: dict,
        warp_centre: Any = KEEP,
    ) -> "Dataset4DStem":
        """A new dataset over ``array`` with this one's calibration and flags.

        The data already has its final scaling, so nothing is normalised,
        clipped or square-rooted again (as in ``copy``).
        ``warp_centre`` is the moved ``meta.detector_warp_centre``.
        """
        from .dataset_4dstem import Dataset4DStem  # noqa: PLC0415 - imports this module

        new = Dataset4DStem.from_array(
            array,
            name=f"{self.name} ({step['op']})",
            origin=origin,
            sampling=self.sampling.copy(),
            units=list(self.units),
            signal_units=self.signal_units,
            meta=copy_meta(self.meta, shape=array.shape, warp_centre=warp_centre),
            normalize=False,
            clip_neg_values=False,
            copy=False,
            device=self.device,
        )
        for name in ("transform_to_amplitudes", "probe_index"):
            setattr(new, name, getattr(self, name))
        new.bright_field_mask_threshold = self.bright_field_mask_threshold
        new.is_bright_field = self.is_bright_field
        new.is_cropped = self.is_cropped
        return self._carry_record(new, step, dose=True)

    def recentre_origin(self: Any, force: bool = False) -> "Dataset4DStem":
        """Move corner-origin (FFT-shifted) patterns to the centred convention.

        Some cameras and converters store each pattern with the zero frequency
        at pixel ``(0, 0)``: the bright-field disk is split over the four
        corners. scatterem expects it at ``(det_y // 2, det_x // 2)``. This
        method applies ``fftshift`` over the detector axes once, sets the
        detector origin to the centre pixel and records the operation in
        ``fourier_shift_dim`` and ``processing_steps``.

        Parameters
        ----------
        force : bool, optional
            Shift also when :func:`scatterem.data.origin_check.is_clearly_corner_origin`
            does not find a clear corner-origin disk. Default False.

        Returns
        -------
        Dataset4DStem
            A new dataset with centred patterns. ``meta`` is a copy.

        Raises
        ------
        ValueError
            If the mean pattern is not clearly corner-origin and ``force`` is
            False. A second shift would move a centred disk to the corners.
        """
        mean = self._averaged_diffraction_pattern()
        margin = corner_origin_margin(mean)
        if not force and not is_clearly_corner_origin(mean):
            raise ValueError(
                "the mean pattern is not clearly corner-origin (margin "
                f"{margin:.1f} px), so recentre_origin does not shift it; pass "
                "force=True to shift anyway"
            )
        shifted = torch.fft.fftshift(self.array, dim=(-2, -1))
        origin = np.array(self.origin, dtype=float)
        origin[2:] = [s // 2 for s in shifted.shape[-2:]]
        step = {"op": "recentre_origin", "corner_origin_margin_px": margin}
        size = tuple(int(v) for v in shifted.shape[-2:])
        half = [n // 2 for n in size]
        centre = moved_warp_centre(self.meta, half, wrap=size)
        new = self._derived(shifted, origin, step, centre)
        new.fourier_shift_dim = (2, 3)
        logger.info("recentre_origin: fftshift of the detector axes (%s)", step)
        return new

    def pad_detector(
        self: Any,
        size: int | Sequence[int] | None = None,
        *,
        k_max: float | Sequence[float] | None = None,
        unit: Literal["inv_angstrom", "alpha"] = "inv_angstrom",
    ) -> "Dataset4DStem":
        """Zero-pad a centred detector to a larger frame.

        The detector pixel size ``dk`` does not change, so the frame covers a
        larger k-range: ``k_max`` grows by ``size / det``. The centre pixel
        ``det // 2`` moves to ``size // 2``, and the detector origin moves
        with it. Use this when the bright-field disk fills most of the frame,
        for example after a crop at the detector edge:
        ``bright_field_radius_and_center`` refuses a disk radius above 45 %
        of the shorter frame side.

        Parameters
        ----------
        size : int or sequence of int, optional
            New detector shape ``(det_y, det_x)``, or one int for both.
        k_max : float or sequence of float, optional
            Smallest half-width of the new frame, per axis or one value. The
            new size per axis is ``2 * ceil(k_max / dk)``. Give either
            ``size`` or ``k_max``.
        unit : {"inv_angstrom", "alpha"}, optional
            Unit of ``k_max``: 1/Angstrom (default), or the convergence
            semi-angle ``meta.semiconvergence_angle / meta.wavelength``.
            Both need a calibrated ``dk`` (``sampling[2:]``). This ``"alpha"`` is
            nominal; ``annular_detector_mask(units="bf_radius")`` uses the
            fitted bright-field radius instead.

        Returns
        -------
        Dataset4DStem
            A new dataset with the padded detector. ``meta`` is a copy with the
            new shape.

        Raises
        ------
        ValueError
            If neither or both of ``size`` and ``k_max`` are given, if the new
            frame is smaller than the detector, or if ``unit`` is unknown.
        """
        if (size is None) == (k_max is None):
            raise ValueError("give exactly one of size and k_max")
        n_y, n_x = (int(s) for s in self.shape[-2:])
        if k_max is not None:
            p_y, p_x = self._k_max_to_size(k_max, unit)
        else:
            p_y, p_x = (int(v) for v in _pair(size))  # pyright: ignore[reportArgumentType]
        if p_y < n_y or p_x < n_x:
            raise ValueError(f"pad size {(p_y, p_x)} is smaller than {(n_y, n_x)}")
        top, left = p_y // 2 - n_y // 2, p_x // 2 - n_x // 2
        padding = (left, p_x - n_x - left, top, p_y - n_y - top)
        padded = torch.nn.functional.pad(self.array, padding)
        origin = np.array(self.origin, dtype=float)
        origin[2:] += (top, left)
        step = {"op": "pad_detector", "from": (n_y, n_x), "to": (p_y, p_x)}
        step["offset"] = (top, left)
        centre = moved_warp_centre(self.meta, (top, left))
        return self._derived(padded, origin, step, centre)

    def _k_max_to_size(
        self: Any, k_max: float | Sequence[float], unit: str
    ) -> tuple[int, int]:
        k_y, k_x = _pair(k_max)
        if unit == "alpha":
            meta = self._require_meta()
            scale = float(meta.semiconvergence_angle) / float(meta.wavelength)
            k_y, k_x = k_y * scale, k_x * scale
        elif unit != "inv_angstrom":
            raise ValueError(f"unit must be 'inv_angstrom' or 'alpha'; got {unit!r}")
        dk_y, dk_x = (float(v) for v in self.sampling[2:4])
        # The small tolerance stops float error from adding a pixel pair.
        return (
            2 * math.ceil(k_y / dk_y - 1e-9),
            2 * math.ceil(k_x / dk_x - 1e-9),
        )
