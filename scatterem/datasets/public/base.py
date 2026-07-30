"""Base class for downloadable single-tilt 4D-STEM datasets.

Unlike :class:`~scatterem.datasets.scanning_diffraction.PublicScanningDiffractionDataset`
(which extends ``Dataset4dstemTomo`` and attaches a ``Metadata4D`` for the
iterative tomography models), this base produces a plain
:class:`~scatterem.utils.data.datasets.Dataset4dstem` carrying a
``Metadata4dstem`` -- what the direct-ptychography / tilt-corrected-dark-field /
fused-full-field reconstruction methods expect.
"""

import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import DTypeLike

from scatterem.datasets.utils import check_integrity, download_url, zenodo_file_url
from scatterem.utils.data.datasets import (
    Dataset4dstem,
    _metadata4dstem_from_physics,
)

#: Class attributes every concrete subclass must set to a non-None value
#: before it can be instantiated (checked in __init__; see
#: ``_check_subclass_contract``).
_REQUIRED_CONSTANTS = (
    "zenodo_record_id",
    "energy",
    "semiconvergence_angle",
    "scan_step",
)


class PublicDataset4dstem(Dataset4dstem):
    """A published 4D-STEM dataset that downloads itself and IS a ``Dataset4dstem``.

    Subclasses declare the Zenodo record, the files (with md5 checksums) and the
    acquisition constants, and implement :meth:`_load_array`.

    The cache layout is FLAT: files live directly in ``root``, not in
    ``root/<ClassName>/raw``. Datasets published in a single record have unique
    file names, and a flat layout means an existing local copy of the data is
    reused as-is rather than re-downloaded.

    Args:
        root: Directory holding (or to receive) the raw files.
        download: Fetch any missing/corrupt file from Zenodo.
        device: Torch device for the data array.
        calibrate: Run ``calibrate_reciprocal_from_bright_field()`` after
            construction, replacing the placeholder ``dk``. All the paper's
            figure scripts do this, so it defaults to True. If the bright-field
            disk can't be measured (e.g. it sits near/at the detector edge),
            the measurement yields a non-finite/non-positive ``dk``, or the
            measured radius is implausibly large for a real bright-field disk
            (see ``_MAX_PLAUSIBLE_RADIUS_FRACTION``), the dataset is still
            constructed with the placeholder ``dk`` and a warning is raised
            rather than propagating the error -- pass ``calibrate=False`` to
            skip the attempt outright.
        **dataset_kwargs: Forwarded to ``Dataset4dstem.__init__`` (e.g.
            ``normalize``, ``clip_neg_values``, ``transform_to_amplitudes``).
    """

    zenodo_record_id: ClassVar[Optional[str]] = None
    #: ``[(filename, md5), ...]`` -- every file needed to load this dataset.
    resources: ClassVar[list[tuple[str, str]]] = []
    energy: ClassVar[Optional[float]] = None  # eV
    semiconvergence_angle: ClassVar[Optional[float]] = None  # rad
    scan_step: ClassVar[Optional[float]] = None  # Angstrom
    rotation: ClassVar[float] = 0.0  # deg
    #: Host-side dtype cast applied to the loaded array (None = leave as read).
    host_dtype: ClassVar[DTypeLike] = None
    reference: ClassVar[str] = ""

    def __init__(
        self,
        root: Union[str, Path],
        download: bool = False,
        device: Union[str, torch.device] = "cpu",
        calibrate: bool = True,
        **dataset_kwargs: Any,
    ) -> None:
        self._check_subclass_contract()
        self.root = Path(os.path.expanduser(str(root)))

        # Single md5 pass over the resources declared missing/corrupt; if a
        # download is requested, fetch exactly that subset (download_url
        # verifies each fetched file itself and raises on mismatch, so no
        # second verification pass is needed) and trust the result rather than
        # re-hashing everything again.
        missing = self._missing_resources()
        if missing and download:
            self._download_resources([(fn, md5) for fn, md5, _reason in missing])
            missing = []

        if missing:
            detail = ", ".join(f"{fn} ({reason})" for fn, _md5, reason in missing)
            raise RuntimeError(
                f"{type(self).__name__}: missing or corrupt data file(s) in "
                f"{self.raw_folder}: {detail}. Pass download=True to fetch them "
                f"from https://zenodo.org/records/{self.zenodo_record_id}"
            )

        array = self._load_array()
        if self.host_dtype is not None:
            array = array.astype(self.host_dtype)

        physics = _metadata4dstem_from_physics(
            array.shape,
            energy=self.energy,
            semiconvergence_angle=self.semiconvergence_angle,
            scan_step=self.scan_step,
            rotation=self.rotation,
        )

        Dataset4dstem.__init__(
            self,
            array=array,
            name=type(self).__name__,
            origin=physics.origin,
            sampling=physics.sampling,
            units=physics.meta.units,
            meta=physics.meta,
            device=device,
            _token=type(self)._token,
            **dataset_kwargs,
        )

        if calibrate:
            self._calibrate_or_warn(placeholder_sampling=physics.sampling)

    # --- data access hook ---------------------------------------------------
    def _load_array(self) -> np.ndarray:
        """Read (and repair) the raw files into a ``(ny, nx, M, M)`` array."""
        raise NotImplementedError

    # --- construction guards -------------------------------------------------
    def _check_subclass_contract(self) -> None:
        """Fail fast with a clear message if a subclass forgot a constant.

        Without this, a subclass that forgets e.g. ``energy`` constructs fine
        (every field defaults to ``None``/``[]``) and only breaks later, deep
        inside ``calibrate_reciprocal_from_bright_field`` or ``_load_array``,
        with a confusing ``TypeError``. An empty ``resources`` is equally
        silent: ``all(...)`` over zero items is ``True``, so
        ``_missing_resources`` would report nothing missing and every
        integrity check would be skipped.
        """
        missing_attrs = [
            name for name in _REQUIRED_CONSTANTS if getattr(self, name, None) is None
        ]
        if not self.resources:
            missing_attrs.append("resources")
        if missing_attrs:
            raise TypeError(
                f"{type(self).__name__} is missing required class attribute(s): "
                f"{', '.join(missing_attrs)}. Subclasses of PublicDataset4dstem "
                "must set these before they can be instantiated."
            )

    #: A real bright-field disk can't have a radius larger than this fraction
    #: of the detector's shorter side -- validated against the four datasets
    #: this base ships for: Gd2O3 (112 px, ratio 0.17), carbon (96 px, 0.25),
    #: Co3O4 (64 px, 0.22) and Au low-dose (96 px after crop, 0.17), all
    #: comfortably below; a uniform/featureless detector (no real disk edge)
    #: measures ~0.59.
    _MAX_PLAUSIBLE_RADIUS_FRACTION = 0.45

    def _calibrate_or_warn(self, placeholder_sampling: Sequence[float]) -> None:
        """Best-effort ``calibrate_reciprocal_from_bright_field()``.

        The underlying bright-field crop/measurement isn't robust to a disk
        that sits near/at the detector edge (an off-centre crop box can clip
        to an empty slice, raising ``RuntimeError``/``ValueError`` deep inside
        torch) or to genuinely featureless data: that case doesn't crash, but
        every other failure mode here warns loudly while this one would
        silently produce a plausible-looking, physically wrong ``dk`` that
        propagates uncaught into direct ptychography/tcDF/FFF. Caught via the
        one unambiguous tell available: a real disk's radius can't exceed the
        detector half-width, and in practice sits well under it (see
        ``_MAX_PLAUSIBLE_RADIUS_FRACTION``). Since all of this runs inside
        ``__init__`` after a potentially multi-GB load, a raised exception
        here would make an otherwise-valid dataset unconstructible -- warn and
        keep the placeholder ``dk`` instead.
        """
        try:
            dk = self.calibrate_reciprocal_from_bright_field()
            if not np.isfinite(dk) or dk <= 0:
                raise ValueError(f"non-finite or non-positive dk ({dk!r})")
            rBF = self.radius_bright_field
            detector_shape = tuple(int(s) for s in self.detector_shape)
            if rBF is not None and rBF > self._MAX_PLAUSIBLE_RADIUS_FRACTION * min(
                detector_shape
            ):
                raise ValueError(
                    f"implausible bright-field radius (rBF={rBF!r} px on a "
                    f"{detector_shape} detector) -- a real disk can't occupy "
                    "this much of the detector; this is either a broken "
                    "acquisition/threshold, or the data is already cropped "
                    "to its bright-field disk"
                )
        except torch.cuda.OutOfMemoryError:
            # torch.cuda.OutOfMemoryError subclasses RuntimeError, so it would
            # otherwise be swallowed by the broad clause below and the
            # placeholder dk kept -- clause order (this must come first) is
            # what makes it win. A GPU resource failure is not a calibration
            # failure: it must crash loudly rather than silently degrade dk
            # to a physically wrong (1.0, 1.0), which would then run
            # undetected through aberration fitting / SSB / tcDF / FFF.
            raise
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            self.sampling = placeholder_sampling
            warnings.warn(
                f"{type(self).__name__}: automatic bright-field calibration "
                f"failed ({exc!r}). dk is left at its (1.0, 1.0) placeholder "
                "-- inspect the data and call "
                "calibrate_reciprocal_from_bright_field() manually.",
                # stacklevel=3: __init__ (frame 2) calls this method (frame 1,
                # the warn() call itself); the common case has no subclass
                # __init__ override, so frame 3 is the user's construction
                # line. Approximate if a subclass does add its own __init__.
                stacklevel=3,
            )

    # --- cache management ---------------------------------------------------
    @property
    def raw_folder(self) -> Path:
        """Flat cache directory -- the files live directly in ``root``."""
        return self.root

    def _missing_resources(self) -> list[tuple[str, str, str]]:
        """``[(filename, md5, reason)]`` for resources not present with the
        declared md5 -- one md5 pass per file. ``reason`` distinguishes
        "missing" from "corrupt" using ``Path.is_file()`` (a stat, not a
        hash), so it's free information, not an extra pass.
        """
        missing = []
        for filename, md5 in self.resources:
            fpath = self.raw_folder / filename
            if not check_integrity(fpath, md5):
                reason = "missing" if not fpath.is_file() else "corrupt (md5 mismatch)"
                missing.append((filename, md5, reason))
        return missing

    def download(self) -> None:
        """Fetch every declared resource from Zenodo.

        Argument-free by design: the three sibling public-dataset bases
        (``chen2021.py``, ``sha2022.py``, ``you2024.py``) all declare
        ``def download(self) -> None``, and subclasses following that
        established pattern would ``TypeError`` on construction if this
        signature required an argument. ``__init__`` uses the private
        ``_download_resources`` instead so it can fetch only the subset it
        found missing/corrupt.
        """
        self._download_resources(self.resources)

    def _download_resources(self, resources: list[tuple[str, str]]) -> None:
        """Fetch exactly ``resources`` from Zenodo.

        ``download_url`` already no-ops for a file whose md5 matches -- but
        that no-op still costs a full md5 pass over the file, so ``__init__``
        passes a narrower list (just what it found missing) to skip
        re-hashing files already known to be valid.
        """
        os.makedirs(self.raw_folder, exist_ok=True)
        for filename, md5 in resources:
            download_url(
                zenodo_file_url(self.zenodo_record_id, filename),
                root=self.raw_folder,
                filename=filename,
                md5=md5,
            )

    # --- pretty-printing -----------------------------------------------------
    def _summary_rows(self) -> dict[str, Any]:
        rows = super()._summary_rows()
        rows["root"] = self.root
        if self.reference:
            rows["reference"] = self.reference
        if self.zenodo_record_id:
            rows["data"] = (
                f"https://zenodo.org/records/{self.zenodo_record_id} (CC-BY-4.0)"
            )
        return rows
