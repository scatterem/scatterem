"""The alternative constructors of :class:`AcquisitionMetadata`.

A mixin, so that ``acquisition.py`` stays within the 500-line module limit
once the class holds its ``geometry`` by composition (M2, 2026-09-28). It holds
no state and is not :class:`~scatterem.utils.store.Serializable`, so it is
absent from the ancestry the store records. Every method builds a new
``AcquisitionMetadata``; ``cls``/``self`` are cast to it for the type checker.
"""

from typing import TYPE_CHECKING, cast

import torch

from scatterem.data.geometry import Geometry

if TYPE_CHECKING:
    from scatterem.data.acquisition import AcquisitionMetadata

__all__: list[str] = []


def _zero_geometry(device: torch.device) -> Geometry:
    return Geometry(
        translation=torch.zeros(2, dtype=torch.float32, device=device),
        rotation=torch.zeros(3, dtype=torch.float32, device=device),
    )


class _AcquisitionConstructors:
    """``with_geometry``, ``from_dataset`` and ``from_config``."""

    def with_geometry(self, geometry: Geometry | None = None) -> "AcquisitionMetadata":
        """A copy of this metadata that carries ``geometry``.

        The copy shares this metadata's field values (tensors and the
        ``aberrations`` object by reference); only ``geometry`` differs.

        Parameters
        ----------
        geometry : Geometry, optional
            Pose of the copy. Default: zero translation ``(2,)`` and zero
            rotation ``(3,)``, float32, on the device of ``vacuum_probe``, else
            of the aberration coefficients, else the CPU. Any geometry this
            metadata already holds is replaced, not kept.

        Returns
        -------
        AcquisitionMetadata
            New instance of this metadata's class.
        """
        meta = cast("AcquisitionMetadata", self)
        if geometry is None:
            if meta.vacuum_probe is not None:
                device = meta.vacuum_probe.device
            elif meta.aberrations is not None:
                device = meta.aberrations.array.device
            else:
                device = torch.device("cpu")
            geometry = _zero_geometry(device)
        return type(meta)(
            energy=meta.energy,
            semiconvergence_angle=meta.semiconvergence_angle,
            sampling=meta.sampling,
            units=meta.units,
            shape=meta.shape,
            rotation=meta.rotation,
            defocus_guess=meta.defocus_guess,
            sample_thickness_guess=meta.sample_thickness_guess,
            vacuum_probe=meta.vacuum_probe,
            aberrations=meta.aberrations,
            slice_thickness=getattr(meta, "slice_thickness", 0.0),
            scan_affine=meta.scan_affine,
            geometry=geometry,
            parent_scan=meta.parent_scan,
            detector_warp=getattr(meta, "detector_warp", None),
            detector_warp_centre=getattr(meta, "detector_warp_centre", None),
        )


