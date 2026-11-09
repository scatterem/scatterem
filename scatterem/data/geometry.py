"""Rigid-body pose of one acquisition: the ``geometry`` of ``AcquisitionMetadata``.

Moved out of ``data_classes.py`` (2026-09-28) so that
:class:`~scatterem.data.acquisition.AcquisitionMetadata` can hold one by
composition without importing the legacy classes.
"""

from dataclasses import dataclass

from torch import Tensor

from scatterem.utils.store import Serializable

__all__ = ["Geometry"]


@dataclass
class Geometry(Serializable):
    """Rigid-body scan/projection geometry (translation and rotation).

    Parameters
    ----------
    translation : Tensor
        Real-space translation, shape ``(2,)``.
    rotation : Tensor
        Euler angles ``(phi, theta, psi)`` in radians, shape ``(3,)``, in the
        convention ``R = Rx(psi) @ Ry(theta) @ Rz(phi)``. ``R`` maps
        lab -> volume.
    """

    SERIAL_FIELDS = ("translation", "rotation")
    translation: Tensor
    rotation: Tensor

    def __str__(self) -> str:
        """Pretty print all class members"""
        return (
            "Geometry:\n"
            f"  translation:    {self.translation.cpu().numpy()}\n"
            f"  rotation:    {self.rotation.cpu().numpy()}"
        )

    def __repr__(self) -> str:
        return self.__str__()
