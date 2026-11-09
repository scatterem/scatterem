"""
Diffraction-space calibration from the scanned centre of mass.

Two things happen here. First, the centre of mass of every diffraction
pattern is measured and its slow drift across the scan (the descan) is fitted
away with a smooth surface -- see :func:`_calculate_intensities_center_of_mass`
and the ``plane`` / ``parabola`` / ``bezier_two`` models it can fit. Second,
the residual CoM field is used to recover the rotation between the scan axes
and the detector axes, by exploiting the fact that a correctly oriented field
is irrotational -- see :func:`find_scan_rotation`.

Everything is torch-native: functions accept and return ``torch.Tensor``,
take a ``device`` argument (default: the input tensor's device, else CPU), and
the angle sweep and the surface fits are batched rather than looped.

Split out of one former flat module (995 lines). The cut follows
the pipeline, each stage importing only the one before it:

    _common.py     ``_to_tensor`` and the device rule
    _surfaces.py   the three surface models and :func:`fit_2D`
    _origin.py     :func:`fit_origin` -- fit and subtract the descan
    _com.py        measure the CoM across the scan
    _rotation_ops.py     curl, divergence and the angle helpers
    _rotation_solver.py  :func:`find_scan_rotation` and its result records
    _rotation_sign.py    the 180° test
    _rotation_fused.py   the optional Triton evaluation of the objective


The matplotlib figures of the CoM fields and of the rotation search are not
here: they are :mod:`scatterem.vis.rotation_figures`, since this data-layer
package does not plot.

Every name the flat module defined is re-exported here, private ones included:
``bright_field.py`` and the calibration test modules import
``_calculate_intensities_center_of_mass`` and more underscore names by that
path.

What does NOT survive: the flat module also carried its own imports as
attributes (``calibration.np``, ``calibration.torch``, ``calibration.plt``).
Those are gone -- reach for the real module instead. Nothing in the tree used
them.
"""

from ._com import (
    _calculate_intensities_center_of_mass as _calculate_intensities_center_of_mass,
)
from ._common import _to_tensor as _to_tensor
from ._origin import _ORIGIN_SURFACES as _ORIGIN_SURFACES
from ._origin import fit_origin as fit_origin
from ._rotation_solver import RotationSearch as RotationSearch
from ._rotation_solver import ScanRotation as ScanRotation
from ._rotation_solver import find_scan_rotation as find_scan_rotation
from ._surfaces import _bernstein_quadratic as _bernstein_quadratic
from ._surfaces import _fit_linear_model as _fit_linear_model
from ._surfaces import _fit_nonlinear_model as _fit_nonlinear_model
from ._surfaces import _linear_design_matrix as _linear_design_matrix
from ._surfaces import _make_xy_flat as _make_xy_flat
from ._surfaces import bezier_two as bezier_two
from ._surfaces import fit_2D as fit_2D
from ._surfaces import parabola as parabola
from ._surfaces import plane as plane


__all__ = [
    "RotationSearch",
    "ScanRotation",
    "bezier_two",
    "fit_2D",
    "find_scan_rotation",
    "fit_origin",
    "parabola",
    "plane",
]
