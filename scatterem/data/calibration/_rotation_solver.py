"""Rotation between scan and detector, found with the CoM field.

In the phase-object approximation the centre of mass (CoM) of each
diffraction pattern is proportional to the gradient of the projected
potential at the probe position (Waddell and Chapman 1979; Close et al. 2015;
Lazić, Bosch and Lazar 2016; Müller-Caspary et al. 2017). So the true CoM
field is free of curl. A detector turned by an angle against the scan turns
every CoM vector and adds curl. This module turns the measured field back over
a set of trial angles and takes the angle with the least curl, or with the
largest divergence (Ophus 2019). Two handedness hypotheses are tested: the
field as measured, and the field with its two components exchanged. The
180° ambiguity is resolved in :mod:`._rotation_sign`.

Conventions: axis 0 is "x", axis 1 is "y". A field is the pair
``(comp_a0, comp_a1)``. The active rotation is
``x' = x cos θ − y sin θ``, ``y' = x sin θ + y cos θ``. All angles in the API
are in degrees and carry the suffix ``_deg``.
"""

from __future__ import annotations

import importlib
import logging
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import numpy as np
from torch import Tensor

from scatterem.data.calibration._rotation_ops import (
    _angle_trig,
    _as_angle_grid,
    _as_scan_spacing,
    _curves_from_moments,
    _moments,
    _validate_components,
    com_field_tensors,
    curl_2d,
    divergence_2d,
    wrap_angle_deg,
)
from scatterem.data.calibration._rotation_sign import resolve_half_turn

__all__ = [
    "DEFAULT_ANGLE_GRID_DEG",
    "ObjectiveCurves",
    "RotationSearch",
    "ScanReflectionWarning",
    "ScanRotation",
    "curl_2d",
    "divergence_2d",
    "find_scan_rotation",
    "rotation_objective",
    "wrap_angle_deg",
]

logger = logging.getLogger(__name__)

#: Default trial angles: [-90°, 90°) in steps of 0.1°. The objective has
#: period 180°, so one period covers every rotation; the 180° test supplies
#: the other half-turn.
DEFAULT_ANGLE_GRID_DEG = (np.arange(1800) - 900) / 10.0

_FUSED_MODULE = "scatterem.data.calibration._rotation_fused"
_OBJECTIVES = ("curl", "divergence")
_HYPOTHESES = ("axes_kept", "axes_exchanged")


class ScanReflectionWarning(UserWarning):
    """The CoM field matches a mirror image of the scan."""


class ObjectiveCurves(NamedTuple):
    """Objective curves of both handedness hypotheses.

    Attributes
    ----------
    values : Tensor
        float64, shape ``(2, A)``, on the device of the field. Row 0 is the
        field as measured, row 1 the field with exchanged components.
    evaluator : str
        ``"tensor_ops"`` or ``"gpu_kernel"``: the implementation that ran.
    """

    values: Tensor
    evaluator: str


@dataclass(frozen=True)
class RotationSearch:
    """Record of the angle search, for :func:`scatterem.vis.rotation_figures.angle_search_figure`.

    Attributes
    ----------
    curve_kind : str
        ``"curl"`` or ``"divergence"``.
    angle_grid_deg : numpy.ndarray
        float64 ``(A,)``: the trial angles as given.
    hypothesis_curves : dict of str to numpy.ndarray
        ``"axes_kept"`` and/or ``"axes_exchanged"``, float64 ``(A,)``. A hypothesis
        that was not evaluated has no key.
    search_angle_deg : float
        The trial angle at the extremum of the chosen hypothesis.
    scan_turn_deg : float
        The returned angle, equal to ``ScanRotation.scan_turn_deg``.
    axes_exchanged : bool
        The chosen hypothesis.
    rotated_180 : bool
        True when the 180° test added a half-turn.
    sign_test : str
        ``"kept"``, ``"flipped"``, ``"undecided"`` or ``"off"``.
    potential_skewness : float or None
        Skewness of the integrated potential; None when the test did not run.
    evaluator : str
        ``"tensor_ops"`` or ``"gpu_kernel"``.
    """

    curve_kind: str
    angle_grid_deg: np.ndarray
    hypothesis_curves: dict
    search_angle_deg: float
    scan_turn_deg: float
    axes_exchanged: bool
    rotated_180: bool
    sign_test: str
    potential_skewness: float | None
    evaluator: str


@dataclass(frozen=True)
class ScanRotation:
    """Result of :func:`find_scan_rotation`.

    Attributes
    ----------
    scan_turn_deg : float
        Total rotation applied to the (possibly swapped) field, in degrees,
        in the half-open interval ``[-180, 180)``. It includes the half-turn
        of the 180° test.
    centroid_a0, centroid_a1 : Tensor
        The corrected field on the resolved device. dtype float64 if either
        input was float64, else float32.
    axes_exchanged : bool
        True when the components were exchanged before the rotation.
    rotated_180 : bool
        True when the 180° test added a half-turn. Then ``scan_turn_deg`` holds
        the half-turn and the corrected field is the negated field.
    search : RotationSearch or None
        The search record when the caller asked for it and a search ran.
    """

    scan_turn_deg: float
    centroid_a0: Tensor
    centroid_a1: Tensor
    axes_exchanged: bool
    rotated_180: bool
    search: RotationSearch | None


def rotation_objective(
    comp_a0,
    comp_a1,
    angle_grid_deg,
    *,
    curve_kind: str = "curl",
    scan_spacing=(1.0, 1.0),
    evaluator: str = "by_capability",
) -> ObjectiveCurves:
    """Objective curves over the trial angles, for both handedness hypotheses.

    For one hypothesis and one angle θ the statistic is
    ``S_curl(θ) = Σ κ_θ² / Σ ‖J‖²_F`` (or ``S_div`` with the divergence δ_θ),
    summed over all cells. ``J`` is the Jacobian of the field. Reasons: the
    squares make it invariant to ``f → −f``; the ratio is invariant to the
    scale of the field and lies in ``[0, 2]``; the common denominator makes
    the two hypotheses comparable. The rotation identities
    ``κ_θ = cos θ κ + sin θ δ`` give the closed form
    ``S_curl(θ) = [(K+D)/2 + (K−D)/2 cos 2θ + M sin 2θ] / F``. So one pass
    over the scan gives all angles; there is no loop over the angles and no
    array of size ``A × n0·n1``. The period is exactly 180°. Within one
    hypothesis the least curl and the largest divergence fall on one angle
    (Ophus 2019). All sums are in float64 for every input dtype. If
    ``F = 0`` (constant field), the curve is 0.0.

    Parameters
    ----------
    comp_a0, comp_a1 : Tensor
        Real tensors ``(n0, n1)``, ``n0, n1 >= 2``.
    angle_grid_deg : array_like
        1-D finite trial angles in degrees, at least one.
    curve_kind : {"curl", "divergence"}, optional
    scan_spacing : float or (float, float), optional
        As in :func:`curl_2d`. Only the ratio ``s0/s1`` changes the angle.
    evaluator : {"by_capability", "tensor_ops"}, optional
        ``"by_capability"`` tries the fused Triton kernel of
        :mod:`._rotation_fused` and falls back to torch; ``"tensor_ops"``
        always runs the reference implementation.

    Returns
    -------
    ObjectiveCurves
        ``values`` float64 ``(2, A)`` on the device of ``comp_a0``; row 0 as
        measured, row 1 swapped. ``evaluator`` names the implementation.

    Raises
    ------
    ValueError
        For bad shapes, dtypes, angles, ``curve_kind`` or ``evaluator``.
    """
    if curve_kind not in _OBJECTIVES:
        raise ValueError(f"curve_kind must be one of {_OBJECTIVES}, got {curve_kind!r}")
    if evaluator not in ("by_capability", "tensor_ops"):
        raise ValueError(
            f"evaluator must be 'by_capability' or 'tensor_ops', got {evaluator!r}"
        )
    s0, s1 = _as_scan_spacing(scan_spacing)
    fx, fy = _validate_components(comp_a0, comp_a1)
    if fx.ndim != 2:
        raise ValueError(f"fields must be 2-D, got shape {tuple(fx.shape)}")
    angles = _as_angle_grid(angle_grid_deg)
    if evaluator == "by_capability" and fx.is_cuda:
        try:
            fused = importlib.import_module(_FUSED_MODULE)
        except ImportError:  # pragma: no cover - module import never needs Triton
            fused = None
        if fused is not None:
            values = fused.rotation_curves_triton(
                comp_a0, comp_a1, angles, curve_kind=curve_kind, scan_spacing=(s0, s1)
            )
            if values is not None:
                logger.debug("rotation objective: fused triton kernel ran")
                return ObjectiveCurves(values, "gpu_kernel")
    c2, s2 = _angle_trig(angles, fx.device)
    values = _curves_from_moments(_moments(fx, fy, s0, s1), c2, s2, curve_kind)
    logger.debug("rotation objective: torch reference ran")
    return ObjectiveCurves(values, "tensor_ops")


def _rotate(fx: Tensor, fy: Tensor, angle_in_deg: float) -> tuple[Tensor, Tensor]:
    """Active rotation of the field by ``angle_in_deg``."""
    t = math.radians(angle_in_deg)
    c, s = math.cos(t), math.sin(t)
    return c * fx - s * fy, s * fx + c * fy


def _extremum(curve: np.ndarray, curve_kind: str) -> int:
    """Index of the extremum; on exact ties the first index wins."""
    return int(np.argmin(curve) if curve_kind == "curl" else np.argmax(curve))


def _better(v_swapped: float, v_measured: float, curve_kind: str) -> bool:
    """True when the swapped extremum is strictly better (ties keep the data)."""
    return v_swapped < v_measured if curve_kind == "curl" else v_swapped > v_measured


def _check_options(curve_kind, fixed_axes_exchanged, fixed_angle_deg) -> None:
    """Raise ``ValueError`` for an invalid option of :func:`find_scan_rotation`."""
    if curve_kind not in _OBJECTIVES:
        raise ValueError(f"curve_kind must be one of {_OBJECTIVES}, got {curve_kind!r}")
    if fixed_axes_exchanged not in (None, True, False):
        raise ValueError("fixed_axes_exchanged must be None, True or False")
    if fixed_angle_deg is not None and not math.isfinite(float(fixed_angle_deg)):
        raise ValueError("fixed_angle_deg must be finite")


def _search(objective, angles: np.ndarray, hyps: list[bool], curve_kind: str):
    """Grid search: the angle, the handedness, the curves, extrema and evaluator.

    ``objective(angles)`` is :func:`rotation_objective` on the field. Each
    hypothesis takes its extremum; with two hypotheses the better extremum
    wins (:func:`_better`).
    """
    res = objective(angles)
    values, evaluator_used = res.values.cpu().numpy(), res.evaluator
    curves = {_HYPOTHESES[int(h)]: values[int(h)].copy() for h in hyps}
    best = {h: _extremum(values[int(h)], curve_kind) for h in hyps}
    axes_exchanged = hyps[0]
    if len(hyps) == 2:
        v0, v1 = values[0, best[False]], values[1, best[True]]
        axes_exchanged = _better(v1, v0, curve_kind)
    extrema = {_HYPOTHESES[int(h)]: float(values[int(h), best[h]]) for h in hyps}
    search_angle = float(angles[best[axes_exchanged]])
    return search_angle, axes_exchanged, curves, extrema, evaluator_used


def _handedness_at(objective, angle: float, hyps: list[bool], curve_kind: str) -> bool:
    """The handedness at a fixed angle: the better of two hypotheses, or the one."""
    if len(hyps) == 1:
        return hyps[0]
    v = objective([angle]).values[:, 0].cpu().numpy()
    logger.debug("objective at fixed angle: as measured %r, swapped %r", v[0], v[1])
    return _better(v[1], v[0], curve_kind)


def _report(quiet: bool, decision: tuple, found_reflection: bool) -> None:
    """Log the decision; warn for an inconclusive 180° test or a reflection.

    ``decision`` is ``(angle, search_angle, extrema, axes_exchanged,
    sign_test, third_moment)``.
    """
    angle, search_angle, extrema, axes_exchanged, sign_test, third_moment = decision
    if sign_test == "undecided" and not quiet:
        logger.warning(
            "The 180° test was inconclusive (skewness %.3g); the angle may "
            "be off by 180°.",
            third_moment,
        )
    logger.log(
        logging.DEBUG if quiet else logging.INFO,
        "scan rotation %.4g° (search %.4g°, extrema %s, swapped %s, 180° test %s)",
        angle,
        search_angle,
        extrema,
        axes_exchanged,
        sign_test,
    )
    if found_reflection and not quiet:
        warnings.warn(
            "The CoM field matches a mirror image of the scan: the solver had "
            "to exchange the CoM components. No rotation can undo a "
            "reflection. Fix the detector orientation in your data (exchange "
            "or negate one CoM axis) and run the solver again.",
            ScanReflectionWarning,
            stacklevel=3,
        )


def find_scan_rotation(
    centroid_a0,
    centroid_a1,
    *,
    run_device=None,
    quiet: bool = False,
    angle_grid_deg=None,
    fixed_angle_deg: float | None = None,
    fixed_axes_exchanged: bool | None = None,
    resolve_180: bool = True,
    curve_kind: str = "curl",
    scan_spacing=(1.0, 1.0),
    keep_search: bool = False,
    evaluator: str = "by_capability",
) -> ScanRotation:
    """Find the rotation between scan and detector from the CoM field.

    The solver evaluates the objective of :func:`rotation_objective` over the
    trial angles, picks the extremum (minimum for curl, maximum for
    divergence) and decides the handedness: the hypothesis with the better
    extremum wins. On an exact tie the field stays as measured, because a
    reflection is a strong claim that needs evidence. On exact ties within a
    curve the first angle in the caller's order wins. Then the 180° test of
    :func:`._rotation_sign.resolve_half_turn` decides the sign.

    The solver reports with ``logging`` (INFO for the decision, WARNING for
    an inconclusive 180° test) and with :class:`ScanReflectionWarning` for a
    found reflection: only the caller can fix that, by a change to the data,
    and a warnings filter can turn it into an error. ``quiet=True`` reports
    nothing above DEBUG. The solver draws no figure.

    Parameters
    ----------
    centroid_a0, centroid_a1 : Tensor or numpy.ndarray
        CoM components along axis 0 and axis 1: 2-D, real, one shape, at
        least ``2 × 2``, finite.
    run_device : str or torch.device, optional
        Device rule: an explicit device wins; else the device of the first
        tensor input; else the CPU.
    quiet : bool, optional
        No log records above DEBUG and no warnings.
    angle_grid_deg : array_like, optional
        Trial angles in degrees, 1-D, finite, any order. Default
        :data:`DEFAULT_ANGLE_GRID_DEG`, ``[-90, 90)`` in steps of 0.1°: one
        period of the objective, with a grid error of at most 0.05°.
    fixed_angle_deg : float, optional
        Skip the search and use this angle. The result angle is this angle
        wrapped into ``[-180, 180)``. No 180° test runs, and no record is
        returned (``search`` is None). If ``fixed_axes_exchanged`` is None, the two
        hypotheses are compared at this angle.
    fixed_axes_exchanged : bool, optional
        Fix the handedness and skip its test.
    resolve_180 : bool, optional
        Run the 180° test. Default True: without it the angle is ambiguous by
        180°, which inverts the sign of every phase downstream.
    curve_kind : {"curl", "divergence"}, optional
        Default ``"curl"``: its target is 0 for an ideal field, so its
        minimum tells how good the fit is.
    scan_spacing : float or (float, float), optional
        Scan step along axis 0 and axis 1. Only the ratio matters.
    keep_search : bool, optional
        Also return the search record. The other outputs are bit-identical
        with and without it.
    evaluator : {"by_capability", "tensor_ops"}, optional
        Implementation of the objective, see :func:`rotation_objective`.

    Returns
    -------
    ScanRotation
        ``scan_turn_deg`` in ``[-180, 180)``: the total rotation, including a
        half-turn from the 180° test. The corrected field is
        ``R(scan_turn_deg) P^h f``: the half-turn is applied as an exact
        negation. ``rotated_180`` flags the half-turn.

    Raises
    ------
    ValueError
        If the inputs are not two finite real 2-D arrays of one shape of at
        least ``2 × 2``, or an option is invalid.
    """
    _check_options(curve_kind, fixed_axes_exchanged, fixed_angle_deg)
    fx, fy = com_field_tensors(centroid_a0, centroid_a1, run_device)
    steps = _as_scan_spacing(scan_spacing)
    angles = _as_angle_grid(
        DEFAULT_ANGLE_GRID_DEG if angle_grid_deg is None else angle_grid_deg
    )
    hyps = (
        [False, True] if fixed_axes_exchanged is None else [bool(fixed_axes_exchanged)]
    )
    objective = partial(
        rotation_objective,
        fx,
        fy,
        curve_kind=curve_kind,
        scan_spacing=steps,
        evaluator=evaluator,
    )
    curves, extrema, evaluator_used = None, {}, "tensor_ops"
    if fixed_angle_deg is None:
        search_angle, axes_exchanged, curves, extrema, evaluator_used = _search(
            objective, angles, hyps, curve_kind
        )
    else:
        search_angle = wrap_angle_deg(fixed_angle_deg)
        axes_exchanged = _handedness_at(objective, search_angle, hyps, curve_kind)
    grad_a0, grad_a1 = (fy, fx) if axes_exchanged else (fx, fy)
    grad_a0, grad_a1 = _rotate(grad_a0, grad_a1, search_angle)
    sign_test, third_moment, rotated_180 = "off", None, False
    if fixed_angle_deg is None and resolve_180:
        sign_test, third_moment = resolve_half_turn(grad_a0, grad_a1, steps)
        if sign_test == "flipped":
            grad_a0, grad_a1, rotated_180 = -grad_a0, -grad_a1, True
    angle = wrap_angle_deg(search_angle + (180.0 if rotated_180 else 0.0))
    _report(
        quiet,
        (angle, search_angle, extrema, axes_exchanged, sign_test, third_moment),
        found_reflection=axes_exchanged and fixed_axes_exchanged is None,
    )
    search = None
    if keep_search and curves is not None:
        search = RotationSearch(
            curve_kind=curve_kind,
            angle_grid_deg=angles.copy(),
            hypothesis_curves=curves,
            search_angle_deg=search_angle,
            scan_turn_deg=angle,
            axes_exchanged=bool(axes_exchanged),
            rotated_180=rotated_180,
            sign_test=sign_test,
            potential_skewness=third_moment,
            evaluator=evaluator_used,
        )
    return ScanRotation(
        angle, grad_a0, grad_a1, bool(axes_exchanged), rotated_180, search
    )
