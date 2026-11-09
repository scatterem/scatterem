"""The stage choice, coefficient mask and scales of ``determine_aberrations``.

Moved out of ``aberration_fitting.py`` (2026-11-08, a pure move);
``aberration_fitting`` imports these names.
"""

import torch

from . import _config


def _build_gradient_mask(
    correct_order: int, user_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Boolean (12,) mask of which aberration coefficients are optimized.

    Pure / allocation-fresh (no mutable-default aliasing). Freezes all coefficients above
    ``correct_order`` (0->C1 only, 1->through C12, 2->through order-2, 3->all), intersected
    with an optional user mask (which is not mutated).
    """
    mask = torch.ones(12, dtype=torch.bool)
    zero_from = {0: 1, 1: 3, 2: 7, 3: 11}[int(correct_order)]
    mask[zero_from:] = False
    if user_mask is not None:
        mask = mask & user_mask.to(torch.bool).clone().detach()
    return mask


def _aberration_precondition_scales(
    wavelength: float, semiconvergence_angle: float, device=None
) -> torch.Tensor:
    """Per-coefficient scale s[i] = wavelength / alpha**power[i] (~1 rad of edge phase per unit).

    Optimizing in units of s makes the cross-order LBFGS landscape isotropic.
    """
    p = _config._ABERRATION_RADIAL_POWER.to(device=device)
    alpha = max(float(semiconvergence_angle), 1e-9)
    return float(wavelength) / (alpha**p)


def _resolve_aberration_stages(
    correction_method: str, correct_order: int
) -> tuple[bool, bool]:
    """Decide which aberration-fitting stages ``determine_aberrations`` runs.

    Returns ``(do_bright_field, do_autofocus)``.

    - ``"bright-field-shifts"``: BF fit only -- fast, deterministic low-order (defocus + 2-fold
      astigmatism) + scan rotation; orders 2/3 (poorly conditioned in BF shifts) on request.
    - ``"autofocus"``: sharpness autofocus only (sparsity / L4 metric by default). Fits higher
      orders but fixes rotation (the caller must already have a correct ``meta.rotation``).
    - ``"combined"`` (default): BF fit for rotation + low-order, then the sharpness autofocus to
      extend to higher orders. The autofocus stage runs only when ``correct_order >= 2`` is
      requested, so the common low-order case stays as fast as the bright-field-shifts path.
    """
    valid = ("bright-field-shifts", "autofocus", "combined")
    if correction_method not in valid:
        raise ValueError(
            f"correction_method must be one of {valid}, got {correction_method!r}"
        )
    do_bf = correction_method in ("bright-field-shifts", "combined")
    do_tv = correction_method == "autofocus" or (
        correction_method == "combined" and int(correct_order) >= 2
    )
    return do_bf, do_tv


def _refuse_autofocus_rotation_fit(do_bf: bool, do_tv: bool, fit_rotation: bool):
    """Autofocus-only mode holds the rotation; a BF / combined run must fit it."""
    if do_tv and not do_bf and fit_rotation:
        raise ValueError(
            "fit_rotation=True is not supported with the autofocus-only method "
            "(correction_method='autofocus'); it fixes rotation. Use 'bright-field-shifts' "
            "or 'combined' to estimate rotation."
        )
