"""Choose the sign of a fitted defocus by the coherence of the corrected image.

The bright-field-shift fit reads C10 from the parallax field. On a crystal the
cross-correlation that measures each shift is multi-peaked, and a wrong peak
can flip the sign of the whole field. The shift field is linear in the
coefficients, so a negated field gives ``-a``: every fitted coefficient, A1
included, has the wrong sign. The check therefore negates all of them.

The image the autofocus scores, ``sum_k Im(ifft2(G_k corrected))`` over the
bright-field pixels ``k``, gives an independent check. A correction is a pure
phase factor on each ``G_k``, so every term keeps its energy whatever the
aberrations are; only the coherence of the sum changes. At the right
aberrations the terms add in phase and the image energy is largest. At the
wrong defocus sign each term keeps a phase error ``2 chi(k)``, the terms
cancel in part and the energy drops. Noise adds incoherently at both signs.

:func:`resolve_defocus_sign` builds the image at the fitted coefficients
``a`` and at ``-a`` (the rotation held), and keeps ``-a`` only when its energy
is more than ``SIGN_FLIP_RATIO`` times that at ``a``. A ratio between
``1 / SIGN_FLIP_RATIO`` and ``SIGN_FLIP_RATIO`` keeps ``a`` and logs a warning:
the image does not decide the sign there.

Evidence (2026-10-09): 32 bright-field-shift fits of the synthetic 3.19 A
lattice (true C10 of +-40,
+-80, +-120 and +-200 A; scans of 20 x 20 steps, which hold the period
exactly, and of 17 x 17 steps, which do not; ``"argmax"`` and ``"guided"``).
Six fits had the wrong sign; their energy ratios E(-C10) / E(C10) were 3.6
to 90, and the check flipped all six. The 26 fits with the right sign had
ratios of 0.005 to 1.94 (the two highest, 1.94 and 1.66, at C10 = -301 and
-15.8 A); the check kept all of them. The threshold 2.5 sits between the
two groups. It was chosen on this one fixture, which is why the check is off
by default.

Those ratios were first measured with C10 alone negated. With all fitted
coefficients negated (re-measured 2026-10-09) the same six fits flip, at
ratios of 3.6 to 90, and the same 26 are kept, at 0.005 to 1.94. Four kept
fits (0.58, 0.71, 1.66 and 1.94) lie in the warning band.

With a planted A1 (14 fits on the same lattice, seven pairs of C10 and A1)
the decision was the same whether the check negated all coefficients or C10
alone. The five argmax fits that the check flipped also had a rotation 11 to
50 deg off, so neither candidate recovered A1: a wrong peak there is not a clean
negation of the field.

Re-measured 2026-11-07 with the v0.4 registration (``measure_bf_parallax``),
``"argmax"``, 20 x 20 steps, C10 of +-40 to +-200 A, with and without the
planted A1 (32 fits). Twelve fits had the wrong sign. The check flipped six
of them (ratios 2.64 to 6.20) and kept six (ratios 0.13 to 2.04; five of
them in the warning band). The 20 fits with the right sign had ratios of
0.02 to 0.97, and the check kept all of them. So the check never flipped a
right sign, but with this registration it misses half of the wrong ones.

Each evaluation is one pass over the bright-field stack, the cost of one
autofocus objective.
"""

import logging
from dataclasses import asdict, dataclass

import torch

from ._autofocus_image import (
    _build_tv_image_evaluator,
    _TvFitContext,
    _windowed_roi,
)
from ._common import _require_meta

logger = logging.getLogger(__name__)

#: Energy ratio above which the negated coefficients replace the fitted ones.
SIGN_FLIP_RATIO = 2.5


@dataclass(frozen=True)
class DefocusSignCheck:
    """What :func:`resolve_defocus_sign` measured and decided.

    Parameters
    ----------
    c10_fitted : float
        C10 from the bright-field-shift fit, in Angstrom.
    c10_kept : float
        C10 after the check: ``c10_fitted`` or ``-c10_fitted``.
    energy_fitted : float
        Energy of the corrected image at the fitted coefficients (mean square
        of the Hann-windowed, mean-free image).
    energy_flipped : float
        The same energy with every fitted coefficient negated.
    flipped : bool
        Whether the check negated the fitted coefficients.
    """

    c10_fitted: float
    c10_kept: float
    energy_fitted: float
    energy_flipped: float
    flipped: bool

    def as_dict(self) -> dict[str, float | bool]:
        """Return the fields as a plain dictionary.

        Returns
        -------
        dict
            Field name to value.
        """
        return asdict(self)


def clear_defocus_sign_check(vBF) -> None:
    """Drop the decision an earlier fit stored on a reused ``vBF``."""
    vBF.defocus_sign_check = None


def _image_context(
    vBF, rotation: float, upsample_int, n_batches: int
) -> tuple[_TvFitContext, tuple[int, int]]:
    """The autofocus image context at a fixed rotation, with no chunk cache."""
    meta = _require_meta(vBF)
    shape = (
        int(round(vBF.array.shape[0] * upsample_int[0])),
        int(round(vBF.array.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(shape)
    ctx = _TvFitContext(
        vBF=vBF,
        semiconvergence_angle=meta.semiconvergence_angle,
        wavelength=meta.wavelength,
        Kx=vBF.k[:, 1],
        Ky=vBF.k[:, 0],
        Qx=Qx,
        Qy=Qy,
        rotation_t=torch.tensor(float(rotation), device=vBF.device),
        factor=float((upsample_int[0] * upsample_int[1]) ** 0.5),
        n_k_tv=int(vBF.n_bright_field),
        n_batches_tv=int(n_batches),
        upsample_int=upsample_int,
        new_shape=shape,
        roi_center=(shape[0] // 2, shape[1] // 2),
        roi_shape=shape,
        sharpness_metric="sparsity",
        n_grad_coeffs=0,
        g_chunks={},
    )
    return ctx, shape


def resolve_defocus_sign(
    vBF, aberrations: torch.Tensor, rotation: float, upsample_int, n_batches: int
) -> tuple[torch.Tensor, DefocusSignCheck]:
    """Keep the fitted coefficients or their negative, whichever is coherent.

    Parameters
    ----------
    vBF : DatasetVirtualBrightField4DStem
        The virtual bright field the fit ran on.
    aberrations : torch.Tensor
        The 12 fitted coefficients; C10 is ``aberrations[0]``. Coefficients
        the fit did not refine are zero, so negating all 12 negates the fit.
    rotation : float
        Scan rotation used to build the image, in degrees.
    upsample_int : tuple of int
        Upsampling of the scan grid, as the fit resolved it.
    n_batches : int
        Number of bright-field chunks per image pass.

    Returns
    -------
    aberrations : torch.Tensor
        A new tensor: ``-aberrations`` when the check flips the sign, an
        unchanged copy otherwise.
    check : DefocusSignCheck
        Both energies and the decision. It is also stored on ``vBF`` as
        ``vBF.defocus_sign_check``, which a later check replaces.
    """
    ctx, shape = _image_context(vBF, rotation, upsample_int, n_batches)
    evaluate, last_image = _build_tv_image_evaluator(ctx)
    center = (shape[0] // 2, shape[1] // 2)

    def energy(candidate: torch.Tensor) -> float:
        with torch.no_grad():
            evaluate(candidate)
        image = last_image["img"]
        if image is None:
            raise RuntimeError("the autofocus evaluator returned no image")
        windowed = _windowed_roi(image - image.mean(), center, shape)
        return float(windowed.pow(2).mean())

    fitted = aberrations.detach().clone()
    flipped = -fitted
    e_fitted = energy(fitted)
    e_flipped = energy(flipped)
    flip = e_flipped > SIGN_FLIP_RATIO * e_fitted
    ambiguous = e_flipped * SIGN_FLIP_RATIO > e_fitted and bool(fitted.any())
    if not flip and ambiguous:
        logger.warning(
            "defocus-sign check is ambiguous: the image energy at -a is %.3g "
            "times that at the fitted a (C10 = %.4g A); kept the fitted sign",
            e_flipped / e_fitted if e_fitted > 0 else float("inf"),
            float(fitted[0]),
        )
    kept = flipped if flip else fitted
    check = DefocusSignCheck(
        c10_fitted=float(fitted[0]),
        c10_kept=float(kept[0]),
        energy_fitted=e_fitted,
        energy_flipped=e_flipped,
        flipped=bool(flip),
    )
    vBF.defocus_sign_check = check
    return kept, check
