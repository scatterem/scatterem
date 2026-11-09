"""TV/sharpness-based aberration optimisation (``determine_aberrations``)."""

import logging
import time
import warnings
from typing import Any

import numpy as np
import torch

from scatterem.data.datasets import Dataset4DStem, DatasetVirtualBrightField4DStem
from scatterem.reconstruction._sampling import _fold_defocus_guess

from ._aberration_finalize import _log_and_finalize_aberration_fit
from ._aberration_stages import (
    _aberration_precondition_scales,
    _build_gradient_mask,
    _refuse_autofocus_rotation_fit,
    _resolve_aberration_stages,
)
from ._autofocus_image import (
    _build_tv_batched_evaluator,
    _build_tv_image_evaluator,
    _prepare_tv_chunk_cache,
    _TvFitContext,
)
from ._common import (
    PhaseSignMode,
    _orient_phase_image,
    _require_meta,
    _resolve_roi_center,
    _resolve_upsample_int,
    _resolve_vbf,
)
from ._defocus_sign import clear_defocus_sign_check, resolve_defocus_sign
from .bright_field import (
    aberrations_and_rotation_from_bright_field_shifts,
)

logger = logging.getLogger(__name__)


def _optimize_aberrations_tv(
    evaluate_tv,
    seed_aberrations: torch.Tensor,
    scales: torch.Tensor,
    free_mask: torch.Tensor,
    *,
    reg_weight: float = 0.05,
    lr: float = 0.5,
    max_iter: int = 20,
    defocus_index: int = 0,
    sweep_radius: float = 4.0,
    sweep_steps: int = 9,
    evaluate_tv_stack=None,
):
    """Preconditioned, regularized, multi-start, keep-best maximization of a sharpness metric.

    Optimizes ``aberr = seed + u * scales`` in scaled (~radian) units ``u`` (isotropic landscape),
    regularizing ``u`` toward 0 (the seed). A coarse coordinate-descent sweep over every free
    coefficient (defocus first) seeds LBFGS to escape the multimodal landscape and reach higher-order
    optima. The best (highest-sharpness) aberration ever seen is returned
    (keep-best); the seed is always a candidate, so the result is never worse than the seed.

    Returns (best_aberrations(12,), info dict: seed_tv, best_tv, improved_over_seed, converged,
    stop_reason, n_iter, max_iter, grad_max).  ``converged`` is the optimizer's own
    first-order test (max|grad| <= tolerance_grad), NOT a literal.
    """
    device = seed_aberrations.device
    seed = seed_aberrations.detach().clone()
    scales = scales.to(device)
    free = free_mask.to(device)

    def aberr_of(u):
        return seed + u * scales * free

    def _stack_points(u0, idx, ds):
        """All ``len(ds)`` candidates of ONE coordinate as a single (len(ds), 12).

        The candidates of a coordinate differ from ``u0`` in a single entry, so
        the whole set is one ``repeat`` plus one column write -- three device
        launches -- rather than ``len(ds)`` clone/setitem pairs.  On a 12-element
        parameter every launch is pure host dispatch (~8 us against ~0 us of
        device work), and the sweep runs this once per free coefficient.
        """
        U = u0.unsqueeze(0).repeat(len(ds), 1)
        U[:, idx] = torch.tensor(ds, dtype=u0.dtype, device=u0.device)
        return U

    best = {"tv": None, "aberr": seed.clone()}

    def keep_best(aberr, tv):
        """The keep-best bookkeeping, split out from the evaluation so the
        batched sweep below can feed it values it already has.  The order of
        calls matters and is preserved: the test is a STRICT ``>``, so the FIRST
        candidate attaining the maximum is the one kept."""
        if best["tv"] is None or tv > best["tv"]:
            best["tv"] = tv
            best["aberr"] = aberr.detach().clone()
        return tv

    def consider(aberr):
        with torch.no_grad():
            tv = float(evaluate_tv(aberr))
        return keep_best(aberr, tv)

    seed_tv = consider(seed)

    # Coarse coordinate-descent multi-start over the free coefficients, in radial order
    # (defocus first, then higher orders), each swept in scaled (~rad) units. Sweeping every free
    # coefficient -- not just defocus -- is what lets the LBFGS step actually reach higher-order
    # optima (coma, Cs); with a defocus-only sweep their weak gradients leave them near the seed.
    u0 = torch.zeros(12, device=device)
    if sweep_steps > 1:
        grid = torch.linspace(-sweep_radius, sweep_radius, sweep_steps).tolist()
        sweep_order = [defocus_index] + [
            i for i in range(12) if bool(free[i]) and i != defocus_index
        ]
        # base_tv carries the running best across coordinates (the committed u0 point was already
        # evaluated by the previous coordinate's sweep, so no need to re-evaluate it here).
        #
        # ``u0`` is all zeros at this point and ``aberr_of(0) == seed + 0 * scales
        # * free`` IS ``seed``, bit-identically -- ``scales`` is
        # ``wavelength / alpha**power``, finite and positive, so no entry can turn
        # the multiply into a NaN.  Its metric is therefore ``seed_tv``, which the
        # line above is already holding, and evaluating it again is a full chunked
        # reconstruction for a number that is in a python local.  Measured on
        # ``r14_ffstem/carbon/fit``: 5.93 ms = 3.07 % of the fit.
        base_tv = seed_tv
        for idx in sweep_order:
            if not bool(free[idx]):
                continue
            best_d = float(u0[idx])
            # The ``sweep_steps`` points of ONE coordinate are mutually
            # independent -- each is ``u0`` with a single entry replaced -- so
            # they are one batched evaluation, not ``sweep_steps`` serial ones.
            # (Across coordinates they are NOT: this is coordinate DESCENT, and
            # ``u0[idx]`` is committed before the next coordinate starts.)
            # Measured on ``r14_ffstem/carbon/fit``: the sweep is 75.5 % of the
            # fit at 65 evaluations of 5.85 ms, 63 of them in these inner grids.
            # ...and one of those points is the one the loop is already standing
            # on: ``grid`` is symmetric about 0 with an odd ``sweep_steps``, so it
            # CONTAINS 0.0, and ``u0[idx]`` is 0 for the coordinate being swept
            # (it is written only AFTER this sweep).  That candidate reproduces the
            # committed ``u0``, whose metric is ``base_tv`` by the invariant above.
            # Dropping it is exact rather than approximate: the keep-best test is a
            # strict ``>`` and ``base_tv <= best["tv"]`` because that point has
            # already been considered, so BOTH branches it could take -- the
            # keep-best update and ``tv > base_tv`` -- are no-ops on it.  Written
            # as a comparison against ``best_d`` rather than against 0.0 so it
            # stays correct if a caller ever seeds ``u0`` non-zero: then nothing
            # matches and the full grid is evaluated, which is the safe direction.
            # 7 of 65 evaluations at the published Fig2 config.
            sweep_grid = [d for d in grid if d != best_d]
            if not sweep_grid:
                # Degenerate ``sweep_radius=0``: every grid point IS the committed
                # point.  Guarded rather than left to ``torch.stack([])``.
                continue
            # ``aberr_of`` applied to the (len(sweep_grid), 12) candidate matrix,
            # written with the SAME association as the scalar spelling
            # (``seed + u * scales * free``, broadcast over the leading axis) so
            # every element is bit-identical to what the per-candidate stack
            # produced.  The rows are then reused below instead of being rebuilt:
            # ``cand_aberr[j]`` IS ``aberr_of(cand[j])``, and it is a contiguous
            # view, so the per-candidate rebuild was pure host dispatch for
            # values already on the device.  Measured on
            # ``r14_ffstem/carbon/fit``: 65 -> 6 launches per swept coefficient.
            cand_aberr = seed + _stack_points(u0, idx, sweep_grid) * scales * free
            if evaluate_tv_stack is not None:
                with torch.no_grad():
                    tvs = evaluate_tv_stack(cand_aberr).tolist()
            else:
                tvs = None
            for j, d in enumerate(sweep_grid):
                aberr = cand_aberr[j]
                tv = keep_best(aberr, tvs[j]) if tvs is not None else consider(aberr)
                if tv > base_tv:
                    base_tv = tv
                    best_d = d
            u0[idx] = best_d

    u = u0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [u], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        aberr = aberr_of(u)
        loss = -evaluate_tv(aberr) + reg_weight * (u * free).pow(2).sum()
        loss.backward()
        with torch.no_grad():
            if u.grad is not None:
                u.grad[~free] = 0
        return loss

    optimizer.step(closure)
    consider(aberr_of(u.detach()))

    # Report the optimizer's OWN termination rather than a literal.  ``LBFGS.step`` has
    # three distinct exits and they call for opposite fixes: ``n_iter == max_iter`` is
    # the iteration cap (raise it); ``max|grad| <= tolerance_grad`` is a genuine
    # first-order stationary point (the only one that is convergence); anything else is
    # ``tolerance_change``, i.e. the strong-Wolfe step or the objective collapsed while
    # the gradient is still large -- a STALL, which more iterations cannot fix.
    # Measured with the earlier surrogate adjoint (not re-measured since) on the FF-STEM rows: Fig2_carbon
    # stopped after 5 of 50 iterations at max|grad| 6.1e-06, Fig1_Gd2O3 after 2 of 50 at
    # 2.7e-01, both on ``tolerance_change`` and 61x / 2.7e+06x above ``tolerance_grad``:
    # neither published fit reached a stationary point or spent its iteration budget.
    group = optimizer.param_groups[0]
    lbfgs_state = optimizer.state.get(u, {})
    n_iter = int(lbfgs_state.get("n_iter", 0))
    grad_max = float(u.grad.abs().max()) if u.grad is not None else float("inf")
    if grad_max <= float(group["tolerance_grad"]):
        stop_reason = "tolerance_grad"
    elif n_iter >= int(group["max_iter"]):
        stop_reason = "max_iter"
    else:
        stop_reason = "tolerance_change"

    info = {
        "seed_tv": seed_tv,
        "best_tv": best["tv"],
        "improved_over_seed": best["tv"] > seed_tv + 1e-6,
        "converged": stop_reason == "tolerance_grad",
        "stop_reason": stop_reason,
        "n_iter": n_iter,
        "max_iter": int(group["max_iter"]),
        "grad_max": grad_max,
    }
    return best["aberr"], info


def _autofocus_seed_from_meta(
    vBF, verbosity: int, bf_options: dict[str, Any], sign_check, update_dataset: bool
) -> tuple[torch.Tensor, float]:
    """Autofocus seed and its rotation: ``meta.aberrations`` (C10 less
    ``defocus_guess``) at ``meta.rotation``, or a BF fit at the rotation it used.

    The BF fit (rotation held, ``bf_options`` passed on) runs when C10, C12a and
    C12b (slots 0-2) of ``meta`` are zero, and writes ``meta`` only with
    ``update_dataset``; slots it leaves at zero keep their ``meta`` value (Cs).
    If it fails, the seed is ``meta.aberrations`` with ``defocus_guess`` folded
    in; if its slots 0-2 are zero, ``ValueError`` (see determine_aberrations).
    ``sign_check`` (``(upsample_int, n_batches)``, or ``None``) runs the
    defocus-sign check on the fitted seed.
    """
    seed_aberr = _require_meta(vBF).aberrations.array.clone().to(vBF.device)
    if float(seed_aberr[:3].abs().max()) >= 1e-12:  # C10, C12a, C12b
        if sign_check is not None:
            logger.warning(
                "check_defocus_sign has no effect: the autofocus seed comes "
                "from meta.aberrations, not from a bright-field-shift fit"
            )
        return _fold_defocus_guess(vBF, seed_aberr), _require_meta(vBF).rotation
    try:
        fitted, rotation = aberrations_and_rotation_from_bright_field_shifts(
            vBF,
            fit_rotation=False,
            update_dataset=update_dataset,
            verbosity=verbosity,
            **bf_options,
        )
    except Exception as e:  # noqa: BLE001
        guess = _fold_defocus_guess(vBF, seed_aberr)
        if float(guess[:3].abs().max()) < 1e-12:
            raise ValueError(
                "the autofocus has no start: C10, C12a, C12b of meta.aberrations "
                "are zero, meta.defocus_guess is 0 and the bright-field-shift fit "
                f"that would seed it failed ({type(e).__name__}: {e}). Set a start, "
                "e.g. dataset.meta.aberrations.array[0] = -50.0 (C10 [A]; defocus == -C10)."
            ) from e
        if verbosity > 0:
            logger.warning("BF-shift seeding failed (%s); TV starts at the guess", e)
        return guess, _require_meta(vBF).rotation
    fitted = fitted.detach().clone().to(vBF.device)
    seed_aberr = torch.where(fitted == 0, seed_aberr.to(fitted.dtype), fitted)
    if sign_check is not None:
        seed_aberr, _ = resolve_defocus_sign(
            vBF, seed_aberr, float(rotation), *sign_check
        )
    if verbosity > 0:
        logger.info(
            "TV seed from bright-field shifts: %s", seed_aberr[:4].cpu().numpy()
        )
    return seed_aberr, float(rotation)


def determine_aberrations(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem,
    correction_method: str = "combined",
    fit_rotation: bool = True,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: float | None = None,
    bin_factors: tuple[int, ...] = (2, 1, 1),
    upsample: int | str = "nyquist",
    n_batches: int = 25,
    roi_shape: tuple[int, int] = (128, 128),
    roi_center: str | tuple[int, int] = "center",
    num_iterations: int = 10,
    lr: float = 20,
    bright_field_mask_threshold: float = 0.5,
    target_percentage_nonzero_pixels: float = 0.75,
    correct_order: int = 1,
    gradient_mask: torch.Tensor | None = None,
    verbosity: int = 0,
    update_dataset: bool = True,
    n_center_indices: int = 25,
    phase_sign: PhaseSignMode = "preserve",
    sharpness_metric: str = "sparsity",
    reg_weight: float = 0.05,
    alignment_method: str = "reference",
    peak_selection: str = "argmax",
    init_method: str = "closed_form",
    use_com_rotation_prior: str = "auto",
    check_defocus_sign: bool = False,
) -> tuple[torch.Tensor, DatasetVirtualBrightField4DStem]:
    """Perform a joint ptychography reconstruction and aberration determination.

    Parameters
    ----------
    dataset : Dataset4DStem or DatasetVirtualBrightField4DStem
        Object containing the diffraction pattern.
    correction_method : str, optional
        How to determine the aberrations. "combined" (default) fits
        low-order + rotation from bright-field shifts, then runs the
        sharpness autofocus to extend to higher orders when
        ``correct_order >= 2``. "bright-field-shifts" does only the fast
        low-order + rotation fit. "autofocus" does only the autofocus
        (rotation fixed).
    fit_rotation : bool, optional
        Whether to fit the rotation. Default is True.
    registration_upsample_factor : int, optional
        Upsampling factor for the registration. Default is 10.
    lowpass_fwhm_bright_field : float or None, optional
        FWHM of the lowpass filter for the bright field. Default is None.
    bin_factors : tuple of int, optional
        Bin factors for the bright field, used for bright field shifts.
        Default is ``(2, 1, 1)``.
    upsample : int or str, optional
        Upsampling factor for the diffraction pattern. Default is
        "nyquist".
    n_batches : int, optional
        SSB batches of the defocus sign check and the TV step. Default is 25.
    roi_shape : tuple of int, optional
        Shape of the region of interest. Default is ``(128, 128)``.
    roi_center : str or tuple of int, optional
        Center of the region of interest. Default is "center".
    num_iterations : int, optional
        Number of optimization iterations. Default is 10.
    lr : float, optional
        Learning rate for the optimizer. Default is 20.
    bright_field_mask_threshold : float, optional
        Threshold for the bright field, used for bright field shifts.
        Default is 0.5.
    target_percentage_nonzero_pixels : float, optional
        Target percentage of nonzero pixels for the bright field. Default
        is 0.75.
    correct_order : int, optional
        Highest order corrected (default 1: C1, C12); by autofocus when it runs
        (the bright-field stage then fits order 1), else by the bright-field fit.
    gradient_mask : torch.Tensor or None, optional
        Mask for the gradient, used for the autofocus stage. Default is
        None.
    verbosity : int, optional
        Verbosity level. Default is 0.
    update_dataset : bool, optional
        Write the fit onto the dataset (detector-frame aberrations). Default True.
        With False, no stage writes ``meta``.
    n_center_indices : int, optional
        Number of center indices to use for the bright field shifts.
        Default is 25.
    phase_sign : PhaseSignMode, optional
        Sign convention applied to the reconstructed phase. Default is
        "preserve".
    sharpness_metric : str, optional
        Autofocus objective for the "autofocus"/"combined" higher-order
        stage: "sparsity" (default, L4/kurtosis, less gameable and
        unbiased for higher orders) or "tv".
    reg_weight : float, optional
        L2 regularization (in scaled units) toward the seed in the
        autofocus optimizer. Default is 0.05.
    alignment_method : str, optional
        "reference" (default) uses the existing single-reference
        cross-correlation path. "pairwise" uses graph-synchronised
        pairwise cross-correlation, which is more robust when large
        aberration shifts cause the reference image to be blurred.
    peak_selection : str, optional
        Which cross-correlation peak is each bright-field shift: "argmax"
        (default, the global maximum) or "guided" (the highest peak near a
        smooth prediction from the pixels nearer the centre). Use "guided" on
        crystals, whose correlation is multi-peaked; "argmax" then warns.
    init_method : str, optional
        Seed of the bright-field-shift fit: "closed_form" (default, the
        closed-form C1, C12 and rotation) or "zeros" (caps that fit at order 1).
    use_com_rotation_prior : str, optional
        "auto" (default) seeds the rotation from the diffraction centre of
        mass when ``meta.rotation`` is unset; "off" always starts from
        ``meta.rotation``.
    check_defocus_sign : bool, optional
        After each bright-field-shift fit, build the corrected image at the
        fitted coefficients ``a`` and at ``-a``, and keep ``-a`` when its
        image energy is more than 2.5 times as large (see ``_defocus_sign``).
        A wrong peak negates the whole shift field, so the check negates every
        coefficient, not C10 alone. In "autofocus" mode the check runs on the
        seed fit; a seed from ``meta.aberrations`` is not checked. The
        decision is stored on the returned vBF as ``defocus_sign_check``
        (``None`` when no check ran). Default is False; with False the fit is
        unchanged.

    Returns
    -------
    aberrations : torch.Tensor
        The 12 fitted aberration coefficients.
    dataset : DatasetVirtualBrightField4DStem
        The virtual bright field the fit ran on. Its ``aberration_rotation`` is
        the scan rotation (deg) the aberrations hold at, set in every mode.

    Raises
    ------
    ValueError
        If the roi_center is invalid.
    ValueError
        If the dataset is not a valid Dataset4DStem or
        DatasetVirtualBrightField4DStem object.
    ValueError
        If the upsample is invalid.
    ValueError
        If the autofocus has no start: ``correction_method="autofocus"``,
        C10, C12a and C12b of ``meta.aberrations`` zero, ``meta.defocus_guess``
        0, and the bright-field-shift fit that seeds the autofocus fails.

    Notes
    -----
    The start value. The bright-field-shift fit (``"combined"`` and
    ``"bright-field-shifts"``) is a least-squares fit of the measured
    parallax shifts. It does not read ``meta.aberrations``, so it needs no
    start. The autofocus is a local search: it sweeps each free coefficient
    over about +-4 radians of edge phase (``lambda / alpha**p`` per unit)
    around its seed, then refines with L-BFGS and a pull towards the seed.
    The sharpness of the corrected image is an oscillating function of the
    residual aberration, so the search finds the true value only from a
    seed in its basin. In ``"autofocus"`` mode the seed is
    ``meta.aberrations`` (C10 less ``meta.defocus_guess``). If C10, C12a and
    C12b are zero, a bright-field-shift fit gives them. If it fails too, the
    seed holds no information about these free terms (also with Cs set), and
    the result would depend only on the start, so the function raises
    ``ValueError``. A non-zero C10, C12a or C12b removes it.
    """
    roi_center = _resolve_roi_center(roi_center, dataset)

    do_bf, do_tv = _resolve_aberration_stages(correction_method, correct_order)
    _refuse_autofocus_rotation_fit(do_bf, do_tv, fit_rotation)

    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)

    semiconvergence_angle = _require_meta(dataset).semiconvergence_angle
    wavelength = _require_meta(dataset).wavelength

    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)
    clear_defocus_sign_check(vBF)
    bf_options: dict[str, Any] = {"peak_selection": peak_selection}
    bf_options["init_method"] = init_method
    bf_options["use_com_rotation_prior"] = use_com_rotation_prior
    # ``_resolve_aberration_stages`` guarantees at least one stage runs and each
    # binds ``aberrations``, so these placeholders never reach the report below.
    aberrations = torch.zeros(12, device=vBF.device)
    rotation = 0.0
    direct_ptycho_image = None
    # Shapes from the vBF stack, not vBF.G — the streaming provider may keep no resident G.
    new_shape = (
        int(round(vBF.array.shape[0] * upsample_int[0])),
        int(round(vBF.array.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)
    Kx = vBF.k[:, 1]
    Ky = vBF.k[:, 0]

    if do_bf:
        with torch.no_grad():
            _t0 = time.perf_counter()
            if verbosity > 0:
                logger.info("Fitting aberrations and rotation from bright field shifts")

            aberrations, rotation = aberrations_and_rotation_from_bright_field_shifts(
                vBF,
                fit_rotation=fit_rotation,
                target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
                registration_upsample_factor=registration_upsample_factor,
                lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
                bin_factors=bin_factors,
                verbosity=verbosity,
                update_dataset=update_dataset,
                n_center_indices=n_center_indices,
                alignment_method=alignment_method,
                **bf_options,
                correct_order=1 if do_tv else correct_order,  # autofocus: orders >= 2
            )
            if check_defocus_sign:
                aberrations, _ = resolve_defocus_sign(
                    vBF, aberrations, float(rotation), upsample_int, n_batches
                )

            if verbosity > 0:
                logger.info(
                    "Time to fit aberrations and rotation: %.3f s",
                    time.perf_counter() - _t0,
                )

    if do_tv:
        vBF.array.requires_grad = False
        # All vBF flavors serve chunks through get_G_chunk, fetched lazily
        # inside the checkpointed closure in ``_build_tv_image_evaluator``
        # (materializing them here would defeat the memory budget).
        n_k_tv = vBF.n_bright_field

        # Resolve the seed and its rotation (passed on, not written to meta),
        # preconditioning scales, free mask, and leading count of free gradients.
        if do_bf:
            # seed the autofocus from the bright-field-shift fit just run above
            seed_aberr = aberrations.detach().clone().to(vBF.device)
            tv_rotation = float(rotation)
        else:
            seed_aberr, tv_rotation = _autofocus_seed_from_meta(
                vBF, verbosity, bf_options,
                (upsample_int, n_batches) if check_defocus_sign else None,
                update_dataset,
            )  # fmt: skip
            rotation = tv_rotation  # the seed's: the aberrations hold at it

        scales = _aberration_precondition_scales(
            wavelength, semiconvergence_angle, device=vBF.device
        )
        free_mask = _build_gradient_mask(correct_order, gradient_mask).to(vBF.device)
        # The analytic adjoint's cost is LINEAR in the number of coefficients
        # differentiated, and ``aberr_of`` zeroes the frozen ones before they
        # reach ``u.grad`` -- so a leading count skips dead work.
        _free_idx = torch.nonzero(free_mask.reshape(-1), as_tuple=False)
        n_grad_coeffs = int(_free_idx[-1].item()) + 1 if _free_idx.numel() else 0

        rotation_t = torch.tensor(tv_rotation, device=vBF.device)
        factor = np.sqrt(upsample_int[0] * upsample_int[1])
        n_batches_tv, g_chunks = _prepare_tv_chunk_cache(
            vBF, upsample_int, n_batches, n_k_tv
        )
        ctx = _TvFitContext(
            vBF=vBF,
            semiconvergence_angle=semiconvergence_angle,
            wavelength=wavelength,
            Kx=Kx,
            Ky=Ky,
            Qx=Qx,
            Qy=Qy,
            rotation_t=rotation_t,
            factor=factor,
            n_k_tv=n_k_tv,
            n_batches_tv=n_batches_tv,
            upsample_int=upsample_int,
            new_shape=new_shape,
            roi_center=roi_center,
            roi_shape=roi_shape,
            sharpness_metric=sharpness_metric,
            n_grad_coeffs=n_grad_coeffs,
            g_chunks=g_chunks,
        )
        evaluate_tv, last_image = _build_tv_image_evaluator(ctx)
        evaluate_tv_stack = _build_tv_batched_evaluator(ctx)
        aberrations, tv_info = _optimize_aberrations_tv(
            evaluate_tv,
            seed_aberr,
            scales,
            free_mask,
            reg_weight=reg_weight,
            lr=lr if lr <= 1.0 else 0.5,
            max_iter=num_iterations,
            evaluate_tv_stack=evaluate_tv_stack,
        )
        if not tv_info["improved_over_seed"]:
            warnings.warn(
                "TV aberration refinement did not improve sharpness over the seed; "
                "keeping the seed aberrations.",
                UserWarning,
                stacklevel=2,
            )
        # recompute the image at the chosen aberrations for the return value
        with torch.no_grad():
            evaluate_tv(aberrations)
        direct_ptycho_image = last_image["img"]
        if direct_ptycho_image is not None:
            direct_ptycho_image = _orient_phase_image(
                direct_ptycho_image,
                phase_sign=phase_sign,
                verbosity=verbosity,
            )
        if verbosity > 0:
            logger.info("aberrations = %s", aberrations[0:4].detach().cpu().numpy())
    _log_and_finalize_aberration_fit(
        vBF,
        aberrations,
        rotation,
        direct_ptycho_image,
        update_dataset,
        fit_rotation,
    )
    return aberrations.detach(), vBF
