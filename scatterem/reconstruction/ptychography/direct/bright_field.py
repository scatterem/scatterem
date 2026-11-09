"""The rotation/aberration fits built on bright-field shifts.

Shift *measurement* is in :mod:`._bf_parallax`.

Frames. The fits model the measured shifts with
``scatterem.utils.transfer.aberrations_to_image_shifts``, which reads its
vector in the SHIFT-MODEL frame (it rotates the argument of ``grad chi``), and
:func:`fit_aberrations_and_rotation_to_bright_field_shifts` returns that frame.
:func:`aberrations_and_rotation_from_bright_field_shifts` converts its result to
the DETECTOR frame of ``meta.aberrations``
(``scatterem.utils.transfer.shift_model_to_detector_frame``, at the fitted
rotation) before it returns it or writes it, so the direct-ptychography
kernels, the ptychography probe and tilt-corrected dark field read it as
meant. The two frames differ only in the m-fold pairs at ``rotation != 0``.
"""

import logging
import math
from dataclasses import replace

import numpy as np
import torch
from tqdm.auto import tqdm

from scatterem.data.aberrations import Aberrations
from scatterem.data.datasets import (
    DatasetVirtualBrightField4DStem,
)
from scatterem.utils.transfer import (
    aberrations_to_image_shifts,
    shift_model_to_detector_frame,
)

from ._bf_parallax import measure_bf_parallax
from ._bright_field_closed_form import (
    _rotation_matrix_2x2 as _rotation_matrix_2x2,
)
from ._bright_field_closed_form import (
    fit_low_order_aberrations_and_rotation_closed_form as fit_low_order_aberrations_and_rotation_closed_form,
)
from ._bright_field_fit_result import (
    BrightFieldShiftFit,
    _shift_fit_result,
    _worst_of_levels,
)
from ._bright_field_refine import ORDER_END, ShiftProblem, refine_shift_fit
from ._com_rotation_prior import _com_rotation_prior
from ._common import _detector_dr, _require_meta
from ._level_shifts import level_shifts
from ._peak_selection import validate_peak_selection, warn_if_multi_peaked

logger = logging.getLogger(__name__)


def fit_aberrations_and_rotation_to_bright_field_shifts(
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray | torch.Tensor,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    include_in_fit_mask: torch.Tensor | None = None,
    rotation_init: torch.Tensor | None = None,
    rotation_requires_grad: bool = True,
    init_method: str = "closed_form",
    residual_warn_rtol: float = 0.15,
    correct_order: int = 1,
    max_iter: int = 250,
    lr: float = 1e-1,
    verbosity: int = 0,
) -> BrightFieldShiftFit:
    """Fit the aberrations and rotation to the bright field shifts.

    Parameters
    ----------
    target_shifts : torch.Tensor
        Target bright-field shifts, shape `(N, 2)`, one row per pixel of
        ``bright_field_mask`` in row-major (``torch.argwhere``) order, as
        ``aberrations_to_image_shifts(...)[bright_field_mask]`` lays them out.
    k_sampling : numpy.ndarray
        Real-space sampling of the detector grid, ``dataset.dr`` (Angstrom),
        shape `(2,)` -- NOT the reciprocal detector pixel size: the shift
        model's frequencies are ``fftfreq(shape, k_sampling)``.
    wavelength : float
        Electron wavelength, in Angstrom.
    bright_field_mask : torch.Tensor
        Bright-field mask, shape `(H, W)`.
    include_in_fit_mask : torch.Tensor, optional
        Mask of bright-field shifts to include in the fit. Default is None,
        which includes all of them.
    rotation_init : torch.Tensor, optional
        Initial rotation, in degrees. Default is None, which initializes
        the rotation to 0.
    rotation_requires_grad : bool, optional
        Whether to optimize the rotation. If False the rotation is held at
        `rotation_init` (0 when None), and the closed-form seed is solved at
        that rotation. Default is True.
    init_method : str, optional
        Initialization method, either "closed_form" (default: the
        closed-form C1, C12 and rotation estimate) or "zeros". Either seed is
        then refined over the coefficients ``correct_order`` frees; "zeros"
        additionally caps that at order 1.
    residual_warn_rtol : float, optional
        Relative residual limit. Above it the fit emits a `UserWarning` and
        the result's ``reliable`` is False. The relative residual is
        `RMS(final_shifts - target_shifts) / RMS(target_shifts)`. Default
        is 0.15.
    correct_order : int, optional
        Highest aberration order refined: 0 is C1 only, 1 (default) C1 and
        C12a/b, 2 adds coma and three-fold astigmatism (C21, C23), 3 adds
        C30, C32 and C34. Higher-order terms correlate strongly with the
        low-order ones over the bright-field disk, so freeing them without
        such aberrations in the data biases C1 (about 6% on low-defocus
        data); conversely an order-1 fit absorbs a real spherical aberration
        into C1.
    max_iter : int, optional
        Maximum number of LBFGS iterations. Default is 250.
    lr : float, optional
        Learning rate. Default is 1e-1.
    verbosity : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    BrightFieldShiftFit
        The fitted aberrations (Angstrom) and rotation (degrees), with the
        relative residual and the ``reliable`` flag. It unpacks as
        ``aberrations, rotation = fit``. Check ``reliable`` before a pipeline
        writes the values as results; the class docstring gives the fallback
        order. The aberrations are in the SHIFT-MODEL frame of
        ``scatterem.utils.transfer.aberrations_to_image_shifts`` (the
        model fitted here): at ``rotation != 0`` every m-fold pair differs
        from the detector-frame vector ``meta.aberrations`` holds. Convert
        with ``scatterem.utils.transfer.shift_model_to_detector_frame``
        at the fitted rotation before using them as ``meta.aberrations``;
        :func:`aberrations_and_rotation_from_bright_field_shifts` does.

    Notes
    -----
    The refinement runs in float64 whatever the input dtype (outputs are
    float32), with LBFGS in coordinates scaled by each coefficient's shift
    sensitivity, so it converges and its result does not depend on float32
    rounding of the inputs.
    """

    device = target_shifts.device
    target_shifts = target_shifts.to(torch.float32)
    if bright_field_mask.device != device:
        bright_field_mask = bright_field_mask.to(device)
    n_bf = int(bright_field_mask.sum().item())
    if include_in_fit_mask is None:
        include = torch.ones(n_bf, device=device, dtype=torch.bool)
    else:
        # (N,) per pixel or (N, 2) per component: a pixel is whole-in/whole-out.
        include = include_in_fit_mask.to(device).reshape(n_bf, -1)[:, 0].bool()

    # --- Seed the optimization -------------------------------------------------
    rotation_init_t = (
        torch.zeros(1, device=device, dtype=torch.float32)
        if rotation_init is None
        else rotation_init.to(device).to(torch.float32)
    )
    if init_method not in ("closed_form", "zeros"):
        raise ValueError(
            f"init_method must be 'closed_form' or 'zeros', got {init_method!r}"
        )
    if correct_order not in ORDER_END:
        raise ValueError(f"correct_order must be 0, 1, 2 or 3, got {correct_order!r}")
    seed_info = None
    free = torch.arange(12, device=device) < ORDER_END[int(correct_order)]
    if init_method == "closed_form":
        seed, seed_rot, seed_info = fit_low_order_aberrations_and_rotation_closed_form(
            target_shifts,
            k_sampling,
            wavelength,
            bright_field_mask,
            include_in_fit_mask=include,
            rotation_prior=rotation_init_t,
            rotation_fixed=not rotation_requires_grad,
        )
        # Held rotation (rotation_fixed) or the closed-form one when the shifts fix it:
        # well posed. Else (near-zero defocus, rotation free) keep the prior, free.
        freeze_rotation = seed_info["well_posed"] or not rotation_requires_grad
        rotation0 = seed_rot if freeze_rotation else rotation_init_t
    else:
        seed = torch.zeros(12, device=device, dtype=torch.float32)
        free[3:] = False  # order 1 at most (C1, C12a, C12b)
        freeze_rotation = False
        rotation0 = rotation_init_t

    if verbosity > 0:
        logger.info("Optimizing aberrations...")
    problem = ShiftProblem(
        target_shifts.to(torch.float64)[include],
        include,
        k_sampling,
        wavelength,
        bright_field_mask,
    )
    aberrations64, rotation64 = refine_shift_fit(
        problem,
        torch.where(free, seed, torch.zeros_like(seed)),
        rotation0,
        free,
        fit_rotation=rotation_requires_grad and not freeze_rotation,
        max_iter=max_iter,
        lr=lr,
    )
    aberrations = aberrations64.to(torch.float32)
    rotation_opt = rotation64.to(torch.float32)

    with torch.no_grad():
        final_shifts = aberrations_to_image_shifts(
            aberrations, rotation_opt, k_sampling, wavelength, bright_field_mask.shape
        )[bright_field_mask]

    if verbosity > 0:
        logger.info("Final shifts max: %s", final_shifts.max(0).values.cpu().numpy())
        logger.info("Initial rotation: %.2f deg", rotation_init_t.item())
        logger.info("Final   rotation: %.2f deg", rotation_opt.item())
        logger.info("Final aberrations: %s", aberrations.cpu().numpy())

    return _shift_fit_result(
        aberrations,
        rotation_opt,
        final_shifts,
        target_shifts,
        include,
        residual_warn_rtol,
        well_posed=seed_info is None or bool(freeze_rotation),
    )


def aberrations_and_rotation_from_bright_field_shifts(
    dataset: DatasetVirtualBrightField4DStem,
    fit_rotation: bool = True,
    target_percentage_nonzero_pixels: float = 0.75,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: float | None = None,
    bin_factors: tuple[int, ...] = (2, 1),
    verbosity: int = 0,
    update_dataset: bool = True,
    n_center_indices: int = 25,
    use_com_rotation_prior: str = "auto",
    alignment_method: str = "reference",
    init_method: str = "closed_form",
    peak_selection: str = "argmax",
    correct_order: int = 1,
) -> BrightFieldShiftFit:
    """First determine the bright field shifts, then fit aberrations and rotation.

    Parameters
    ----------
    dataset : DatasetVirtualBrightField4DStem
        Dataset4DStem or DatasetVirtualBrightField4DStem object containing
        the diffraction pattern.
    fit_rotation : bool, optional
        Whether to fit the rotation. Default is True.
    target_percentage_nonzero_pixels : float, optional
        Target percentage of nonzero pixels for the bright field. Default
        is 0.75.
    registration_upsample_factor : int, optional
        The final cross-correlation peak grid is ``1 / registration_upsample_factor``
        scan pixel (``peak_grid_divisor`` of :func:`~scatterem.reconstruction.ptychography.direct.measure_bf_parallax`).
        Default is 10.
    lowpass_fwhm_bright_field : float, optional
        FWHM of the lowpass filter for the bright field. Default is None,
        which applies no lowpass filter.
    bin_factors : tuple of int, optional
        Bin factors for the bright field, applied in order. Default is
        `(2, 1)`.
    verbosity : int, optional
        Verbosity level. Default is 0.
    update_dataset : bool, optional
        Whether to write the fitted aberrations (detector frame, as returned)
        and rotation back onto `dataset.meta`. Default is True. The write
        happens whether or not the fit is ``reliable``, and a vBF shares
        ``meta`` with its parent dataset. To write only a reliable fit, pass
        False, check ``reliable`` and then write the values.
    n_center_indices : int, optional
        Number of center indices to use for the bright field shifts.
        Default is 25.
    use_com_rotation_prior : str, optional
        When "auto" (default), computes a diffraction COM-curl rotation
        prior and uses it as the initial rotation when `meta.rotation` is
        unset (< 1e-6 deg). An out-of-core parent dataset is measured block
        by block, to the same prior as its eager twin. If no pattern has a
        finite COM, a warning is logged and `meta.rotation` is kept. Set to
        "off" to always use `meta.rotation` directly.
    alignment_method : str, optional
        "reference" (default) registers every BF image onto one common
        reference (``shift_estimator="mean_template"`` of
        :func:`~scatterem.reconstruction.ptychography.direct.measure_bf_parallax`). "pairwise" registers
        detector neighbours onto each other and solves for all shifts by
        least squares (``shift_estimator="neighbour_mesh"``), which is more
        robust when large aberration shifts blur the reference image.
    init_method : str, optional
        How the aberration fit is initialised, forwarded to
        `fit_aberrations_and_rotation_to_bright_field_shifts`.
        "closed_form" (default) seeds it with the closed-form C1, C12 and
        rotation estimate; "zeros" starts from zero and caps the fit at
        order 1. Either way the refined orders are set by
        ``correct_order``.
    peak_selection : str, optional
        Which cross-correlation peak is taken as each shift: "argmax"
        (default, the global maximum) or "guided" (for periodic specimens,
        whose correlation is multi-peaked). With "guided" and
        ``alignment_method="reference"`` on a periodic reference image, the
        images (or detector bins) are registered in order of increasing
        detector radius; each one takes the highest peak near a smooth
        prediction from the ones nearer the centre (see
        :mod:`._peak_selection`). The low-pass and the bin factors apply
        as for "argmax". On a non-periodic reference image "guided" is
        "argmax". With ``alignment_method="pairwise"``, "guided" takes,
        among the peaks within 2 % of the highest, the one nearest to zero
        shift. With "argmax" on a periodic specimen a ``UserWarning`` flags
        a likely wrong pick.
    correct_order : int, optional
        Highest aberration order the fit refines, forwarded to
        `fit_aberrations_and_rotation_to_bright_field_shifts`. Default is 1
        (C1, C12 and rotation); a real spherical aberration is then absorbed
        into C1 unless 3 is requested.

    Returns
    -------
    BrightFieldShiftFit
        The aberrations (Angstrom) and rotation (degrees) of the last bin
        level. The aberrations are ``(12,)`` in the DETECTOR frame of
        ``meta.aberrations`` (the frame the direct-ptychography kernels and
        tilt-corrected dark field read): the shift-model-frame fit converted
        with ``scatterem.utils.transfer.shift_model_to_detector_frame`` at
        the fitted rotation, the same as the fit at ``rotation == 0``.
        ``relative_residual`` is the largest over the bin levels, and
        ``well_posed`` is False if any level was not well posed, so a failed
        earlier level makes the result not ``reliable``. It unpacks as
        ``aberrations, rotation = fit``. When ``reliable`` is False, use the
        fallback order in the `BrightFieldShiftFit` docstring.

    Raises
    ------
    ValueError
        If the dataset is not a valid Dataset4DStem object, if
        ``bin_factors`` is empty, or if the vBF does not carry its detector
        sampling (a 4-entry ``sampling`` and a 2-D ``bright_field_mask``).
    """
    if not bin_factors:
        raise ValueError(f"bin_factors must not be empty, got {bin_factors!r}")
    validate_peak_selection(peak_selection)
    device = dataset.device
    wavelength = _require_meta(dataset).wavelength
    vBF = dataset
    # Normalized center reference from the first n_center radius-ordered
    # columns only — identical values to slicing the full normalized stack
    # (the elementwise per-pixel divide commutes with column selection), and
    # it works on the streaming provider whose stack may live on the CPU.
    inds_r = vBF.bright_field_inds_ordered_by_radius
    norm_cols = vBF.diffraction_pattern_mean_normalized[
        inds_r[:n_center_indices, 0], inds_r[:n_center_indices, 1]
    ].to(device)
    center_cols = (
        vBF.array[..., :n_center_indices].to(device) / norm_cols[None, None, :]
    )
    G_ref = G_ref_initial = torch.fft.fft2(center_cols.mean(-1))

    sampling = torch.as_tensor(dataset.sampling[:2], device=device, dtype=torch.float32)
    detector_dr = _detector_dr(dataset)
    rot_value = float(_require_meta(dataset).rotation)
    if use_com_rotation_prior == "auto" and abs(rot_value) < 1e-6:
        rot_value = _com_rotation_prior(dataset, rot_value, verbosity)
    rot_tensor = torch.tensor([rot_value], device=device, dtype=torch.float32)
    global_shifts = torch.zeros(
        (len(vBF.bright_field_inds_centered), 2), device=device, dtype=torch.float32
    )
    order = vBF.bright_field_inds_radial_order.to(device=device)
    kind = "pairwise" if alignment_method == "pairwise" else "reference-based"
    estimator = "neighbour_mesh" if kind == "pairwise" else "mean_template"
    spectra = _VbfSpectra(vBF)
    sigma_px = (
        None
        if lowpass_fwhm_bright_field is None
        else float(lowpass_fwhm_bright_field) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    )
    pbar = tqdm(bin_factors, desc=f"Self-calibration ({kind})", disable=not verbosity)
    level_fits: list[BrightFieldShiftFit] = []
    fitted_global_shifts = measured_shifts = global_shifts
    try:
        for bin in pbar:
            measured_incremental_shifts_px = level_shifts(
                spectra,
                vBF,
                G_ref,
                peak_selection=peak_selection,
                shift_estimator=estimator,
                binning=max(1, int(bin)),
                peak_grid_divisor=int(registration_upsample_factor),
                lowpass_sigma_px=sigma_px,
                verbosity=verbosity,
                measure=measure_bf_parallax,  # the module attribute tests patch
            ).to(device=device, dtype=torch.float32)
            measured_incremental_shifts_px -= measured_incremental_shifts_px.mean(dim=0)
            inc_shifts_angstroms = measured_incremental_shifts_px * sampling
            pbar.set_postfix(
                {"Res. Shift [A]": f"{inc_shifts_angstroms.max().item():.2f}"}
            )
            global_shifts += inc_shifts_angstroms
            measured_shifts = global_shifts.clone()
            shift_fit = fit_aberrations_and_rotation_to_bright_field_shifts(
                global_shifts,
                detector_dr,
                wavelength,
                vBF.bright_field_mask,
                rotation_init=rot_tensor,
                rotation_requires_grad=fit_rotation,
                init_method=init_method,
                correct_order=correct_order,
                verbosity=verbosity,
            )
            level_fits.append(shift_fit)
            fitted_global_shifts = aberrations_to_image_shifts(
                shift_fit.aberrations,
                shift_fit.rotation,
                detector_dr,
                wavelength,
                vBF.bright_field_mask.shape,
            )[vBF.bright_field_mask]
            global_shifts = fitted_global_shifts

            # Add the measured shifts to the vBF's alignment OVERLAY, so the next
            # bin level and the reference see the aligned G without a change to
            # the (possibly shared) cache. Later steps correct the parallax with
            # the FITTED aberrations; a kept overlay would remove it twice (in
            # "combined" it pulled the autofocus C1 to 0), hence the ``finally``.
            vBF.add_alignment_shifts(measured_incremental_shifts_px[order])
            G_ref = vBF.get_G_chunk(0, n_center_indices).mean(dim=-1)
    finally:
        vBF.clear_alignment_shifts()
    shift_fit = _worst_of_levels(level_fits)
    warn_if_multi_peaked(
        G_ref_initial,
        measured_shifts / sampling,
        fitted_global_shifts / sampling,
        peak_selection,
    )

    if verbosity > 0:
        from ._plotting import show_bright_field_shift_maps

        logger.info(
            "Interpolated shifts for %d bright field indices",
            len(global_shifts),
        )
        show_bright_field_shift_maps(vBF, global_shifts, fitted_global_shifts)

    return _publish_detector_frame(vBF, shift_fit, update_dataset)


class _VbfSpectra:
    """The BF image spectra of a vBF provider as a ``SpectrumSource``.

    Index ``i`` is BF pixel ``i`` in the original order of
    ``bright_field_inds_centered``; the provider stores its BF axis in radial
    order, so :meth:`read` maps the indices through the radial rank. The
    provider applies its alignment overlay at fetch time.
    """

    def __init__(self, vbf: DatasetVirtualBrightField4DStem) -> None:
        self._vbf = vbf
        order = vbf.bright_field_inds_radial_order.to(device="cpu", dtype=torch.long)
        self._rank = torch.argsort(order)
        self.image_count = int(order.numel())
        self.scan_grid_shape = (int(vbf.array.shape[0]), int(vbf.array.shape[1]))
        self.run_device = torch.device(vbf.device)

    def read(self, indices) -> torch.Tensor:
        idx = self._rank[torch.as_tensor(list(indices), dtype=torch.long)]
        cols = self._vbf.get_G_columns(idx.to(self.run_device))
        return torch.permute(cols, (2, 0, 1)).to(self.run_device)


def _publish_detector_frame(
    vBF: DatasetVirtualBrightField4DStem,
    fit: BrightFieldShiftFit,
    update_dataset: bool,
) -> BrightFieldShiftFit:
    """The shift-model-frame fit turned to the detector frame, stored if asked.

    Everything before this models the measured shifts, in the shift model's
    frame; the vector returned and written into ``meta.aberrations`` is the
    detector-frame one its readers use, turned at the FINAL rotation (the
    identity, bit for bit, at rotation 0). The fit's quality fields are kept.
    """
    fit = replace(
        fit, aberrations=shift_model_to_detector_frame(fit.aberrations, fit.rotation)
    )
    if update_dataset:
        meta = _require_meta(vBF)
        meta.aberrations = Aberrations(array=fit.aberrations)
        meta.rotation = fit.rotation.item()
        meta.defocus_guess = 0.0  # C10 is in aberrations alone
    return fit
