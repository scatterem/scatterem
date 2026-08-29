import math
import warnings
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from tqdm.auto import tqdm

import scatterem.vis as vis
from scatterem.nn.functional.ptychography import (
    build_aberration_bf_factor,
    correct_aberrations_inplace,
    correct_aberrations_kouter,
    correct_aberrations_kouter_planes,
    correct_aberrations_ksum,
)
from scatterem.nn.functional.ptychography import (
    phase_contrast_transfer_function as _phase_contrast_transfer_function,
)
from scatterem.utils.calibration import (
    _calculate_intensities_center_of_mass,
    _solve_for_center_of_mass_relative_rotation,
)
from scatterem.utils.data.aberrations import Aberrations
from scatterem.utils.data.datasets import (
    Dataset4dstem,
    DatasetVirtualBrightField4dstem,
)
from scatterem.utils.subpixel import (
    make_neighbor_pairs,
    pairwise_relative_shifts,
    relative_shifts,
    synchronize_shifts,
)
from scatterem.utils.grids import fft_frequencies_2d
from scatterem.utils.aberration_basis import cartesian_to_polar
from scatterem.utils.transfer import (
    aberrations_to_image_shifts,
    double_and_triple_pixel_counts_torch,
)
from scatterem.vis.normalization import NormalizationConfig

PhaseSignMode = Literal["preserve", "positive", "negative"]


def _phase_image_skewness(
    phase_image: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Return normalized third moment used to choose the physical phase branch."""
    phase = phase_image.detach().to(torch.float32)
    centered = phase - phase.mean()
    std = centered.square().mean().sqrt()
    # The two guards are resolved ON DEVICE with ``torch.where`` rather than by
    # ``if <0-dim cuda tensor>``.  Each python ``if`` on a device scalar is a
    # blocking ``cudaStreamSynchronize``, and this function sits at the TAIL of
    # the reconstruction, so the drains do not overlap with anything -- they
    # serialise as sync / launch a handful of tiny kernels / sync again.  The
    # returned values are unchanged in every case: the fallback is still exactly
    # ``0.0`` where ``std`` is non-finite or ``<= eps``, or where the ratio is
    # non-finite.  ``std.pow(3)`` cannot raise, so computing the ratio
    # unconditionally is safe -- a zero or non-finite ``std`` makes it
    # non-finite, which the same mask replaces.
    skewness = centered.pow(3).mean() / std.pow(3)
    finite = torch.isfinite(std) & (std > eps) & torch.isfinite(skewness)
    return torch.where(finite, skewness, 0.0)


def _orient_phase_image(
    phase_image: torch.Tensor,
    phase_sign: PhaseSignMode = "positive",
    verbosity: int = 0,
) -> torch.Tensor:
    """Resolve the global direct-ptycho phase sign ambiguity.

    Direct ptychography reconstructs a signed phase channel, while the
    total-variation aberration objective is invariant under ``phase -> -phase``.
    For normal positive projected electrostatic potentials, atom columns should
    give a positive-skew phase image.
    """
    if phase_sign == "preserve":
        return phase_image
    if phase_sign not in ("positive", "negative"):
        raise ValueError(
            "phase_sign must be one of 'preserve', 'positive', or 'negative', "
            f"got {phase_sign!r}"
        )

    skewness = _phase_image_skewness(phase_image)
    target_sign = 1.0 if phase_sign == "positive" else -1.0
    if verbosity > 0:
        # Only the diagnostic path pays the D2H sync.
        should_flip = float(skewness.detach().cpu()) * target_sign < 0.0
        action = "flipping" if should_flip else "preserving"
        print(
            f"Direct ptycho phase skewness: {float(skewness.detach().cpu()):+.4f}; "
            f"{action} phase sign for {phase_sign} contrast"
        )
        return -phase_image if should_flip else phase_image
    # Apply the sign as a device-side multiply by exactly +-1.0, which is
    # bit-exact for every float and preserves NaN/inf, so the returned array is
    # identical to the branched form while the decision never reaches the host.
    sign = torch.where(skewness * target_sign < 0.0, -1.0, 1.0)
    return phase_image * sign


# radial power of k for each of the 12 cartesian aberration coefficients
_ABERRATION_RADIAL_POWER = torch.tensor(
    [2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=torch.float32
)


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
    p = _ABERRATION_RADIAL_POWER.to(device=device)
    alpha = max(float(semiconvergence_angle), 1e-9)
    return float(wavelength) / (alpha**p)


def _resolve_aberration_stages(
    correction_method: str, correct_order: int
) -> tuple[bool, bool]:
    """Decide which aberration-fitting stages ``determine_aberrations`` runs.

    Returns ``(do_bright_field, do_autofocus)``.

    - ``"bright-field-shifts"``: BF fit only -- fast, deterministic low-order (defocus + 2-fold
      astigmatism) + scan rotation. Cannot fit higher orders (they are degenerate in BF shifts).
    - ``"autofocus"``: sharpness autofocus only (sparsity / L4 metric by default). Fits higher
      orders but fixes rotation (the caller must already have a correct ``meta.rotation``).
      The legacy name ``"total-variation"`` is a deprecated alias for this.
    - ``"combined"`` (default): BF fit for rotation + low-order, then the sharpness autofocus to
      extend to higher orders. The autofocus stage runs only when ``correct_order >= 2`` is
      requested, so the common low-order case stays as fast as the bright-field-shifts path.
    """
    if correction_method == "total-variation":
        warnings.warn(
            "correction_method='total-variation' is deprecated; use 'autofocus' "
            "(the autofocus metric defaults to sparsity, not total variation).",
            DeprecationWarning,
            stacklevel=2,
        )
        correction_method = "autofocus"
    valid = ("bright-field-shifts", "autofocus", "combined")
    if correction_method not in valid:
        raise ValueError(
            f"correction_method must be one of {valid} (or the deprecated "
            f"'total-variation'), got {correction_method!r}"
        )
    do_bf = correction_method in ("bright-field-shifts", "combined")
    do_tv = correction_method == "autofocus" or (
        correction_method == "combined" and int(correct_order) >= 2
    )
    return do_bf, do_tv


def _windowed_roi(
    image: torch.Tensor,
    roi_center: tuple[int, int],
    roi_shape: tuple[int, int],
) -> torch.Tensor:
    """Crop a centred region of interest from ``image`` and apply a separable Hann window.

    Robust to ``roi_shape`` larger than the image and to off-image centres: the ROI is clamped to
    fit inside the image, and the Hann window is sized to the *actual* crop (so it never mismatches
    the cropped shape). The window removes the hard-crop boundary discontinuity that would otherwise
    inflate the total-variation sharpness metric.
    """
    h, w = int(image.shape[-2]), int(image.shape[-1])
    ry = min(int(roi_shape[0]), h)
    rx = min(int(roi_shape[1]), w)
    # clamp the centre so the [y0, y0+ry) window lies fully inside the image
    cy = min(max(int(roi_center[0]), ry // 2), h - (ry - ry // 2))
    cx = min(max(int(roi_center[1]), rx // 2), w - (rx - rx // 2))
    y0 = cy - ry // 2
    x0 = cx - rx // 2
    roi = image[..., y0 : y0 + ry, x0 : x0 + rx]
    wy = torch.hann_window(ry, periodic=False, device=image.device, dtype=image.dtype)
    wx = torch.hann_window(rx, periodic=False, device=image.device, dtype=image.dtype)
    return roi * torch.outer(wy, wx)


def _image_sharpness(image: torch.Tensor, kind: str = "sparsity") -> torch.Tensor:
    """Differentiable image-sharpness metric (higher = sharper); maximised by the aberration fit.

    - ``"sparsity"`` (default): L4 / kurtosis-like ``mean(I^4) / mean(I^2)^2``. Rewards sparse,
      high-contrast atom-column spikes; aberration-induced ringing fills the background and
      *lowers* it, so the optimum sits at the true aberration. Empirically **unbiased for
      higher-order aberrations (coma, Cs) at adequate NA**, where total variation is biased (ringing
      inflates TV). Scale- and sign-invariant (even powers).
    - ``"tv"``: total variation (mean |gradient|). Legacy; gameable by high-frequency ringing, so its
      optimum is displaced for higher orders.
    """
    if kind == "sparsity":
        i2 = (image**2).mean()
        i4 = (image**4).mean()
        return i4 / (i2 * i2 + 1e-12)
    if kind == "tv":
        dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
        dx = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
        return dy + dx
    raise ValueError(f"sharpness_metric must be 'sparsity' or 'tv', got {kind!r}")


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
):
    """Preconditioned, regularized, multi-start, keep-best maximization of a sharpness metric.

    Optimizes ``aberr = seed + u * scales`` in scaled (~radian) units ``u`` (isotropic landscape),
    regularizing ``u`` toward 0 (the seed). A coarse coordinate-descent sweep over every free
    coefficient (defocus first) seeds LBFGS to escape the multimodal landscape and reach higher-order
    optima. The best (highest-sharpness) aberration ever seen is returned
    (keep-best); the seed is always a candidate, so the result is never worse than the seed.

    Returns (best_aberrations(12,), info dict: seed_tv, best_tv, improved_over_seed, converged).
    """
    device = seed_aberrations.device
    seed = seed_aberrations.detach().clone()
    scales = scales.to(device)
    free = free_mask.to(device)

    def aberr_of(u):
        return seed + u * scales * free

    best = {"tv": None, "aberr": seed.clone()}

    def consider(aberr):
        with torch.no_grad():
            tv = float(evaluate_tv(aberr))
        if best["tv"] is None or tv > best["tv"]:
            best["tv"] = tv
            best["aberr"] = aberr.detach().clone()
        return tv

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
        base_tv = consider(aberr_of(u0))
        for idx in sweep_order:
            if not bool(free[idx]):
                continue
            best_d = float(u0[idx])
            for d in grid:
                u = u0.clone()
                u[idx] = d
                tv = consider(aberr_of(u))
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

    info = {
        "seed_tv": seed_tv,
        "best_tv": best["tv"],
        "improved_over_seed": best["tv"] > seed_tv + 1e-6,
        "converged": True,
    }
    return best["aberr"], info


def plot_bright_field_shifts(
    query_points,
    shift_values,
    sampling,
    wavelength,
    opt_rotation,
    arrow_scale,
    suptitle=None,
):
    """
    Plot bright field shifts before and after rotation correction.
    Args:
        query_points: torch.Tensor - query points (2D: N x 2), order by radius in the bright field, increasing order.
        shift_values: torch.Tensor - shift values (2D: N x 2), order by radius in the bright field, increasing order.
        sampling: torch.Tensor - sampling (2D: 2)
        wavelength: float - wavelength in Angstroms
        opt_rotation: float - rotation in degrees. The default is 0.
        arrow_scale: float - arrow scale. The default is 25e-2.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)
    # ax2 = ax[0].twinx()  # Create right axis
    ax3 = ax[0].twiny()  # Create top axis
    shift_values = shift_values.cpu().numpy()
    # Draw arrows for each position
    q_query_points = query_points.cpu().numpy() * sampling[0].cpu().numpy()

    ax[0].quiver(
        q_query_points[:, 1],
        q_query_points[:, 0],
        shift_values[:, 1] * arrow_scale,
        shift_values[:, 0] * arrow_scale,
        fc="r",
        ec="r",
        label="Shift vectors",
    )
    ax[0].scatter(
        q_query_points[:, 1],
        q_query_points[:, 0],
        color="b",
        s=1,
        marker="x",
        label="Query points",
    )

    f = 1.2
    # Calculate k ranges
    k_min_x = q_query_points[:, 1].min()
    k_max_x = q_query_points[:, 1].max()
    k_min_y = q_query_points[:, 0].min()
    k_max_y = q_query_points[:, 0].max()

    # Calculate alpha ranges (alpha = k*lambda)
    alpha_min_x = k_min_x * wavelength * 1e3  # Convert to mrad
    alpha_max_x = k_max_x * wavelength * 1e3
    alpha_min_y = k_min_y * wavelength * 1e3
    alpha_max_y = k_max_y * wavelength * 1e3

    # Create 5 evenly spaced ticks centered at zero
    k_ticks_x = np.linspace(k_min_x, k_max_x, 5)
    k_ticks_y = np.linspace(k_min_y, k_max_y, 5)
    alpha_ticks_x = np.linspace(alpha_min_x, alpha_max_x, 5)
    alpha_ticks_y = np.linspace(alpha_min_y, alpha_max_y, 5)

    # Set limits and ticks
    ax[0].set_ylim(k_min_y * f, k_max_y * f)
    ax[0].set_xlim(k_min_x * f, k_max_x * f)
    # ax2.set_ylim(alpha_min_y * f, alpha_max_y * f)
    ax3.set_xlim(alpha_min_x * f, alpha_max_x * f)

    ax[0].set_title("Bright Field Shifts")
    ax[0].set_ylabel(r"$k_y$ [$\mathrm{\AA^{-1}}$]")
    ax[0].set_xlabel(r"$k_x$ [$\mathrm{\AA^{-1}}$]")

    ax[0].set_yticks(k_ticks_y)
    ax[0].set_xticks(k_ticks_x)
    # ax2.set_yticks(alpha_ticks_y)
    ax3.set_xticks(alpha_ticks_x)

    # ax2.set_yticklabels([f'{a:.1f}' for a in alpha_ticks_y])
    ax3.set_xticklabels([f"{a:.1f}" for a in alpha_ticks_x])

    # ax2.set_ylabel(r"$\alpha_x$ [mrad]")
    ax3.set_xlabel(r"$\alpha_x$ [mrad]")

    ax[0].legend()

    # Create twin axes
    ax2 = ax[1].twinx()  # Create right axis
    ax3 = ax[1].twiny()  # Create top axis

    # Calculate rotation matrix
    angle_rad = torch.deg2rad(-opt_rotation).item()
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Apply rotation to query points
    q_query_points_rot = np.dot(q_query_points, rot_matrix)

    ax[1].quiver(
        q_query_points_rot[:, 1],
        q_query_points_rot[:, 0],
        shift_values[:, 1] * arrow_scale,
        shift_values[:, 0] * arrow_scale,
        fc="r",
        ec="r",
        label=f"Shift vectors (bin {bin})",
    )
    ax[1].scatter(
        q_query_points_rot[:, 1],
        q_query_points_rot[:, 0],
        color="b",
        s=1,
        marker="x",
        label="Query points",
    )

    # Calculate alpha ranges (alpha = k*lambda)
    alpha_min_x = k_min_x * wavelength * 1e3  # Convert to mrad
    alpha_max_x = k_max_x * wavelength * 1e3
    alpha_min_y = k_min_y * wavelength * 1e3
    alpha_max_y = k_max_y * wavelength * 1e3

    # Create 5 evenly spaced ticks centered at zero
    k_ticks_x = np.linspace(k_min_x, k_max_x, 5)
    k_ticks_y = np.linspace(k_min_y, k_max_y, 5)
    alpha_ticks_x = np.linspace(alpha_min_x, alpha_max_x, 5)
    alpha_ticks_y = np.linspace(alpha_min_y, alpha_max_y, 5)

    # Set limits and ticks
    ax[1].set_ylim(k_min_y * f, k_max_y * f)
    ax[1].set_xlim(k_min_x * f, k_max_x * f)
    ax2.set_ylim(alpha_min_y * f, alpha_max_y * f)
    ax3.set_xlim(alpha_min_x * f, alpha_max_x * f)

    ax[1].set_title("Bright Field Shifts (Rotated)")
    # ax[1].set_ylabel(r"$k_y$ [$\mathrm{\AA^{-1}}$]")
    ax[1].set_xlabel(r"$k_x$ [$\mathrm{\AA^{-1}}$]")

    # ax[1].set_yticks(k_ticks_y)
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks(k_ticks_x)
    ax2.set_yticks(alpha_ticks_y)
    ax3.set_xticks(alpha_ticks_x)

    ax2.set_yticklabels([f"{a:.1f}" for a in alpha_ticks_y])
    ax3.set_xticklabels([f"{a:.1f}" for a in alpha_ticks_x])

    ax2.set_ylabel(r"$\alpha_y$ [mrad]")
    ax3.set_xlabel(r"$\alpha_x$ [mrad]")

    # ax[1].legend()
    plt.tight_layout()
    plt.show()


def _rotation_matrix_2x2(angle: torch.Tensor) -> torch.Tensor:
    """2x2 rotation matrix R(angle) for a scalar-tensor angle in radians."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def estimate_scan_rotation_from_com(
    intensities: torch.Tensor,
    reciprocal_sampling: tuple[float, float] = (1.0, 1.0),
    dp_mask: torch.Tensor | None = None,
) -> float:
    """Independent scan<->detector rotation estimate (degrees) from the diffraction center-of-mass curl.

    Computes the per-scan-position diffraction COM and finds the rotation making the COM field
    curl-free, via the (validated) calibration utilities. Used as a prior for the bright-field
    aberration fit when ``meta.rotation`` is unset.

    Args:
        intensities: (Rx, Ry, Qx, Qy) diffraction intensities.
        reciprocal_sampling: detector reciprocal sampling (dx, dy).
        dp_mask: optional (Qx, Qy) detector mask.
    Returns:
        Estimated scan rotation in degrees.
    """
    device = intensities.device
    (
        com_measured_x,
        com_measured_y,
        _,
        _,
        com_normalized_x,
        com_normalized_y,
    ) = _calculate_intensities_center_of_mass(
        intensities, reciprocal_sampling, dp_mask=dp_mask, device=device
    )
    rotation_rad, _transpose, _cx, _cy = _solve_for_center_of_mass_relative_rotation(
        com_measured_x,
        com_measured_y,
        com_normalized_x,
        com_normalized_y,
        plot_rotation=False,
        verbose=False,
        device=device,
    )
    return float(np.rad2deg(rotation_rad))


def fit_low_order_aberrations_and_rotation_closed_form(
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    include_in_fit_mask: torch.Tensor | None = None,
    rotation_prior: torch.Tensor | float | None = None,
    well_posed_rtol: float = 1e-3,
    robust: bool = False,
    robust_iter: int = 5,
    huber_c: float = 1.345,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Closed-form low-order aberration + scan-rotation estimate from bright-field shifts.

    Solves the linear map ``shift = M @ q`` over bright-field pixels, decomposes
    ``M = P @ R(alpha)`` in closed form, resolves the (P, alpha) <-> (-P, alpha +- 180 deg)
    degeneracy with ``rotation_prior``, and extracts (C1, C12a, C12b, rotation).

    Args:
        target_shifts: (N, 2) bright-field shifts in Angstrom, row-major over ``bright_field_mask``.
        k_sampling: detector reciprocal sampling used to build the q-grid (same as the forward model).
        wavelength: electron wavelength in Angstrom.
        bright_field_mask: (H, W) bool mask of bright-field pixels.
        include_in_fit_mask: optional (N,) bool subset of BF pixels to include.
        rotation_prior: scan rotation prior in degrees (e.g. ``meta.rotation``); used only to pick
            the 180-degree branch. If None, the canonical ``C1 >= 0`` branch is returned.
        well_posed_rtol: threshold on ``2*lambda*|C1| / ||M||``; below this, rotation is
            unrecoverable (near-zero defocus) and ``well_posed`` is False.

    Returns:
        (aberrations[12], rotation_deg[1], info) where only indices 0,1,2 (C1, C12a, C12b) are
        populated; info has ``residual`` (RMS shift residual, Angstrom), ``well_posed`` (bool),
        ``branch_flipped`` (bool).
    """
    device = target_shifts.device
    shape = tuple(bright_field_mask.shape)
    if bright_field_mask.device != device:
        bright_field_mask = bright_field_mask.to(device)

    q = torch.fft.fftshift(
        fft_frequencies_2d(shape, k_sampling, False, device=device), dim=(-2, -1)
    )  # (2, H, W) = [k0, k1]
    K = q[:, bright_field_mask].T  # (N, 2)
    S = target_shifts

    if include_in_fit_mask is not None:
        # Accept a (N,) per-pixel mask or the existing (N, 2) per-component default; reduce to
        # per-pixel (a pixel is whole-in/whole-out in practice, so column 0 suffices).
        m = include_in_fit_mask.to(device).reshape(K.shape[0], -1)[:, 0].bool()
        K = K[m]
        S = S[m]

    # Solve in float64 for a robust 2x2 algebra, regardless of input dtype.
    K64 = K.to(torch.float64)
    S64 = S.to(torch.float64)
    Mt = torch.linalg.lstsq(K64, S64).solution  # (2, 2): S = K @ Mt
    inlier_frac = 1.0
    if robust:
        # IRLS with Huber weights on the PER-POINT (per-BF-pixel) shift residual.
        # Down-weights outlier shift vectors (e.g. cross-correlation mis-locks on the
        # repetitive lattice) that otherwise bias the plain least-squares fit. Scale
        # from the MAD of the residual magnitudes; delta = huber_c * scale.
        for _ in range(int(robust_iter)):
            rn = (K64 @ Mt - S64).norm(dim=1)  # (N,) per-point residual magnitude
            scale = 1.4826 * torch.median(rn) + 1e-12
            delta = huber_c * scale
            w = torch.where(rn <= delta, torch.ones_like(rn), delta / (rn + 1e-30))
            sw = w.sqrt()[:, None]
            Mt = torch.linalg.lstsq(K64 * sw, S64 * sw).solution
        rn = (K64 @ Mt - S64).norm(dim=1)
        scale = 1.4826 * torch.median(rn) + 1e-12
        delta = huber_c * scale
        w = torch.where(rn <= delta, torch.ones_like(rn), delta / (rn + 1e-30))
        inlier_frac = float((rn <= delta).float().mean().item())
        # weighted RMS residual (inlier fit quality)
        residual = ((w * rn.square()).sum() / (w.sum() + 1e-30)).sqrt()
    else:
        residual = (K64 @ Mt - S64).square().mean().sqrt()
    M = Mt.mT  # so that shift = M @ k

    # Signed 2x2 polar: M = P @ R(alpha0), alpha0 canonical (forces trace(P0) >= 0 -> C1 >= 0).
    alpha0 = torch.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    P0 = M @ _rotation_matrix_2x2(alpha0).mT

    mag = torch.sqrt(
        (M[0, 0] + M[1, 1]) ** 2 + (M[1, 0] - M[0, 1]) ** 2
    )  # = 2*lam*|C1|
    well_posed = bool((mag / (M.norm() + 1e-30) > well_posed_rtol).item())

    alpha = alpha0
    P = P0
    branch_flipped = False
    if rotation_prior is not None and well_posed:
        prior = (
            float(rotation_prior.item())
            if torch.is_tensor(rotation_prior)
            else float(rotation_prior)
        )
        rot0 = math.degrees(float(alpha0.item()))

        def _ang_dist(a, b):
            return abs((a - b + 180.0) % 360.0 - 180.0)

        if _ang_dist(rot0 + 180.0, prior) < _ang_dist(rot0, prior):
            alpha = alpha0 + math.pi
            P = -P0
            branch_flipped = True

    lam = wavelength
    C1 = (P[0, 0] + P[1, 1]) / (2 * lam)
    C12a = (P[1, 1] - P[0, 0]) / (2 * lam)
    C12b = (P[0, 1] + P[1, 0]) / (2 * lam)
    rotation_deg = ((torch.rad2deg(alpha) + 180.0) % 360.0) - 180.0

    aberrations = torch.zeros(12, device=device, dtype=torch.float32)
    aberrations[0] = C1.to(torch.float32)
    aberrations[1] = C12a.to(torch.float32)
    aberrations[2] = C12b.to(torch.float32)

    info = {
        "residual": float(residual.item()),
        "well_posed": well_posed,
        "branch_flipped": branch_flipped,
        "inlier_frac": inlier_frac,
    }
    return aberrations, rotation_deg.reshape(1).to(torch.float32), info


def fit_aberrations_and_rotation_to_bright_field_shifts(
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    include_in_fit_mask: torch.Tensor | None = None,
    rotation_init: torch.Tensor | None = None,
    rotation_requires_grad: bool = True,
    init_method: str = "closed_form",
    residual_warn_rtol: float = 0.15,
    max_iter: int = 250,
    lr: float = 1e-1,
    verbosity: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit the aberrations and rotation to the bright field shifts.

    Args:
        target_shifts: torch.Tensor - target bright field shifts (2D: N x 2), order by radius in the bright field, increasing order.
        k_sampling: torch.Tensor - k-space sampling (2D: 2)
        wavelength: float - wavelength in Angstroms
        bright_field_mask: torch.Tensor - bright field mask (2D: H x W)
        include_in_fit_mask: torch.Tensor | None - mask for the bright field shifts to include in the fit. If None, all bright field shifts are included.
        rotation_init: torch.Tensor | None - initial rotation in degrees. If None, the rotation is initialized to 0.
        rotation_requires_grad: bool - whether to require the rotation to be optimized. The default is True.
        init_method: str - initialization method. Either 'closed_form' or 'zeros'. The default is 'closed_form'.
        residual_warn_rtol: float - relative residual threshold above which a UserWarning is emitted.
            The relative residual is RMS(final_shifts - target_shifts) / RMS(target_shifts). Default 0.15.
        max_iter: int - maximum number of iterations. The default is 50.
        lr: float - learning rate. The default is 1.
        verbosity: int - verbosity level. The default is 0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: aberrations and rotation. The aberrations are in units of Angstroms. The rotation is in degrees.
    """

    # Check that all tensors are on same device and dtype
    device = target_shifts.device
    target_shifts = target_shifts.to(torch.float32)

    if bright_field_mask.device != device:
        bright_field_mask = bright_field_mask.to(device)

    bright_field_inds = torch.argwhere(bright_field_mask)

    if include_in_fit_mask is not None:
        if include_in_fit_mask.device != device:
            include_in_fit_mask = include_in_fit_mask.to(device)
    else:
        include_in_fit_mask = torch.ones(
            bright_field_inds.shape, device=device, dtype=torch.bool
        )

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
    use_closed_form = init_method == "closed_form"
    seed_info = None
    freeze_rotation = False
    if use_closed_form:
        seed_aberr, seed_rot, seed_info = (
            fit_low_order_aberrations_and_rotation_closed_form(
                target_shifts,
                k_sampling,
                wavelength,
                bright_field_mask,
                include_in_fit_mask=include_in_fit_mask,
                rotation_prior=rotation_init_t,
            )
        )
        aberrations = seed_aberr.detach().clone().requires_grad_(True)
        if seed_info["well_posed"]:
            rotation_opt = seed_rot.detach().clone()
            freeze_rotation = True  # rotation determined; refine aberrations only
        else:
            # near-zero defocus: rotation unrecoverable -> keep the prior, leave it free
            rotation_opt = rotation_init_t.clone()
    else:
        aberrations = torch.zeros(
            12, device=device, requires_grad=True, dtype=torch.float32
        )

        def zero_grad_hook(grad):
            if grad is not None:
                grad[3:] = 0
            return grad

        aberrations.register_hook(zero_grad_hook)
        rotation_opt = rotation_init_t.clone()

    opt_params = [aberrations]
    fit_rotation_now = rotation_requires_grad and not freeze_rotation
    if fit_rotation_now:
        rotation_opt.requires_grad = True
        opt_params.append(rotation_opt)
    optimizer = torch.optim.LBFGS(
        opt_params,
        lr=lr,
        max_iter=max_iter,
        max_eval=None,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        predicted_shifts = aberrations_to_image_shifts(
            aberrations, rotation_opt, k_sampling, wavelength, bright_field_mask.shape
        )
        predicted_shifts = predicted_shifts[bright_field_mask]
        loss = torch.nn.functional.huber_loss(
            predicted_shifts[include_in_fit_mask], target_shifts[include_in_fit_mask]
        )
        loss.backward()
        return loss

    if verbosity > 0:
        print("Optimizing aberrations...")
    loss = optimizer.step(closure)

    with torch.no_grad():
        final_shifts = aberrations_to_image_shifts(
            aberrations, rotation_opt, k_sampling, wavelength, bright_field_mask.shape
        )
        final_shifts = final_shifts[bright_field_mask]
        final_shifts_np = final_shifts.cpu().numpy()

    if verbosity > 0:
        print("\nFinal shifts max:", final_shifts_np.max(0))
        print(f"Initial rotation: {rotation_init_t.detach().item():.2f} deg")
        print(f"Final   rotation: {rotation_opt.detach().item():.2f} deg")
        print(f"Final aberrations: {aberrations.detach().cpu().numpy()}")

    with torch.no_grad():
        masked_target = target_shifts
        resid = final_shifts
        if include_in_fit_mask is not None:
            m = include_in_fit_mask.reshape(final_shifts.shape[0], -1)[:, 0].bool()
            resid = resid[m]
            masked_target = masked_target[m]
        rms_resid = (resid - masked_target).pow(2).mean().sqrt()
        rms_target = masked_target.pow(2).mean().sqrt().clamp_min(1e-12)
        rel = (rms_resid / rms_target).item()
    if rel > residual_warn_rtol:
        warnings.warn(
            f"Aberration fit residual is high (relative residual {rel:.2f} > "
            f"{residual_warn_rtol}); the fit may be unreliable.",
            UserWarning,
        )
    if seed_info is not None and not seed_info["well_posed"]:
        warnings.warn(
            "Near-zero defocus: scan rotation is not determined by the bright-field shifts; "
            "using the rotation prior. Aberration/rotation estimate is degenerate.",
            UserWarning,
        )

    return aberrations.detach(), rotation_opt.detach()


def aberrations_and_rotation_from_bright_field_shifts(
    dataset: DatasetVirtualBrightField4dstem,
    fit_rotation: bool = True,
    target_percentage_nonzero_pixels: float = 0.75,
    n_batches: int = 25,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: Optional[float] = None,
    bin_factors: tuple[int, ...] = (2, 1),
    verbosity: int = 0,
    update_dataset: bool = True,
    n_center_indices: int = 25,
    use_com_rotation_prior: str = "auto",
    alignment_method: str = "reference",
    init_method: str = "closed_form",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    First determine the bright field shifts, then fit aberrations and rotation to the bright field shifts.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        fit_rotation: Whether to fit the rotation. The default is True.
        bright_field_mask_threshold: Threshold to determine the bright field pixels, relative to max=1. The default is 0.5.
        target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field. The default is 0.75.
        n_batches: Number of batches for the bright field shifts. The default is 25.
        registration_upsample_factor: Upsampling factor for the registration. The default is 10.
        lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field. The default is None, which means no lowpass filter.
        bin_factors: Bin factors for the bright field. The default is (2, 1).
        verbosity: Verbosity level. The default is 0.
        update_dataset: Whether to update the dataset. The default is True.
        n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
        use_com_rotation_prior: When ``"auto"`` (default), computes a diffraction COM-curl rotation
            prior and uses it as the initial rotation when ``meta.rotation`` is unset (< 1e-6 deg).
            Set to ``"off"`` to always use ``meta.rotation`` directly.
        alignment_method: ``"reference"`` (default) uses the existing single-reference
            cross-correlation path (``bright_field_shifts``).  ``"pairwise"`` uses
            graph-synchronised pairwise CC (``bright_field_shifts_pairwise``), which is
            more robust when large aberration shifts cause the reference image to be
            blurred.  The default preserves the previous behaviour exactly.
        init_method: How the aberration fit is initialised, forwarded to
            ``fit_aberrations_and_rotation_to_bright_field_shifts``. ``"closed_form"``
            (default) fits the full order-2 vector. ``"zeros"`` freezes coefficients
            >= 3, i.e. order 1 (C1 + C12 only), which is the well-conditioned choice on
            low-defocus data where order 2 diverges to large spurious coma/trefoil.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Aberrations, rotation.

    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.
    """

    device = dataset.device
    wavelength = dataset.meta.wavelength
    vBF = dataset
    # Normalized center reference from the first n_center radius-ordered
    # columns only — identical values to slicing the full normalized stack
    # (the elementwise per-pixel divide commutes with column selection), and
    # it works on the streaming provider whose stack may live on the CPU.
    inds_r = vBF.bright_field_inds_ordered_by_radius
    norm_cols = vBF.diffraction_pattern_mean_normalized[
        inds_r[:n_center_indices, 0], inds_r[:n_center_indices, 1]
    ].to(device)
    center_cols = vBF.array[..., :n_center_indices].to(device) / norm_cols[None, None, :]
    bright_field_center_image = center_cols.mean(-1)
    G_ref = torch.fft.fft2(bright_field_center_image)

    sampling = torch.as_tensor(
        dataset.meta.sampling[:2], device=device, dtype=torch.float32
    )
    rot_value = float(dataset.meta.rotation)
    if use_com_rotation_prior == "auto" and abs(rot_value) < 1e-6:
        try:
            intensities = dataset.parent_dataset.array
            # dr is the detector sampling; only the ratio of the two axes matters here, as the
            # COM-curl rotation search is invariant to uniform scaling of the COM field.
            recip = tuple(float(v) for v in dataset.parent_dataset.dr[:2])
            rot_value = estimate_scan_rotation_from_com(
                intensities, reciprocal_sampling=recip
            )
            if verbosity > 0:
                print(f"COM-curl scan-rotation prior: {rot_value:.2f} deg")
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError) as e:
            if verbosity > 0:
                print(f"COM rotation prior unavailable ({e}); using meta.rotation")
    rot_tensor = torch.tensor([rot_value], device=device, dtype=torch.float32)
    global_shifts = torch.zeros(
        (len(vBF.bright_field_inds_centered), 2), device=device, dtype=torch.float32
    )
    order = vBF.bright_field_inds_radial_order.to(device=device)
    desc = (
        "Self-calibration (pairwise)"
        if alignment_method == "pairwise"
        else "Self-calibration (reference-based)"
    )
    pbar = tqdm(bin_factors, desc=desc, disable=not verbosity)
    for bin in pbar:
        if alignment_method == "pairwise":
            measured_incremental_shifts_px = bright_field_shifts_pairwise(
                G_ref,
                vBF,
                vBF.bright_field_inds_centered,
                vBF.bright_field_inds_centered_ordered_by_radius,
                bin,
                n_batches,
                registration_upsample_factor,
                verbosity,
                lowpass_fwhm_bright_field,
            )
        else:
            measured_incremental_shifts_px = bright_field_shifts(
                G_ref,
                vBF,
                vBF.bright_field_inds_centered,
                vBF.bright_field_inds_centered_ordered_by_radius,
                bin,
                n_batches,
                registration_upsample_factor,
                verbosity,
                lowpass_fwhm_bright_field,
            )
        measured_incremental_shifts_px -= torch.mean(
            measured_incremental_shifts_px, axis=0
        )
        inc_shifts_angstroms = measured_incremental_shifts_px * sampling
        pbar.set_postfix({"Res. Shift [A]": f"{inc_shifts_angstroms.max().item():.2f}"})
        global_shifts += inc_shifts_angstroms
        opt_aberrations, opt_rotation = (
            fit_aberrations_and_rotation_to_bright_field_shifts(
                global_shifts,
                dataset.parent_dataset.dr,
                wavelength,
                vBF.bright_field_mask,
                rotation_init=rot_tensor,
                rotation_requires_grad=fit_rotation,
                init_method=init_method,
                verbosity=verbosity,
            )
        )
        fitted_global_shifts = aberrations_to_image_shifts(
            opt_aberrations,
            opt_rotation,
            dataset.parent_dataset.dr,
            wavelength,
            vBF.bright_field_mask.shape,
        )
        fitted_global_shifts = fitted_global_shifts[vBF.bright_field_mask]
        model_incremental_shifts = fitted_global_shifts - global_shifts
        global_shifts = fitted_global_shifts

        # Accumulate the measured shifts into the vBF's alignment OVERLAY
        # (per-pixel phase ramps compose additively in the shift vector), so
        # subsequent get_G_chunk/get_G_columns fetches — including the next
        # bin level's and the reference update below — see the self-calibrated
        # G without ever mutating the (possibly shared/resident) cache.
        vBF.add_alignment_shifts(measured_incremental_shifts_px[order])
        G_ref = vBF.get_G_chunk(0, n_center_indices).mean(dim=-1)

    if verbosity > 0:
        print(
            f"Interpolated shifts for {len(model_incremental_shifts)} bright field indices"
        )
        cc_shifts_x = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        cc_shifts_y = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        cc_shifts_x[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            global_shifts[:, 1]
        )
        cc_shifts_y[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            global_shifts[:, 0]
        )

        fitted_shifts_x = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        fitted_shifts_y = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        fitted_shifts_x[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            fitted_global_shifts[:, 1]
        )
        fitted_shifts_y[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            fitted_global_shifts[:, 0]
        )
        titles = [
            "CC Shift Values X",
            "CC Shift Values Y",
            "Fitted Shifts X",
            "Fitted Shifts Y",
        ]
        plots = [cc_shifts_x, cc_shifts_y, fitted_shifts_x, fitted_shifts_y]
        vis.show_2d(plots, cbar=True, title=titles, cmap="RdBu")

    if update_dataset:
        vBF.meta.aberrations = Aberrations(array=opt_aberrations)
        vBF.meta.rotation = opt_rotation.item()
        vBF.meta.defocus_guess = -opt_aberrations[0].item()

    return opt_aberrations, opt_rotation


def bright_field_shifts(
    G_ref: torch.Tensor,
    vBF,
    bright_field_inds_centered: torch.Tensor,
    bright_field_inds_centered_ordered_by_radius: torch.Tensor,
    bin: int,
    n_batches: int = 16,
    registration_upsample_factor: int = 10,
    verbosity: int = 1,
    fwhm_lowpass_bf=None,
) -> torch.Tensor:
    """
    Calculate the bright field shifts.
    The bright field shifts are calculated by performing a registration between the reference and moving images.
    The registration is performed by performing an upsampled cross-correlation between the reference and moving images.
    The bright field shifts are then returned. If binning is used, the bright field shifts are interpolated to all bright field indices.

    NOTE: the running-reference update below makes the RESULT depend on the
    batch grouping (each batch registers against a reference built from the
    previous batches) — a property of the algorithm, not of how G is stored.
    Reproducibility therefore requires a fixed ``n_batches``.

    Args:
        G_ref: torch.Tensor - reference image (2D: H x W)
        vBF: DatasetVirtualBrightField4dstem (eager or streaming) — G columns
            are fetched through ``get_G_columns`` (alignment overlay applied).
        bright_field_inds_centered: torch.Tensor - bright field indices centered (2D: N x 2)
        bright_field_inds_centered_ordered_by_radius: torch.Tensor - bright field indices centered ordered by radius (2D: N x 2)
        bin: int - bin size. If None, no binning is performed.
        n_batches: int - number of batches. The default is 16.
        registration_upsample_factor: int - registration upsample factor. The default is 10.
        verbosity: int - verbosity. The default is 1.
        fwhm_lowpass_bf: float - FWHM of the lowpass filter. If None, no lowpass filter is used.

    Returns:
        torch.Tensor - bright field shifts (2D: N x 2), order by radius in the bright field, increasing order.
    """
    device = vBF.device
    n_bf_total = vBF.n_bright_field
    scan_ny, scan_nx = int(vBF.array.shape[0]), int(vBF.array.shape[1])
    kx = torch.fft.fftfreq(scan_nx, device=device).reshape(1, -1)
    ky = torch.fft.fftfreq(scan_ny, device=device).reshape(-1, 1)

    if fwhm_lowpass_bf is not None:
        sigma = fwhm_lowpass_bf / (2 * (2 * np.log(2)) ** 0.5)
        kx2 = kx**2  # shape (1, N)
        ky2 = ky**2  # shape (M, 1)
        k2 = kx2 + ky2  # shape (M, N)
        gaussian_filter = torch.exp(-2 * (np.pi**2) * (sigma**2) * k2)

    bright_field_inds_order = torch.argsort(
        torch.sum(bright_field_inds_centered**2, dim=1)
    )
    bright_field_inds_ordered_by_radius_binned = torch.ceil(
        bright_field_inds_centered_ordered_by_radius / bin
    ).int()
    bright_field_inds_binned = torch.ceil(bright_field_inds_centered / bin).int()
    bright_field_unique_inds_binned = torch.unique(bright_field_inds_binned, dim=0)
    bright_field_unique_inds_binned_order = torch.argsort(
        torch.sum(bright_field_unique_inds_binned**2, dim=1)
    )
    bright_field_unique_inds_binned_ordered = bright_field_unique_inds_binned[
        bright_field_unique_inds_binned_order
    ]

    # Calculate batch_size based on n_batches
    total_elements = len(bright_field_unique_inds_binned_ordered)
    batch_size = max(1, total_elements // n_batches)

    if verbosity > 0:
        bf_ref_before = torch.fft.ifft2(G_ref, dim=(0, 1), norm="ortho").real

    bright_field_shifts = torch.zeros(
        len(bright_field_unique_inds_binned_ordered), 2, device=device
    )
    N = len(bright_field_unique_inds_binned_ordered)
    desc = "Registering bright field images"
    pbar = tqdm(range(0, N, batch_size), desc=desc, disable=not verbosity)
    for i in pbar:
        # Determine batch indices and actual batch size upfront
        if bin > 1:
            batch_inds = bright_field_unique_inds_binned_ordered[i : i + batch_size]
        else:
            remaining_elements = len(bright_field_unique_inds_binned_ordered) - i
            actual_batch_size = min(batch_size, remaining_elements)
            batch_inds = np.arange(i, i + actual_batch_size)
            max_indices = n_bf_total - 1
            batch_inds = np.clip(batch_inds, 0, max_indices)

            if batch_inds[-1] >= n_bf_total:
                batch_inds = batch_inds[batch_inds < n_bf_total]
                if len(batch_inds) == 0:
                    print(f"Warning: No valid indices for batch starting at {i}")
                    continue

        # Determine actual batch size and allocate G_moving appropriately
        actual_batch_size = len(batch_inds)
        # print(f"{i:02d} actual_batch_size = {actual_batch_size}")

        if bin > 1:
            # Allocate G_moving with the proper size for binning
            G_moving_batch = torch.zeros(
                actual_batch_size,
                scan_ny,
                scan_nx,
                device=device,
                dtype=torch.complex64,
            )
            # perform the binning in a loop (columns fetched through the
            # provider: memory-resident gathers, overlay applied)
            for j, ri in enumerate(batch_inds):
                take = (
                    bright_field_inds_ordered_by_radius_binned[:, 0] == ri[0]
                ).__and__(bright_field_inds_ordered_by_radius_binned[:, 1] == ri[1])
                G_moving_batch[j] = torch.mean(
                    vBF.get_G_columns(torch.where(take)[0]), dim=-1
                )

            if fwhm_lowpass_bf is not None:
                G_moving_batch = G_moving_batch * gaussian_filter
        else:
            # print(f"Taking indices: {bright_field_inds_centered_ordered_by_radius[batch_inds]}")
            G_moving_batch = torch.permute(vBF.get_G_columns(batch_inds), (2, 0, 1))

        xy_shift = relative_shifts(
            G_ref, G_moving_batch, upsample_factor=registration_upsample_factor
        )
        # Slice xy_shift to match the actual batch size we need
        xy_shift = xy_shift[:actual_batch_size]
        bright_field_shifts[i : i + actual_batch_size] = xy_shift

        dx = xy_shift[:, 1]
        dy = xy_shift[:, 0]
        phase_ramp = torch.exp(
            -1j
            * 2
            * np.pi
            * (dx[:, None, None] * kx[None, :, :] + dy[:, None, None] * ky[None, :, :])
        )
        G_ref = G_ref * i / (i + batch_size) + torch.mean(
            G_moving_batch * phase_ramp, dim=0
        ) * batch_size / (i + batch_size)

    if verbosity > 0:
        bf_ref_after = torch.fft.ifft2(G_ref, dim=(0, 1), norm="ortho").real
        fig, ax = vis.show_2d(
            [bf_ref_before, bf_ref_after],
            cbar=True,
            title=[
                "Bright Field reference image [Before]",
                "Bright Field reference image [After]",
            ],
        )

    if bin > 1:
        # if verbosity > 0:
        #     print(
        #         "Interpolating shifts from binned indices to all bright field indices..."
        #     )

        # Convert to numpy for scipy interpolation
        binned_points = bright_field_unique_inds_binned_ordered.cpu().numpy() * bin
        shift_values = bright_field_shifts.cpu().numpy()
        query_points = bright_field_inds_centered_ordered_by_radius.cpu().numpy()

        all_bright_field_shifts_np = griddata(
            binned_points, shift_values, query_points, method="cubic", fill_value=0.0
        )
        zero_shifts = torch.from_numpy(all_bright_field_shifts_np).sum(1) == 0

        inds_with_zero_shifts = torch.where(zero_shifts)[0]
        # Only re-register the leftover zero-shift indices if there ARE any --
        # an empty index set makes G_zero_shifts a (0, H, W) tensor, and cuFFT
        # rejects a zero-length batch with CUFFT_INVALID_SIZE.
        if inds_with_zero_shifts.numel() > 0:
            G_zero_shifts = vBF.get_G_columns(inds_with_zero_shifts)
            G_zero_shifts = torch.permute(G_zero_shifts, (2, 0, 1))
            xy_shift = relative_shifts(
                G_ref, G_zero_shifts, upsample_factor=registration_upsample_factor
            )
            all_bright_field_shifts_np[inds_with_zero_shifts] = xy_shift.cpu().numpy()
        # if verbosity > 0:
        #     print(f"Indices with zero shifts after interpolation= {zero_shifts.sum()}")
    else:
        all_bright_field_shifts_np = bright_field_shifts.cpu().numpy()
    # print(f"all_bright_field_shifts_np.shape = {all_bright_field_shifts_np.shape}")

    inverse_order = torch.argsort(bright_field_inds_order).cpu().numpy()
    all_bright_field_shifts_np = all_bright_field_shifts_np[inverse_order]
    # print(f"inverse_order.shape = {inverse_order.shape}")
    # print(f"all_bright_field_shifts_np.shape = {all_bright_field_shifts_np.shape}")
    result = all_bright_field_shifts_np.copy().astype(np.float32)
    # result[:,0] = all_bright_field_shifts_np[:,1]
    # result[:,1] = all_bright_field_shifts_np[:,0]
    return torch.as_tensor(result, device=device)


def bright_field_shifts_pairwise(
    G_ref: torch.Tensor,
    vBF,
    bright_field_inds_centered: torch.Tensor,
    bright_field_inds_centered_ordered_by_radius: torch.Tensor,
    bin: int,
    n_batches: int = 16,
    registration_upsample_factor: int = 10,
    verbosity: int = 1,
    fwhm_lowpass_bf=None,
    connectivity: int = 8,
) -> torch.Tensor:
    """Pairwise graph-synchronised bright-field shift estimation.

    Drop-in replacement for :func:`bright_field_shifts` with the identical return
    contract: (N, 2) shifts in the original ``bright_field_inds_centered`` order
    (i.e. **not** ordered by radius).

    Rather than aligning every BF image against a single global reference, each
    neighbouring pair of BF images is cross-correlated independently.  The
    overdetermined system of pairwise differences is solved in the least-squares
    sense (graph-Laplacian synchronisation).  This removes the single-reference
    bias that causes large errors when the reference itself is shifted by a large
    aberration.

    Every step is chunk-size-invariant (per-bin means are a partition of
    the BF axis; per-pair correlations are independent; the synchronisation is
    one small global least-squares), so the streamed result is identical to
    the eager one for any provider residency.

    Args:
        G_ref: (H, W) reference image in Fourier domain (unused here, kept for
            drop-in compatibility with the reference-based signature).
        vBF: DatasetVirtualBrightField4dstem (eager or streaming) — G columns
            are fetched through ``get_G_columns`` (alignment overlay applied);
            the BF axis is in radial order.
        bright_field_inds_centered: (N, 2) BF detector coords in original order.
        bright_field_inds_centered_ordered_by_radius: (N, 2) BF coords in radial
            order (matches the provider's BF axis).
        bin: binning factor.  If > 1, images are averaged within each bin before
            pairwise CC (same binning strategy as ``bright_field_shifts``).
        n_batches: not used (kept for API compatibility).
        registration_upsample_factor: subpixel upsample factor.
        verbosity: not used (kept for API compatibility).
        fwhm_lowpass_bf: FWHM of optional Gaussian low-pass filter in Fourier
            domain (same as ``bright_field_shifts``).
        connectivity: 4 or 8 for the neighbour graph.

    Returns:
        (N, 2) shifts in the original ``bright_field_inds_centered`` order.
    """
    device = vBF.device
    n_bf_total = vBF.n_bright_field
    scan_ny, scan_nx = int(vBF.array.shape[0]), int(vBF.array.shape[1])

    if fwhm_lowpass_bf is not None:
        kx = torch.fft.fftfreq(scan_nx, device=device).reshape(1, -1)
        ky = torch.fft.fftfreq(scan_ny, device=device).reshape(-1, 1)
        sigma = fwhm_lowpass_bf / (2 * (2 * np.log(2)) ** 0.5)
        gaussian_filter = torch.exp(-2 * (np.pi**2) * (sigma**2) * (kx**2 + ky**2))

    if bin > 1:
        # Average images within each spatial bin (same binning as bright_field_shifts)
        bright_field_inds_ordered_by_radius_binned = torch.ceil(
            bright_field_inds_centered_ordered_by_radius / bin
        ).int()
        unique_bins = torch.unique(bright_field_inds_ordered_by_radius_binned, dim=0)
        unique_bins_order = torch.argsort(torch.sum(unique_bins**2, dim=1))
        unique_bins_ordered = unique_bins[unique_bins_order]

        G_binned_list = []
        coords_binned_list = []
        for ri in unique_bins_ordered:
            take = (bright_field_inds_ordered_by_radius_binned[:, 0] == ri[0]) & (
                bright_field_inds_ordered_by_radius_binned[:, 1] == ri[1]
            )
            g = torch.mean(vBF.get_G_columns(torch.where(take)[0]), dim=-1)  # (H, W)
            if fwhm_lowpass_bf is not None:
                g = g * gaussian_filter
            G_binned_list.append(g)
            coords_binned_list.append(ri.float())

        # Stack: (N_bins, H, W) for pairwise CC
        G_pairwise = torch.stack(G_binned_list, dim=0)
        coords_pairwise = torch.stack(coords_binned_list, dim=0)  # (N_bins, 2)
    else:
        # No binning: use all images in radial order. NOTE: this
        # materializes the FULL G stack (~17 GiB at full scan) — prefer
        # bin_factors ending >= 2 on large data.
        G_pairwise = vBF.get_G_columns(
            torch.arange(n_bf_total, device=device)
        ).permute(2, 0, 1)  # (N, H, W)
        if fwhm_lowpass_bf is not None:
            # Match the reference path, which low-passes every image before CC
            # (the bin>1 branch above already applies this per averaged bin).
            G_pairwise = G_pairwise * gaussian_filter
        coords_pairwise = bright_field_inds_centered_ordered_by_radius  # (N, 2)

    # Build neighbour graph and run pairwise cross-correlations. NB: this loops over
    # M ~ 4*N pairs (one relative_shifts call each); fine for typical BF disks, but for
    # very large unbinned disks consider increasing the bin factor for speed.
    pairs = make_neighbor_pairs(coords_pairwise.int(), connectivity=connectivity)
    if pairs.shape[0] == 0:
        # Fallback: return zeros if graph has no edges
        return torch.zeros(len(bright_field_inds_centered), 2, device=device)

    deltas = pairwise_relative_shifts(
        G_pairwise, pairs, upsample_factor=registration_upsample_factor
    )
    shifts_pairwise = synchronize_shifts(
        len(G_pairwise), pairs, deltas
    )  # (N_bins or N, 2)
    shifts_pairwise = shifts_pairwise - shifts_pairwise.mean(0)

    if bin > 1:
        # Interpolate from binned coords back to all BF indices (radial order)
        binned_points = (unique_bins_ordered.cpu().numpy() * bin).astype(np.float32)
        shift_values = shifts_pairwise.cpu().numpy()
        query_points = bright_field_inds_centered_ordered_by_radius.cpu().numpy()

        all_shifts_np = griddata(
            binned_points, shift_values, query_points, method="cubic", fill_value=0.0
        )

        # Fill any remaining zeros via direct CC against mean
        zero_mask = torch.from_numpy(all_shifts_np).sum(1) == 0
        zero_inds = torch.where(zero_mask)[0]
        if len(zero_inds) > 0:
            G_zeros = vBF.get_G_columns(zero_inds).permute(2, 0, 1)
            # streamed mean over all BF pixels (chunked; overlay applied)
            _acc = None
            for _s, _e in _chunk_ranges(n_bf_total, 16):
                _part = vBF.get_G_chunk(_s, _e).sum(dim=-1)
                _acc = _part if _acc is None else _acc + _part
            G_ref_mean = _acc / n_bf_total  # (H, W)
            xy_shift = relative_shifts(
                G_ref_mean, G_zeros, upsample_factor=registration_upsample_factor
            )
            all_shifts_np[zero_inds.numpy()] = xy_shift.cpu().numpy()
    else:
        all_shifts_np = shifts_pairwise.cpu().numpy()

    # Convert from radial order back to original bright_field_inds_centered order
    # bright_field_inds_order = argsort(sum(centered**2)) maps original→radius position
    # inverse of that permutation maps radius→original
    bright_field_inds_order = torch.argsort(
        torch.sum(bright_field_inds_centered**2, dim=1)
    )
    inverse_order = torch.argsort(bright_field_inds_order).cpu().numpy()
    result = all_shifts_np[inverse_order].copy().astype(np.float32)
    return torch.as_tensor(result, device=device)




def _resolve_vbf(dataset, bright_field_mask_threshold):
    """Datasets build/cache their own provider flavor (Dataset4dstem.get_vbf);
    anything else is already a vBF/provider and passes through."""
    if isinstance(dataset, Dataset4dstem):
        return dataset.get_vbf(bright_field_mask_threshold)
    return dataset


def _resolve_upsample_int(source, upsample, verbosity=0):
    """Resolve ``upsample`` ("nyquist" or a number) into the integer per-axis
    factors. ``source`` is any object exposing ``sampling`` and ``meta``
    (a Dataset4dstem or a vBF). Shared by the reconstruction, the empirical
    SSNR and the depth section — they MUST agree on the grid."""
    if upsample == "nyquist":
        scan_sampling = np.array(source.sampling[:2])
        nyquist_sampling = source.meta.wavelength / (
            4 * source.meta.semiconvergence_angle
        )
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if verbosity > 0:
            print(f"scan_sampling = {scan_sampling}")
            print(f"nyquist_sampling = {nyquist_sampling}")
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
        return upsample_int
    if isinstance(upsample, str):
        raise ValueError(f"Invalid upsample: {upsample}")
    if isinstance(upsample, (int, float)):
        upsample_float = np.array([upsample, upsample])
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
        return upsample_int
    raise ValueError(f"Invalid upsample: {upsample}")


def _chunk_ranges(n_total: int, n_batches):
    """Yield ``(s, e)`` BF-pixel ranges: ceil-division batches with a clamped
    tail. Never yields an empty range (CUFFT_INVALID_SIZE guard) even when
    ``n_batches`` exceeds the number of non-empty batches — reachable on the
    empirical-SSNR path where half-pixel subsets are reconstructed."""
    n_b = int(n_batches) if n_batches is not None else 1
    size = max(1, int(math.ceil(n_total / max(1, n_b))))
    for s in range(0, n_total, size):
        yield s, min(s + size, n_total)


def _upsampled_grid(vBF, upsample_int):
    """``(new_shape, Qy, Qx)`` of the reconstruction grid — derived from the
    vBF STACK shape (the streaming provider may keep no resident G)."""
    new_shape = (
        int(round(vBF.array.shape[0] * upsample_int[0])),
        int(round(vBF.array.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)
    return new_shape, Qy, Qx


def _g_chunk_kouter_view(vBF, s: int, e: int):
    """``G[..., s:e]`` as a ``[Nk, Nqy, Nqx]`` view -- **or None** when that view
    is not already contiguous, i.e. when reaching it would cost a real copy.

    ``vBF.G`` is ``torch.fft.fft2(array, dim=(0, 1))``, and torch/cuFFT works in
    the ``[B, ny, nx]`` layout and hands back a permuted VIEW: the resident G has
    strides ``(nx, 1, ny*nx)`` and ``is_contiguous() == False``, so it is ALREADY
    bright-field-outermost in memory.  ``get_G_chunk``'s ``.contiguous()``
    therefore does not "compact a non-contiguous slice" -- it TRANSPOSES G into
    the ``[Nqy, Nqx, Nk]`` order, which is the order the ``ifft2`` that follows
    then has to transpose back out of (measured: the copy 0.150 ms and the
    transform 1.66x slower, at the registered 245^2 x 90 chunk).  Asking for the
    k-outer view instead costs **nothing at all** -- the permuted slice is
    already dense, so ``.contiguous()`` returns it unchanged.

    Returned READ-ONLY, deliberately, which is what makes it free.  The caller
    ``get_G_chunk`` serves needs an OWNED chunk because
    ``correct_aberrations_inplace`` mutates it; ``correct_aberrations_kouter`` is
    always out of place and never touches its input, so no ownership copy is
    required on this path.  Do not hand this tensor to anything that mutates.

    None is returned -- and the caller falls back to the shipped
    ``[Nqy, Nqx, Nk]`` body -- whenever the layout is not free:

    * an alignment overlay is active (``_align_shifts``); the overlay's ramp is
      built in the ``[Nqy, Nqx, Nk]`` layout and only runs during a
      bright-field-shift fit, so it is not worth a second spelling here;
    * the provider keeps no resident ``G`` (``DatasetVirtualBrightFieldStreaming``
      at ``g_residency`` "vbf_gpu"/"cpu_pinned" raises on ``.G``).  Those two
      residencies *also* produce dense-but-k-outer chunks (an ``fft2`` output and
      a ``movedim(0, -1)`` of a ``[Nk, ny, nx]`` host buffer respectively), so the
      same win is available there through a provider-side fetch -- unmeasured, and
      out of scope here;
    * the permuted view is not contiguous for any other reason, in which case
      reaching the k-outer layout would cost a transposing copy (0.525 ms
      measured) that is larger than the transform saving it buys (0.349 ms) --
      D49's own refutation, and this guard is what keeps it from applying.
    """
    if getattr(vBF, "_align_shifts", None) is not None:
        return None
    try:
        G = vBF.G
    except (RuntimeError, AttributeError):
        return None
    view = G[..., s:e].permute(2, 0, 1)
    return view if view.is_contiguous() else None


def _direct_ptychography_kouter(
    Gprime_k,
    aberrations,
    ds_rotation,
    semiconvergence_angle,
    wavelength,
    Qy,
    Qx,
    Kx,
    Ky,
    upsample,
    reduce,
    A,
):
    """:func:`_direct_ptychography` for a ``[Nk, Nqy, Nqx]`` chunk.

    Identical arithmetic, and every stage of it is BIT-EXACT against the
    ``[Nqy, Nqx, Nk]`` form at the registered chunk shapes (``torch.equal`` on all
    three: the correction is a per-element multiply with permuted subscripts; the
    transform is the same cuFFT batched plan reached from the layout it wants; and
    the chunk reduction does not reassociate because ``ifft2(dim=(0, 1))`` returns
    a *permuted view* whose outer physical axis is already ``Nk``, so
    ``sum(dim=-1)`` on it and ``sum(dim=0)`` here walk memory in the same order --
    0 of 60025 words differ, ``_probe_d225_copydelta.py``).

    ``Gprime_k`` is only ever READ (the k-outer correction op is out of place), so
    it may be a view of the caller's G.
    """
    factor = np.sqrt(upsample[0] * upsample[1])
    corrected = torch.empty(
        Gprime_k.shape, dtype=Gprime_k.dtype, device=Gprime_k.device
    )
    correct_aberrations_kouter(
        Gprime_k,
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
        A if A is not None else torch.empty(
            (0, 2), dtype=torch.float32, device=Gprime_k.device
        ),
        corrected,
    )
    if reduce == "sum":
        # D24's argument about ``factor``, applied to the ONE remaining full-size
        # constant multiply in this body -- cuFFT's C2C plan is unnormalised, so
        # ``norm="ortho"`` is not free inside the transform: torch applies the
        # 1/sqrt(Nqy*Nqx) as its OWN elementwise kernel over the [Nk, H, W] complex
        # output.  Measured on the registered medium chunk (90, 245, 245): 1.015 ms
        # for 8 chunks = 15.0 % of the row's whole device time, for a scalar the
        # linear reduction that follows can carry.  ``norm="forward"`` puts the
        # normalisation on the FORWARD transform, so the inverse hands back the raw
        # cuFFT output and the pass disappears; the constant then rides along on the
        # multiply the reduced [H, W] image was already paying, so this costs no
        # extra dispatch at any shape (1.28x / 1.24x on the medium / small chunk).
        #
        # NOT bit-exact, and deliberately so: ``torch.equal`` says
        # ``ifft2(norm="forward") * 1/sqrt(Nqy*Nqx)`` IS ``ifft2(norm="ortho")``
        # elementwise, so the only difference is scale-then-sum vs sum-then-scale
        # over the chunk axis -- the same float32 reassociation this path's oracle
        # already grades against an INDEPENDENT complex128 reference (measured
        # 1.9e-07 peak-normalised against a 1e-5 gate), and strictly FEWER roundings.
        # That is also why the ``_is_exact_power_of_two`` split is gone from this
        # branch: the folded constant is 1/245 at the registered grid, never a power
        # of two, so there is no exponent-shift case left to preserve.
        G_bf = torch.fft.ifft2(corrected, dim=(-2, -1), norm="forward")
        scale = factor / math.sqrt(corrected.shape[-2] * corrected.shape[-1])
        return torch.sum(G_bf.imag, dim=0) * scale
    G_bf = torch.fft.ifft2(corrected, dim=(-2, -1), norm="ortho")
    if reduce == "none":
        # The yielded SHAPE contract is [Nqy, Nqx, Nk] and the empirical-SSNR
        # consumer's output is digested bit-exact, so hand back a permuted VIEW
        # (free) rather than change the contract.  Its consumer indexes the chunk
        # axis (``imag[..., pm]``), which copies to a dense [Nqy, Nqx, m] tensor of
        # the same values either way, so the half-sums are unchanged.
        return (G_bf * factor).imag.permute(1, 2, 0)
    raise ValueError(f"invalid reduce: {reduce!r}")


def _iter_chunk_images(vBF, aberrations, rotation_t, Qy, Qx, upsample_int, n_batches, reduce):
    """Yield ``(s, e, image)`` per BF-pixel chunk: fetch (mutation-safe, from
    vBF.get_G_chunk) → tile when upsampling → aberration-correct → ifft2 →
    ``reduce`` over the chunk axis. ``rotation_t`` should be a pre-built
    device tensor so the hot loop launches no per-chunk H2D copies."""
    device = vBF.device
    semiconvergence_angle = vBF.meta.semiconvergence_angle
    wavelength = vBF.meta.wavelength
    # ``vBF.k`` is (Nk, 2), so ``vBF.k[s:e, 1]`` is a COLUMN slice -- stride 2,
    # not dense.  A 1-D ``wp.array`` needs a dense inner stride, so every
    # aberration op ``.contiguous()``-copies BOTH coordinate operands on EVERY
    # call: at the registered medium geometry that is 16 tiny device copies and
    # ~25 us of host time per pass of this loop, and because each copy is a FRESH
    # tensor it also forces a rebind of those two slots in the ops' recorded-
    # command memo.  The ``_contiguous_memo`` those ops call cannot help: it keys
    # on tensor IDENTITY and ``vBF.k[s:e, 1]`` is a new Python object every call,
    # so it misses every time.  Compact the two columns ONCE here instead -- a
    # slice of a contiguous 1-D tensor is itself dense, so the ops' own
    # ``.contiguous()`` then returns it unchanged.  BIT-EXACT: ``.contiguous()``
    # relocates bits, it does not compute, so the kernels read the same words.
    Kx_all = vBF.k[:, 1].contiguous()
    Ky_all = vBF.k[:, 0].contiguous()
    # The aberration kernel's ``A = aperture(K) * exp(-i chi(K))`` factor varies
    # only with the bright-field pixel, so it is constant across every chunk.
    # Built ONCE here for all bright-field pixels and sliced per chunk exactly as
    # ``vBF.k[s:e]`` is; per chunk the build's ~30 us of host time would cost more
    # than the device work it saves.
    A_all = build_aberration_bf_factor(
        aberrations,
        semiconvergence_angle,
        wavelength,
        Kx_all,
        Ky_all,
    )
    # The chunk-sum-first collapse fuses the correction and the ik reduction and
    # is spelled for the [Nqy, Nqx, Nk] layout only, so the k-outer route is not
    # taken when it is enabled -- that variant keeps its shipped body verbatim.
    use_kouter = not (reduce == "sum" and _CHUNK_SUM_FIRST)
    for s, e in _chunk_ranges(vBF.n_bright_field, n_batches):
        # G is stored bright-field-OUTERMOST (an fft2 output is a permuted view),
        # so ``get_G_chunk``'s ownership copy is a transposition into the layout
        # the ifft2 then has to transpose back out of.  When the k-outer view is
        # free -- resident G, no alignment overlay -- take it: no copy at all, and
        # the transform runs as the natural batched form (1.66x, bit-exact).
        Gc_k = _g_chunk_kouter_view(vBF, s, e) if use_kouter else None
        if Gc_k is not None:
            if (upsample_int > 1).any():
                Gc_k = torch.tile(Gc_k, (1, int(upsample_int[0]), int(upsample_int[1])))
            yield s, e, _direct_ptychography_kouter(
                Gc_k,
                aberrations,
                rotation_t,
                semiconvergence_angle,
                wavelength,
                Qy,
                Qx,
                Kx_all[s:e],
                Ky_all[s:e],
                upsample_int,
                reduce,
                A_all[s:e],
            )
            continue
        # ``get_G_chunk``'s copy exists for OWNERSHIP -- the default body's
        # ``correct_aberrations_inplace`` mutates the chunk.  The chunk-sum-first
        # collapse does NOT: its only consumer is ``correct_aberrations_ksum``,
        # which is out of place and never touches its input, exactly as
        # ``correct_aberrations_kouter`` is on the branch above.  So take the same
        # read-only view that branch takes and pay no copy: G is bright-field-
        # OUTERMOST in memory, so the copy this skips is a TRANSPOSE, 1.116 ms of
        # a 3.40 ms row at the registered medium geometry.  ``_g_chunk_kouter_view``
        # already refuses (returns None) for an alignment overlay, a provider with
        # no resident G, or a slice whose permuted form is not dense -- the three
        # cases where the view is not free -- and this falls back to the owned
        # chunk there.  Permuting it back to [Nqy, Nqx, Nk] is a relabelling of
        # the same storage, so what reaches the kernel is ``G[..., s:e]`` itself.
        Gc_view = None if use_kouter else _g_chunk_kouter_view(vBF, s, e)
        Gc = vBF.get_G_chunk(s, e) if Gc_view is None else Gc_view.permute(1, 2, 0)
        if (upsample_int > 1).any():
            Gc = torch.tile(Gc, (int(upsample_int[0]), int(upsample_int[1]), 1))
        yield s, e, _direct_ptychography(
            Gc,
            aberrations,
            rotation_t,
            semiconvergence_angle,
            wavelength,
            Qy,
            Qx,
            Kx_all[s:e],
            Ky_all[s:e],
            device,
            upsample_int,
            reduce=reduce,
            A=A_all[s:e],
        )


def phase_contrast_transfer_function(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    upsample: Union[float, str] = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Performs a joint ptychography reconstruction and aberration determination.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        upsample: Upsampling factor for the diffraction pattern.
        bright_field_mask_threshold: Threshold for the bright field.
        verbosity: Verbosity level.
        n_batches: Number of batches for the vBF.

    Returns:
        torch.Tensor: The reconstructed phase image (Imaginary part).
        DatasetVirtualBrightField4dstem: the vBF dataset.

    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.
        ValueError: If the upsample is invalid.
    """
    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)
    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)
    ds_rotation = torch.tensor(vBF.meta.rotation, device=vBF.device)
    semiconvergence_angle = vBF.meta.semiconvergence_angle
    wavelength = vBF.meta.wavelength

    # The PCTF kernel never reads G's VALUES -- gamma depends only on Q, K, the
    # aberrations, the aperture and the wavelength -- so the U^2-tiled copy of G
    # this used to build (``torch.tile(vBF.G, (Uy, Ux, 1))``) was a pure
    # allocate-and-copy of up to 7.7 GiB per call at the Fig1 config.  Only the
    # SHAPE was ever used, and that shape is (len(Qy), len(Qx), len(Kx)).
    # NOTE this is the PCTF only: the reconstruction's own q-tiling of G (see
    # ``_direct_ptychography_*``) is load-bearing -- it is how single-sideband
    # ptychography reaches past the scan Nyquist -- and must not be removed.
    new_shape = (
        int(round(vBF.G.shape[0] * upsample_int[0])),
        int(round(vBF.G.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)
    Kx = vBF.k[:, 1]
    Ky = vBF.k[:, 0]
    aberrations = vBF.meta.aberrations.array
    pctf = _phase_contrast_transfer_function(
        None,
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
    )
    return pctf


# Retained-activation budget for the TV-autofocus chunk loop in
# :func:`determine_aberrations` (see ``use_ckpt`` there).  Gradient-checkpointing
# each chunk buys peak memory by paying a SECOND forward pass inside the
# backward, so it is only worth doing when the memory it saves actually matters.
# What it saves is exactly ``scatterem::correct_aberrations_fwd``'s saved input:
# ``n_k_tv * new_shape[0] * new_shape[1] * 8`` bytes summed over the chunks, i.e.
# one upsampled copy of ``vBF.G``.  Below this budget the checkpoint is skipped
# and the fit runs one forward per objective evaluation instead of two.
#
# The switch is numerically INERT -- the recompute is a deterministic
# re-execution of the same ops on the same inputs, so the aberration gradient is
# identical either way (gated in ``_probe_d4_ckpt.py`` against the adjoint's own
# ``wp.tile_atomic_add`` reproducibility floor, R9d).  Only peak memory and
# runtime move, which is why a budget is a safe thing to condition on.
#
# 4 GiB keeps the case the loop's docstring was written for on the checkpointed
# path: a 1024^2 scan needs ~9 GB here (17 GB with the ifft2 output the sum-first
# collapse has since deleted), and reintroducing that is the OOM the chunking
# exists to prevent.  The published ``Fig1_Gd2O3`` fit config needs 2150 MiB.
_TV_CHUNK_CKPT_BUDGET_BYTES = 4 * 1024**3

# Budget for the TV-autofocus chunk CACHE (see ``_g_chunks`` in
# :func:`determine_aberrations`).  ``vBF.G`` is fixed for the whole fit, but
# ``G[..., s:e]`` is a non-contiguous last-axis slice, so ``get_G_chunk`` compacts
# the same bytes on every chunk body -- ~1250-1700 times per fit at
# ``n_batches=25``.  Fetching each distinct chunk once instead costs one extra
# copy of G, ``n_k_tv * ny * nx * 8`` bytes, which is the SAME quantity the
# checkpoint budget above is written in; hence the same default, so the two
# decisions cannot disagree about what "a copy of G is affordable" means.
#
# Like the checkpoint switch this is numerically INERT: a cached chunk is the
# tensor ``get_G_chunk`` returns, byte for byte (verified by ``torch.equal`` in
# ``_probe_d7_fetch.py``), and its one consumer in the loop,
# ``CorrectAberrations.apply``, is out-of-place (measured in D3-run).  So only
# peak memory and runtime move, which is what makes a budget safe to condition on.
_TV_CHUNK_CACHE_BUDGET_BYTES = 4 * 1024**3


def determine_aberrations(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    correction_method: str = "combined",
    fit_rotation: bool = True,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: Optional[float] = None,
    bin_factors: tuple[int, ...] = (2, 1, 1),
    upsample: Union[int, str] = "nyquist",
    n_batches: int = 25,
    roi_shape: tuple[int, int] = (128, 128),
    roi_center: Union[str, tuple[int, int]] = "center",
    num_iterations: int = 10,
    lr: float = 20,
    bright_field_mask_threshold: float = 0.5,
    target_percentage_nonzero_pixels: float = 0.75,
    correct_order: int = 1,
    gradient_mask: torch.Tensor | None = None,
    verbosity: int = 0,
    update_dataset: bool = True,
    n_center_indices: int = 25,
    phase_sign: PhaseSignMode = "positive",
    sharpness_metric: str = "sparsity",
    reg_weight: float = 0.05,
    alignment_method: str = "reference",
    **kwargs,
) -> tuple[torch.Tensor, DatasetVirtualBrightField4dstem]:
    """
    Performs a joint ptychography reconstruction and aberration determination.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        correction_method: How to determine the aberrations. "combined" (default) fits low-order +
            rotation from bright-field shifts, then runs the sharpness autofocus to extend to higher
            orders when correct_order>=2. "bright-field-shifts" does only the fast low-order +
            rotation fit. "autofocus" does only the autofocus (rotation fixed). The legacy
            "total-variation" is a deprecated alias for "autofocus".
        sharpness_metric: Autofocus objective for the 'autofocus'/'combined' higher-order stage:
            "sparsity" (default, L4/kurtosis — less gameable, unbiased for higher orders) or "tv".
        reg_weight: L2 regularization (in scaled units) toward the seed in the autofocus optimizer.
        fit_rotation: Whether to fit the rotation.
        registration_upsample_factor: Upsampling factor for the registration.
        lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field.
        bin: Bin size for the bright field. Used for bright field shifts.
        upsample: Upsampling factor for the diffraction pattern.
        n_batches: Number of batches for the bright field shifts.
        roi_shape: Shape of the region of interest.
        roi_center: Center of the region of interest.
        num_iterations: Number of optimization iterations.
        lr: Learning rate for the optimizer.
        bright_field_mask_threshold: Threshold for the bright field. Used for bright field shifts.
        target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field.
        correct_order: Order of the aberrations to correct. Used for the autofocus stage.
        gradient_mask: Mask for the gradient. Used for the autofocus stage.
        verbosity: Verbosity level.
        n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
    Returns:
        torch.Tensor: The reconstructed weak phase image.
        DatasetVirtualBrightField4dstem: the input dataset with the aberrations determined.

    Raises:
        ValueError: If the roi_center is invalid.
        ValueError: If the dataset is not a valid Dataset4dstem or DatasetVirtualBrightField4dstem object.
        ValueError: If the upsample is invalid.
    """
    if roi_center == "center":
        roi_center = (dataset.shape[0] // 2, dataset.shape[1] // 2)
    elif roi_center == "dark_field_center_of_mass":
        roi_center = (dataset.shape[0] // 2, dataset.shape[1] // 2)
    elif isinstance(roi_center, tuple):
        roi_center = roi_center
    else:
        raise ValueError(f"Invalid roi_center: {roi_center}")

    do_bf, do_tv = _resolve_aberration_stages(correction_method, correct_order)
    if do_tv and not do_bf and fit_rotation:
        # autofocus-only mode fixes rotation; rotation must come from a BF / combined run
        raise ValueError(
            "fit_rotation=True is not supported with the autofocus-only method "
            "(correction_method='autofocus'); it fixes rotation. Use 'bright-field-shifts' "
            "or 'combined' to estimate rotation."
        )

    ds = dataset

    # Ensure upsample is an integer for torch.tile
    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)

    ds_rotation = ds.meta.rotation
    semiconvergence_angle = ds.meta.semiconvergence_angle
    wavelength = ds.meta.wavelength
    dataset = ds
    device = dataset.device
    wavelength = dataset.meta.wavelength

    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)
    # Shapes from the vBF stack, not vBF.G — the streaming provider may keep
    # no resident G.
    new_shape = (
        int(round(vBF.array.shape[0] * upsample_int[0])),
        int(round(vBF.array.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)
    Kx = vBF.k[:, 1]
    Ky = vBF.k[:, 0]

    if do_bf:
        with torch.no_grad():
            if verbosity > 0:
                print("Fitting aberrations and rotation from bright field shifts")
                import time as _time

                _t0 = _time.perf_counter()

            aberrations, rotation = aberrations_and_rotation_from_bright_field_shifts(
                vBF,
                fit_rotation=fit_rotation,
                target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
                n_batches=n_batches,
                registration_upsample_factor=registration_upsample_factor,
                lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
                bin_factors=bin_factors,
                verbosity=verbosity,
                update_dataset=update_dataset,
                n_center_indices=n_center_indices,
                alignment_method=alignment_method,
            )

            if verbosity > 0:
                print(
                    f"Time to fit aberrations and rotation: {_time.perf_counter() - _t0:.3f} s"
                )

    if do_tv:
        # In 'combined', the BF fit above already determined rotation + a low-order seed; expose
        # the fitted rotation to the (rotation-fixed) TV stage and seed the autofocus from it.
        if do_bf:
            vBF.meta.rotation = float(rotation)

        vBF.array.requires_grad = False
        # Non-resident streaming residencies (vbf_gpu / cpu_pinned) fetch G
        # chunks lazily INSIDE the checkpointed closure below; materializing
        # them here would defeat the memory budget (and a generator feeding
        # the LBFGS closure would retain every chunk through backward).
        # All vBF flavors serve chunks through get_G_chunk (fetched lazily
        # inside the checkpointed closure below). This replaces the old
        # resident-Gprime path, whose only difference was pre-tiling the FULL
        # G at upsample>1 — a U^2-sized tensor held through the whole LBFGS
        # run for no numerical benefit (per-chunk tiling is identical).
        n_k_tv = vBF.n_bright_field

        if do_bf:
            # seed the autofocus from the bright-field-shift fit just run above
            seed_aberr = aberrations.detach().clone().to(vBF.device)
        else:
            # pure total-variation: seed from meta, auto-seeding from a one-off BF fit if empty
            seed_aberr = vBF.meta.aberrations.array.clone().to(vBF.device)
            if float(seed_aberr.abs().max()) < 1e-12:
                try:
                    aberrations_and_rotation_from_bright_field_shifts(
                        vBF,
                        fit_rotation=False,
                        update_dataset=True,
                        verbosity=verbosity,
                    )
                    seed_aberr = vBF.meta.aberrations.array.clone().to(vBF.device)
                    if verbosity > 0:
                        print(
                            "TV seed from bright-field shifts: "
                            f"{seed_aberr[:4].detach().cpu().numpy()}"
                        )
                except Exception as e:  # noqa: BLE001
                    if verbosity > 0:
                        print(f"BF-shift seeding failed ({e}); starting TV from zeros")

        scales = _aberration_precondition_scales(
            wavelength, semiconvergence_angle, device=vBF.device
        )
        free_mask = _build_gradient_mask(correct_order, gradient_mask).to(vBF.device)
        # The analytic aberration adjoint's cost is LINEAR in the number of
        # coefficients it differentiates (measured on the published Fig1 chunk:
        # 1.098 ms at 1, 3.530 ms at 12), and ``aberr_of`` below multiplies the
        # frozen ones by ``free`` -- i.e. by exactly zero -- before they ever
        # reach ``u.grad``.  So everything past the last free coefficient is
        # dead work, and at ``correct_order=1`` that is nine gradients in
        # twelve.  Computed once here, off the hot path.  A leading count, not
        # a mask: ``_build_gradient_mask`` freezes a suffix, and a leading
        # count is still correct (only less aggressive) under a user mask with
        # holes in it.
        _free_idx = torch.nonzero(free_mask.reshape(-1), as_tuple=False)
        n_grad_coeffs = int(_free_idx[-1].item()) + 1 if _free_idx.numel() else 0

        # (#5) the ROI is windowed (Hann) inside evaluate_tv via _windowed_roi, which also
        # clamps the ROI to the reconstructed-image bounds (robust to roi_shape > image).
        rotation_t = torch.tensor(vBF.meta.rotation, device=vBF.device)
        factor = np.sqrt(upsample_int[0] * upsample_int[1])
        last_image = {"img": None}

        # What ``get_G_chunk`` returns is fixed for the duration of this fit:
        # nothing in the TV loop mutates ``vBF`` (``vBF.array.requires_grad`` is
        # False, set above, and the alignment overlay -- when there is one -- is
        # accumulated by the bright-field stage that has already finished).  Yet
        # ``_chunk_image`` re-derives its chunk on every chunk body, and
        # ``G[..., s:e]`` is a non-contiguous last-axis
        # slice, so each of those is a strided compaction of bytes that did not
        # change: 172 MiB of traffic, 0.442 ms, x1250-1700 bodies per fit =
        # ~18 % of it at the published Fig1 config.  Fetch each of the
        # ``n_batches`` distinct chunks ONCE.  (D3-run closed the other two
        # spellings by measurement: handing the strided view to
        # ``correct_aberrations`` instead is WORSE -- the copy moves inside the op
        # and gets a 4-byte-innermost kernel -- and the compaction itself already
        # runs near this card's copy roof, so it can only be deleted, not sped up.)
        #
        # Budgeted, not unconditional: the cache is one more copy of G, and at the
        # 1024^2 scan the chunk loop's docstring was written for that is ~8.6 GiB,
        # i.e. exactly the OOM the chunking exists to prevent.  Streaming
        # residencies (``vbf_gpu`` / ``cpu_pinned``, ``utils/data/out_of_core.py``)
        # are excluded BY NAME rather than by size: their whole budget is that a
        # chunk exists only while it is in use, and their ``G`` property refuses
        # to materialise at all.
        _cache_bytes = (
            int(n_k_tv) * int(vBF.array.shape[0]) * int(vBF.array.shape[1]) * 8
        )
        _g_chunks: dict[tuple[int, int], torch.Tensor] = {}
        if (
            _cache_bytes <= _TV_CHUNK_CACHE_BUDGET_BYTES
            and str(getattr(vBF, "_g_residency", "gpu")) == "gpu"
        ):
            for _s, _e in _chunk_ranges(n_k_tv, n_batches):
                _g_chunks[(_s, _e)] = vBF.get_G_chunk(_s, _e)

        def evaluate_tv(aberr):
            # Reconstruct the phase image (mean over the detector-pixel axis of
            # imag(ifft2(corrected G))) in chunks over that axis. The full
            # [Ny, Nx, N_k] corrected-G + ifft2 cost ~2x17 GB for a 1024^2 scan;
            # chunking caps the peak at one chunk. ifft2 acts per detector pixel,
            # so summing chunk contributions is exact. Above
            # ``_TV_CHUNK_CKPT_BUDGET_BYTES`` each chunk is additionally
            # gradient-checkpointed so per-chunk activations are freed and
            # recomputed in the backward pass; see the ``use_ckpt`` comment
            # below for what is retained when they are not.
            from torch.utils.checkpoint import checkpoint

            # The ``ik``-only factor ``A = aperture(K) * exp(-i chi(K))`` is a
            # function of ``aberr`` alone, so it is built ONCE here and sliced
            # per chunk -- never inside ``_chunk_image``.  The fused kernel would
            # otherwise recompute it (an aperture, an aberration polynomial and a
            # ``cexp``) for every one of its ``Ny * Nx`` threads; hoisting is
            # 1.29x on the kernel at the published Fig1 chunk and bit-exact,
            # because ``_direct_ptychography_build_A`` evaluates the identical
            # six lines the kernel used to.  Per chunk it would be a LOSS: the
            # launch is ~0.042 ms for ~43 elements against a 0.070 ms saving.
            #
            # Safe under the gradient checkpoint below: ``build_aberration_bf_
            # factor`` is ``@torch.no_grad()`` and this tensor is captured by
            # closure, so the recompute reads the identical object the first
            # forward used -- bit-exact a fortiori, and the analytic adjoint
            # never consults it (DISCOVERED D7).
            A_bf = build_aberration_bf_factor(
                aberr.detach(),
                float(semiconvergence_angle),
                float(wavelength),
                Kx,
                Ky,
            )

            def _chunk_image(aberr, s, e):
                # The chunk fetch lives INSIDE the checkpointed function: only
                # (aberr, s, e) are saved for backward, and the recompute
                # re-fetches (or re-FFTs) the same chunk deterministically -- or,
                # when the cache above is populated, reads the identical tensor
                # the first fetch produced, which is deterministic a fortiori.
                Gc = _g_chunks.get((s, e))
                if Gc is None:
                    Gc = vBF.get_G_chunk(s, e)
                if upsample_int[0] > 1 or upsample_int[1] > 1:
                    Gc = torch.tile(
                        Gc, (int(upsample_int[0]), int(upsample_int[1]), 1)
                    )
                # FUSED correct-and-reduce.  The two statements this replaces
                # were ``CorrectAberrations.apply(...)`` -- which allocates an
                # ``[Ny, Nx, n_k]`` corrected chunk, 86.0 MiB at the published
                # Fig1 config -- followed immediately by ``.sum(dim=-1)``, which
                # read all 86 MiB back to collapse them to 1.0 MiB.  Nothing
                # between the two ever looked at the unreduced tensor, and the
                # backward's upstream adjoint was the ``ik``-BROADCAST of a
                # single ``[Ny, Nx]`` image that autograd had to materialise at
                # the same 86 MiB before the analytic kernel could index it.
                # Both passes are gone; the accumulation happens in a register.
                #
                # This is a reassociation of the ``ik`` sum, not a rewrite of the
                # arithmetic, and it is the SAME class and SIZE of perturbation
                # as the sum-first collapse immediately below (measured on the
                # live (512, 512, 43) chunk: chunk image rel 2.73e-07, and
                # against a complex128 reduction of the same corrected G the two
                # float32 forms sit at 3.95e-08 and 1.33e-07 -- both at float32's
                # own error, neither the accurate one).  The fit's own
                # reproducibility floor is far larger; see the comment on the
                # collapse below and docs/perf-lab/reports/2026-08-06-d18-*.md.
                Gprime_summed = correct_aberrations_ksum(
                    Gc,
                    aberr,
                    rotation_t,
                    semiconvergence_angle,
                    wavelength,
                    Qx,
                    Qy,
                    Kx[s:e],
                    Ky[s:e],
                    n_grad_coeffs,
                    A_bf[s:e],
                )
                # Reduce the bright-field-pixel axis BEFORE the transform, not
                # after.  ifft2 acts on the image axes (0, 1) and Im() is
                # R-linear, so
                #     sum_k Im(ifft2(G_k)) == Im(ifft2(sum_k G_k))
                # and this chunk needs ONE two-dimensional transform instead of
                # one per bright-field pixel -- the identity
                # ``enable_chunk_sum_first`` implements for the reconstruction
                # loop (R18), which this body had inlined without it.  The
                # ``factor`` also lands on the [Ny, Nx] real image instead of an
                # [Ny, Nx, n_k] complex one.
                #
                # Unlike the reconstruction switch (``enable_chunk_sum_first``)
                # this is NOT opt-in, and the difference matters: that one feeds
                # published arrays directly, while this is the objective of a
                # nonlinear TV minimisation whose own reproducibility floor is
                # far LARGER than this reassociation.  Measured at the live
                # (512, 512, 43) chunk: the two float32 forms differ by rel
                # 2.73e-07, and against a complex128 reference of the same input
                # they sit at rel 2.62e-07 (this one) and 2.00e-07 (the previous
                # one) -- i.e. they are the same distance from the truth to
                # within 1.3x, and their difference from each other is the size
                # of either one's own float32 error, so neither is the accurate
                # form.  Downstream, the fitted aberration vector moves no more
                # across this change than it moves between two runs of the
                # unmodified code (`wp.tile_atomic_add` in the aberration adjoint
                # makes the gradient non-reproducible, so strong-Wolfe lands
                # elsewhere): on Fig2_carbon every fitted coefficient's
                # cross-change spread is <= 1.5x its own same-arm spread and C10
                # is bit-identical, and on Fig1_Gd2O3 the fit has a SECOND BASIN
                # 0.85 % away in C10 that unmodified code reaches too (1 launch
                # in 10).  See docs/perf-lab/reports/2026-08-06-d2-*.md.
                return (
                    torch.fft.ifft2(Gprime_summed, dim=(0, 1), norm="ortho").imag
                    * factor
                )

            image_sum = None
            # Checkpoint only when the activation it drops is worth a second
            # forward pass.  What NOT checkpointing retains is one saved tensor
            # per chunk -- ``scatterem::correct_aberrations_ksum_fwd`` keeps its
            # ``Gprime_real`` input for the analytic adjoint -- so summed over
            # the chunks it is exactly one upsampled copy of ``vBF.G``, and
            # nothing else: the corrected G is never materialised at all (the
            # ``ik`` sum happens inside the kernel) and the ifft2 output is now a
            # real ``[Ny, Nx]`` image, not a ``[Ny, Nx, n_k]`` complex one.  What the
            # checkpoint COSTS is a whole extra forward per objective
            # evaluation, which is why ``loss.backward`` measures 2.27x the
            # forward it differentiates.  Below the budget, skip it.
            retained_bytes = int(n_k_tv) * int(new_shape[0]) * int(new_shape[1]) * 8
            use_ckpt = (
                torch.is_grad_enabled()
                and aberr.requires_grad
                and retained_bytes > _TV_CHUNK_CKPT_BUDGET_BYTES
            )
            for s, e in _chunk_ranges(n_k_tv, n_batches):
                if use_ckpt:
                    partial = checkpoint(_chunk_image, aberr, s, e, use_reentrant=False)
                else:
                    partial = _chunk_image(aberr, s, e)
                image_sum = partial if image_sum is None else image_sum + partial
            image = image_sum / n_k_tv
            last_image["img"] = image
            roi = _windowed_roi(image, roi_center, roi_shape)
            return _image_sharpness(roi, sharpness_metric)

        aberrations, tv_info = _optimize_aberrations_tv(
            evaluate_tv,
            seed_aberr,
            scales,
            free_mask,
            reg_weight=reg_weight,
            lr=lr if lr <= 1.0 else 0.5,
            max_iter=num_iterations,
        )
        if not tv_info["improved_over_seed"]:
            warnings.warn(
                "TV aberration refinement did not improve sharpness over the seed; "
                "keeping the seed aberrations.",
                UserWarning,
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
            print(f"aberrations = {aberrations[0:4].detach().cpu().numpy()}")
    # (correction_method was validated up front via _resolve_aberration_stages)

    print("\nOptimized aberration coefficients:")
    polar = cartesian_to_polar(
        {
            "C10": aberrations[0].item(),
            "C12a": aberrations[1].item(),
            "C12b": aberrations[2].item(),
            "C21a": aberrations[3].item(),
            "C21b": aberrations[4].item(),
            "C23a": aberrations[5].item(),
            "C23b": aberrations[6].item(),
            "C30": aberrations[7].item(),
        }
    )
    print(f"  C10 (Defocus): {polar['C10']:.3f}")
    print(f"  C12 (Astigmatism magnitude): {polar['C12']:.3f}")
    print(f"  C21 (Coma magnitude): {polar['C21']:.3f}")
    print(f"  C30 (Spherical aberration): {polar['C30']:.3f}")
    print(f"  phi12 (Astigmatism angle): {polar['phi12']:.3f}")
    print(f"  phi21 (Coma angle): {polar['phi21']:.3f}")
    if update_dataset:
        vBF.meta.aberrations.array = aberrations.detach()
        if fit_rotation:
            vBF.meta.rotation = rotation
        if do_tv and direct_ptycho_image is not None:
            vBF.weak_phase_image = direct_ptycho_image.detach()

    if do_bf:
        # The bright-field-shift self-calibration accumulated measured shifts
        # into the vBF's alignment OVERLAY (the do_tv stage above deliberately
        # sees them, matching the historical in-place-ramp behavior). With the
        # vBF cached and REUSED downstream (get_vbf), the overlay must not
        # leak — the reconstruction would double-apply the shifts (overlay
        # ramps plus the fitted-aberration correction). The pristine G cache
        # itself is never touched anymore.
        vBF.clear_alignment_shifts()

    return aberrations.detach(), vBF


def _checkerboard_parity(bright_field_inds: torch.Tensor) -> torch.Tensor:
    """Checkerboard parity of integer detector-grid indices.

    ``bright_field_inds`` are integer (row, col) detector-pixel coordinates
    (``vBF.bright_field_inds_ordered_by_radius``), NOT ``vBF.k`` (which is
    mean-centered floats scaled by the reciprocal sampling — rounding it would
    collapse to one subset). Absolute vs centered indices is immaterial: a
    half-pixel offset only flips which subset is labeled True/False, which does
    not matter for a symmetric half-split.

    Returns ``True`` for the odd colour class (``row + col`` odd), one entry per
    ROW, decided by the index VALUES — so a radius-ordered (neither raster-
    ordered nor contiguous) index list is coloured correctly, unlike anything
    keyed on position within the list.

    Non-integral floats are rejected instead of truncated: casting ``vBF.k``
    would fold distinct rows onto one parity and return a lopsided split, i.e.
    a plausible-looking but wrong SSNR rather than an error.

    No third-party attribution is owed here. The checkerboard two-colouring of
    the integer lattice *is* the parity of ``row + col``; that is the definition
    of a checkerboard, not an implementation of one, so any independent
    implementation necessarily coincides in its arithmetic and there is no
    expression to attribute to anybody.
    """
    if bright_field_inds.numel() == 0:
        # No BF pixels at all: hand back an empty mask so the caller trips its
        # own too-small guard (warn + analytical SSNR) instead of an IndexError.
        return torch.zeros(0, dtype=torch.bool, device=bright_field_inds.device)
    if bright_field_inds.ndim != 2 or bright_field_inds.shape[1] != 2:
        raise ValueError(
            "bright_field_inds must be an (N, 2) tensor of integer (row, col) "
            f"detector indices; got shape {tuple(bright_field_inds.shape)}"
        )
    idx = bright_field_inds
    if idx.is_floating_point():
        rounded = torch.round(idx)
        if not torch.equal(rounded, idx):
            raise ValueError(
                "bright_field_inds must hold whole-number detector-pixel "
                "indices, but got non-integral floats — this is what passing "
                "vBF.k (mean-centered and scaled by the reciprocal sampling) "
                "looks like. Pass vBF.bright_field_inds_ordered_by_radius."
            )
        idx = rounded
    idx = idx.long()
    # Two lattice cells share a colour exactly when their row and column
    # indices agree in parity, so the colour is the disagreement of the two low
    # bits. Two's complement makes `& 1` the parity of negative (centered)
    # indices too, and this equals (row + col) % 2 for every input.
    return torch.ne(idx[:, 0] & 1, idx[:, 1] & 1)


# Minimum number of BF pixels per half-split side for the checkerboard SSNR
# estimate to be meaningful.  Below this threshold we fall back to analytical.
_MIN_HALFSPLIT_PIXELS = 4


# Opt-in fast path for the bright-field chunk reduction in ``_direct_ptychography``.
# OFF by default: the collapse is algebraically exact but reassociates the float32
# chunk sum, so it does not reproduce the default path bit for bit.  See
# :func:`enable_chunk_sum_first`.
_CHUNK_SUM_FIRST = False


def enable_chunk_sum_first(enabled: bool = True) -> bool:
    """Enable the chunk-axis-first reduction in the direct-ptychography loop.

    ``_direct_ptychography`` inverse-transforms a whole bright-field chunk and then
    sums the chunk axis away.  ``ifft2`` is linear and ``Im()`` is R-linear, so

        sum_k Im(ifft2(G_k)) == Im(ifft2(sum_k G_k))

    and reducing the chunk axis *before* the transform replaces one
    two-dimensional transform per bright-field pixel with a single transform per
    chunk.  It also applies the ``sqrt(upsample)`` factor to the ``[H, W]`` real
    result instead of to an ``[H, W, chunk]`` complex tensor whose real half is
    discarded.

    Measured **2.26x** end to end (13.652 -> 6.034 ms, range 2.23-2.29x over
    separate process launches) on a 245**2 scan x 128**2 detector reconstruction,
    717 bright-field pixels, ``n_batches=8``, RTX A6000 -- with peak CUDA memory
    41 MiB *lower*, since the ``[H, W, chunk]`` complex intermediates are never
    materialised.  The win scales with bright-field pixels per chunk, so it is
    **no win at all** on a 62**2 scan (0.997x, ~8 px per chunk and launch-bound)
    and grows with scan size and with small ``n_batches``.

    Default OFF, deliberately.  The identity is exact in exact arithmetic (checked
    to 4.6e-16 of peak in float64 at the live chunk shape), moving the real factor
    across ``Im()`` is bit-exact, and against a complex128 reference the collapsed
    form is *closer* to the truth than the default (2.637e-07 vs 2.716e-07).  The
    two float32 images agree to 3.2e-07 of the image peak -- less than either
    one's own distance from that reference.  But summing before rather than after
    the transform does reassociate the float32 sum, so it is not bit-exact, and
    the published full-field figures are reproduced from the default path.
    Flipping the default is therefore a decision for a person, not for this
    switch.

    ``reduce="none"`` (the empirical-SSNR checkerboard half-split) has no chunk sum
    to collapse and is unaffected either way.

    Args:
        enabled: whether to take the collapsed path.

    Returns:
        bool: the previous setting, so a caller can restore it.
    """
    global _CHUNK_SUM_FIRST
    previous = _CHUNK_SUM_FIRST
    _CHUNK_SUM_FIRST = bool(enabled)
    return previous


def _is_exact_power_of_two(x) -> bool:
    """True when ``x`` is a positive, finite, exact power of two.

    ``math.frexp`` writes ``x = m * 2**e`` with ``0.5 <= |m| < 1``; ``m == 0.5``
    is exactly the "significand is 1.0" case, i.e. multiplication by ``x`` is a
    pure exponent shift and therefore exact.  Written against ``float(x)`` so a
    numpy scalar (what ``np.sqrt`` returns) answers the same as a Python float.
    """
    xf = float(x)
    return xf > 0.0 and math.isfinite(xf) and math.frexp(xf)[0] == 0.5


def _direct_ptychography(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    ds_rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qy: torch.Tensor,
    Qx: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    device: torch.device,
    upsample: np.ndarray,
    reduce: str = "sum",
    A: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    factor = np.sqrt(upsample[0] * upsample[1])
    if reduce == "sum" and _CHUNK_SUM_FIRST:
        # One transform per chunk instead of one per bright-field pixel, and the
        # factor lands on the [H, W] real result rather than an [H, W, chunk]
        # complex one. Reassociates the float32 chunk sum: opt-in, see
        # enable_chunk_sum_first.
        #
        # The aberration correction and the ik reduction are FUSED here, so the
        # (Nqy, Nqx, Nk) corrected chunk is never materialised: on the registered
        # 245**2 x 128**2 reconstruction that deletes a 42.1 MiB write followed
        # immediately by a 42.1 MiB read of the same block, per chunk.  This is the
        # op D18 built for exactly this pair -- ``determine_aberrations`` has used it
        # since; this loop never got it.  ``out`` is deliberately unused on this
        # branch: the fused op does not mutate ``Gprime``, which is the only reason
        # a caller needed a scratch buffer at all.
        Gprime_summed = correct_aberrations_ksum(
            Gprime,
            aberrations,
            ds_rotation,
            semiconvergence_angle,
            wavelength,
            Qx,
            Qy,
            Kx,
            Ky,
            None,
            A,
        )
        return torch.fft.ifft2(Gprime_summed, dim=(0, 1), norm="ortho").imag * factor
    # correct_aberrations_inplace accepts a float OR a pre-built device tensor;
    # hot loops pass a tensor so no per-chunk H2D copy is launched.  ``A`` is the
    # bright-field-only aberration factor, hoisted above the chunk loop by
    # _iter_chunk_images; None means the op builds its own.  ``out`` is a scratch
    # buffer for callers that must keep ``Gprime`` intact: with it the correction
    # is out of place and they need no clone.
    Gprime_corrected = correct_aberrations_inplace(
        Gprime,
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
        A=A,
        out=out,
    )
    if reduce == "sum":
        # The same normalisation fold as _direct_ptychography_kouter, applied here
        # too so the two layouts stay numerically interchangeable: the streaming /
        # alignment-overlay callers reach THIS body while the resident-G ones reach
        # the k-outer twin, and tests/test_eager_vs_ooc_equivalence.py compares the
        # two at atol=1e-7.  Folding one and not the other would put a 1.9e-07
        # difference straight into that comparison.
        G_bf = torch.fft.ifft2(Gprime_corrected, dim=(0, 1), norm="forward")
        scale = factor / math.sqrt(
            Gprime_corrected.shape[0] * Gprime_corrected.shape[1]
        )
        return torch.sum(G_bf.imag, dim=(-1)) * scale
    # reduce="none" keeps ``norm="ortho"``: its consumer (the empirical-SSNR
    # checkerboard half-split) is digested BIT-EXACT, so nothing about a
    # normalisation fold may reach it.
    G_bf = torch.fft.ifft2(Gprime_corrected, dim=(0, 1), norm="ortho")
    if reduce == "none":
        return (G_bf * factor).imag
    raise ValueError(f"invalid reduce: {reduce!r}")


def direct_ptychography(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    upsample: Union[float, str] = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    n_batches: int = 25,
    return_snr: bool = False,
    phase_sign: PhaseSignMode = "positive",
    ssnr_method: str = "analytical",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Performs a joint ptychography reconstruction and aberration determination.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        upsample: Upsampling factor for the diffraction pattern.
        bright_field_mask_threshold: Threshold for the bright field.
        verbosity: Verbosity level.
        n_batches: Number of batches for the vBF.
        return_snr: Whether to compute the ptychography SSNR (method selected by ssnr_method).
        phase_sign: Resolve the global phase sign ambiguity. "positive"
            flips negative-skew phase images so atom columns are positive,
            "negative" enforces the opposite sign, and "preserve" leaves the
            raw direct-ptycho sign unchanged.
        ssnr_method: How to compute the SSNR when return_snr is True.
            "analytical" (default) uses the phase-contrast-transfer-function noise
            model (CTF-aware: ptycho SSNR -> 0 at low q, no DC/low-q transfer);
            "empirical" reconstructs a checkerboard BF half-split and forms the raw
            half-difference power ratio (data-driven, better at high dose).

    Returns:
        torch.Tensor: The reconstructed phase image (Imaginary part).
        torch.Tensor | None: The SNR of the reconstructed phase image if return_snr is True, otherwise None.

    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.
        ValueError: If the upsample is invalid.
    """
    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)
    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)

    # Empirical SSNR owns the reconstruction (checkerboard half-split). Passing the
    # already-built vBF avoids rebuilding it. The half reconstructions call back with
    # return_snr=False, so they fall through to the normal loop (no recursion).
    if return_snr and ssnr_method == "empirical":
        return direct_ptychography_empirical_ssnr(
            vBF,
            upsample=upsample,
            n_batches=n_batches,
            bright_field_mask_threshold=bright_field_mask_threshold,
            snr_blur_sigma=kwargs.get("snr_blur_sigma", 0.0),
            phase_sign=phase_sign,
            verbosity=verbosity,
        )

    device = vBF.device
    new_shape, Qy, Qx = _upsampled_grid(vBF, upsample_int)
    rotation_t = torch.as_tensor(
        float(vBF.meta.rotation), dtype=torch.float32, device=device
    )

    with torch.no_grad():
        if verbosity > 0:
            import time as _time

            _t0 = _time.perf_counter()
        phase_sum = torch.zeros(new_shape, device=device)
        for _, _, img in _iter_chunk_images(
            vBF,
            vBF.meta.aberrations.array,
            rotation_t,
            Qy,
            Qx,
            upsample_int,
            n_batches,
            "sum",
        ):
            phase_sum += img
        # Pixel-count weighting: one global divide. The old
        # `phase_image /= n_batches_eff` averaged per-batch MEANS equally,
        # overweighting a short tail batch (29 vs 34 px at n_batches=64 with
        # 2171 BF pixels -> 1.17x tail overweight).
        phase_image = phase_sum / vBF.n_bright_field

        phase_image = _orient_phase_image(
            phase_image,
            phase_sign=phase_sign,
            verbosity=verbosity,
        )

        if verbosity > 0:
            print(
                f"Time to reconstruct directptychography image: {_time.perf_counter() - _t0:.3f} s"
            )
        if return_snr:  # ssnr_method == "empirical" already returned above
            if ssnr_method == "analytical":
                snr_ptycho = direct_ptychography_ssnr(
                    dataset, upsample=upsample, verbosity=verbosity
                )
            else:
                raise ValueError(f"Unknown ssnr_method: {ssnr_method!r}")
        else:
            snr_ptycho = None
    return phase_image, snr_ptycho


def direct_ptychography_empirical_ssnr(
    dataset,
    upsample="nyquist",
    n_batches: int = 25,
    bright_field_mask_threshold: float = 0.5,
    snr_blur_sigma: float = 0.0,
    phase_sign: "PhaseSignMode" = "positive",
    verbosity: int = 0,
):
    """Full phase image + empirical SSNR from a checkerboard BF half-split.

    Both halves are reconstructed with identical deterministic operations, so all
    aberration/upsample steps cancel in the half-difference. The SSNR is the raw
    half-split power ratio (NO /sqrt(fluence)); it is the SSNR of the full-dose
    reconstruction and is dose- and object-bearing, matching the tcDF channel.
    """
    import warnings

    from scatterem.reconstruction.tilt_corrected_dark_field import (
        compute_ssnr_from_halfset_images,
    )

    # The normal dispatch path passes the pre-built vBF; _resolve_vbf reuses
    # the dataset's cache when handed a raw Dataset4dstem.
    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)

    parity = _checkerboard_parity(vBF.bright_field_inds_ordered_by_radius).to(
        vBF.device
    )
    n_a = int(parity.sum())
    n_b = int((~parity).sum())
    if min(n_a, n_b) < _MIN_HALFSPLIT_PIXELS:
        warnings.warn(
            f"checkerboard BF split too small (n_a={n_a}, n_b={n_b}); "
            "falling back to analytical SSNR"
        )
        return direct_ptychography(
            dataset,
            upsample=upsample,
            n_batches=n_batches,
            return_snr=True,
            phase_sign=phase_sign,
            ssnr_method="analytical",
            verbosity=verbosity,
        )

    # ONE parity-routed pass over shared G chunks: each chunk's per-pixel
    # images are summed into the A or B accumulator by checkerboard parity.
    # No G_A/G_B copies (the old boolean-index splitter duplicated the full
    # ~17 GiB G while the original stayed referenced). Sign is preserved
    # per half so the half-difference is clean (an independent skew-based flip
    # per half would corrupt N=(F_A-F_B)/2); the per-half pixel-count division
    # makes each half identical to a standalone weighted reconstruction.
    upsample_int = _resolve_upsample_int(vBF, upsample, verbosity=0)
    new_shape, Qy, Qx = _upsampled_grid(vBF, upsample_int)
    device = vBF.device
    rotation_t = torch.as_tensor(
        float(vBF.meta.rotation), dtype=torch.float32, device=device
    )
    total_A = torch.zeros(new_shape, device=device)
    total_B = torch.zeros(new_shape, device=device)
    with torch.no_grad():
        for s, e, imag in _iter_chunk_images(
            vBF,
            vBF.meta.aberrations.array,
            rotation_t,
            Qy,
            Qx,
            upsample_int,
            n_batches,
            "none",
        ):
            pm = parity[s:e]
            total_A += imag[..., pm].sum(-1)
            total_B += imag[..., ~pm].sum(-1)

    F_A = total_A / n_a  # phase_sign="preserve" == no orientation flip
    F_B = total_B / n_b

    full = (n_a * F_A + n_b * F_B) / (n_a + n_b)
    full = _orient_phase_image(full, phase_sign=phase_sign, verbosity=verbosity)

    # Reconstruction grid upsample factor (same axis math as compute_ssnr expects).
    upsample_grid = np.array(F_A.shape) / np.array(
        [int(vBF.array.shape[0]), int(vBF.array.shape[1])]
    )
    sampling_up = tuple((np.array(vBF.sampling[:2]) / upsample_grid).tolist())

    ssnr_1d, _, bin_idx = compute_ssnr_from_halfset_images(
        F_A, F_B, sampling_up, gaussian_sigma=snr_blur_sigma, verbosity=verbosity
    )
    ssnr_ptycho = ssnr_1d[bin_idx].reshape(F_A.shape)  # NO /sqrt(fluence)
    return full, ssnr_ptycho


#: Byte budget for the PLANE-BATCHED branch of :func:`_sweep_chunk_kouter`.
#
# Batching the depth planes into one launch grid needs the corrected chunk (and the
# transform that consumes it) materialised for every plane at once instead of one at a
# time, so it trades ``2 * (n_depths - 1) * chunk_bytes`` of device residency for
# ``n_chunks * (n_depths - 1)`` fewer correction calls.  Whether that trade is worth
# anything is a property of the ROW'S BALANCE, not of the code (D272's law), and it was
# measured on both registered sizes with the identical instrument
# (``_probe_d276_price.py``, arm A vs arm B3, both bit-exact):
#
#   ``small``   batched scratch    3.373 MiB   ->  **2.826x** (15.910 -> 5.629 ms),
#               row peak 146.5 MiB, 91.3 % GPU-idle: the removed host calls ARE the row.
#   ``medium``  batched scratch  294.923 MiB   ->  1.0104x (49.963 -> 49.449 ms),
#               row peak 4464.4 MiB, 6.5 % GPU-idle: the removed host calls are already
#               hidden behind device work, so 500 MiB buys nothing measurable.
#
# 16 MiB sits ~5x above the first and ~18x below the second, so neither registered size
# is anywhere near the boundary and the gate is not a fitted threshold on a speed
# measurement (which is what D233 caught being wrong when shape and count move together)
# -- it is a memory guard on a trade whose two ends have both been measured.
_SWEEP_PLANE_BATCH_BYTES = 16 << 20


def _sweep_chunk_kouter(
    Gc_k,
    s: int,
    e: int,
    phase_sums,
    n_depths: int,
    ab_planes,
    A_planes,
    rotation_t,
    semiconvergence_angle,
    wavelength,
    Qx,
    Qy,
    Kx,
    Ky,
    factor,
    ab_stack=None,
    A_stack=None,
):
    """Accumulate all ``n_depths`` planes of one ``[Nk, Nqy, Nqx]`` bright-field
    chunk into ``phase_sums``: correct -> ``ifft2(dim=(-2,-1))`` -> ``sum(dim=0)``.

    ``Gc_k`` is only ever READ (``correct_aberrations_kouter`` is out of place), so
    it may be a free view of the caller's resident ``G`` -- which is what
    :func:`_g_chunk_kouter_view` hands back, and the reason the sweep pays no
    layout copy at all on the common path.  One scratch buffer for the corrected
    chunk is reused by every plane instead of cloning the chunk per plane: the
    clone was a full read+write pass on top of the kernel's own read+write, four
    passes where two do.  Safe across planes -- the ``ifft2`` that consumes the
    buffer is issued on the same stream before the next plane's kernel overwrites
    it.  Explicit shape/dtype, NOT ``empty_like``, so a dense-but-permuted input
    cannot give the Warp view a non-contiguous destination.

    ``ab_stack``/``A_stack`` are the same ``ab_planes``/``A_planes`` packed as
    ``[n_depths, n_ab]`` and ``[n_bf_total, n_depths, 2]``; given both, and given a
    chunk small enough for :data:`_SWEEP_PLANE_BATCH_BYTES`, the whole plane loop
    collapses into ONE correction launch, ONE batched ``ifft2`` and ONE reduction.
    Bit-exact: the correction is per-element (so plane ``ip`` of the batched output
    holds the same words the per-plane launch writes), the ``ifft2`` runs over the
    same inner two axes of the same contiguous per-plane blocks at a higher batch
    count, and the reduction sums the same contiguous ``Nk`` planes.  Confirmed by
    digest at both registered sizes, and it is the digest and not this argument
    that is load-bearing -- cuFFT is free to choose a different algorithm at a
    different batch count.
    """
    if (
        ab_stack is not None
        and A_stack is not None
        and n_depths * Gc_k.numel() * Gc_k.element_size()
        <= _SWEEP_PLANE_BATCH_BYTES
    ):
        corrected = torch.empty(
            (n_depths, *Gc_k.shape), dtype=Gc_k.dtype, device=Gc_k.device
        )
        correct_aberrations_kouter_planes(
            Gc_k,
            ab_stack,
            rotation_t,
            semiconvergence_angle,
            wavelength,
            Qx,
            Qy,
            Kx,
            Ky,
            A_stack[s:e],
            corrected,
        )
        G_bf = torch.fft.ifft2(corrected, dim=(-2, -1), norm="ortho")
        if _is_exact_power_of_two(factor):
            phase_sums += torch.sum(G_bf.imag, dim=1) * factor
        else:
            phase_sums += torch.sum((G_bf * factor).imag, dim=1)
        return
    Gc_k_corrected = torch.empty(
        Gc_k.shape, dtype=Gc_k.dtype, device=Gc_k.device
    )
    for i in range(n_depths):
        correct_aberrations_kouter(
            Gc_k,
            ab_planes[i],
            rotation_t,
            semiconvergence_angle,
            wavelength,
            Qx,
            Qy,
            Kx,
            Ky,
            A_planes[i][s:e],
            Gc_k_corrected,
        )
        G_bf = torch.fft.ifft2(Gc_k_corrected, dim=(-2, -1), norm="ortho")
        if _is_exact_power_of_two(factor):
            # Same power-of-two argument as ``_direct_ptychography``: scaling the
            # reduced [H, W] image instead of the [Nk, H, W] complex one is
            # bit-exact when the factor is an exact power of two, and spares a
            # full read+write pass over the chunk.
            phase_sums[i] += torch.sum(G_bf.imag, dim=0) * factor
        else:
            phase_sums[i] += torch.sum((G_bf * factor).imag, dim=0)




def _gradient_energy(stack: torch.Tensor) -> torch.Tensor:
    """Per-plane squared-gradient map (local sharpness), shape == stack."""
    w = torch.zeros_like(stack)
    gy = stack[:, 1:, :] - stack[:, :-1, :]
    gx = stack[:, :, 1:] - stack[:, :, :-1]
    w[:, :-1, :] += gy**2
    w[:, :, :-1] += gx**2
    return w


def _blur_stack(stack: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur applied per plane of an (N, H, W) stack."""
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=stack.device) - k // 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = (g / g.sum()).to(stack.dtype)
    x = stack[:, None]
    x = torch.nn.functional.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    return x[:, 0]






def direct_ptychography_ssnr(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    upsample: Union[float, str] = "nyquist",
    verbosity: int = 0,
) -> torch.Tensor:
    """
    Compute the analytical spectral signal-to-noise ratio (SSNR) for direct ptychography.

    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object.
        upsample: Upsampling factor for the diffraction pattern.
        verbosity: Verbosity level. If > 1, plots diagnostics.

    Returns:
        torch.Tensor: The analytical SSNR in corner-centered (non-shifted) layout.
    """
    dalpha0 = dataset.sampling[-1] * dataset.meta.wavelength
    pctf_corner_center = phase_contrast_transfer_function(
        dataset, verbosity=1, upsample=upsample
    )
    pctf = torch.fft.fftshift(pctf_corner_center)

    # Calculate ptycho noise
    # Build the noise grid on the SAME upsample-aware axis as the PCTF (defect fix:
    # previously used un-upsampled sampling, compressing the 2R overlap cutoff by U).
    # The overlap-area math is done on-device (torch) so the whole SSNR path stays
    # on the dataset's device — no GPU->CPU->GPU round-trip.
    U = np.array(pctf.shape) / np.array(dataset.shape[:2])
    q = fft_frequencies_2d(
        pctf.shape,
        np.array(dataset.sampling[:2]) / U,
        device=dataset.device,
    )
    qn = torch.norm(q, dim=0)
    q1d = qn.view(-1)
    R = dataset.meta.semiconvergence_angle / dataset.meta.wavelength
    delta_k = dataset.sampling[-1]
    N2, N3 = double_and_triple_pixel_counts_torch(q1d, R, delta_k)
    rBF = dataset.meta.semiconvergence_angle / dalpha0
    Nalpha = np.pi * rBF**2
    ptycho_noise_squared = (N2 + N3).reshape(pctf.shape) / Nalpha
    ptycho_noise_2d = torch.sqrt(ptycho_noise_squared).to(
        device=dataset.device, dtype=torch.float32
    )
    snr_ptycho_analytical = pctf_corner_center / ptycho_noise_2d
    # Analytical Eq.13 form is an amplitude, dose- and object-independent quantity.
    # The empirical channels are POWER ratios proportional to dose. Square to a power
    # ratio and scale by the real fluence so the fallback shares the channels'
    # reference (object power |Psi_s|^2 is still assumed flat — fallback limitation).
    # fluence_per_probe exists only on Dataset4dstem, not the vBF sibling; fall back
    # to 1.0 (per-unit-dose) when a vBF is passed directly.
    fluence = float(getattr(dataset, "fluence_per_probe", 1.0))
    snr_ptycho_analytical = snr_ptycho_analytical**2 * fluence
    snr_ptycho_analytical[ptycho_noise_2d == 0] = 0

    if verbosity > 1:
        nconf = NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        vis.show_2d([pctf], cbar=True, title=["pctf"], norm=nconf)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(N2.reshape(pctf.shape).cpu().numpy())
        ax[1].imshow(N3.reshape(pctf.shape).cpu().numpy())
        ax[2].imshow(pctf_corner_center.cpu().numpy())
        ax[3].imshow((ptycho_noise_2d == 0).cpu().numpy())
        plt.show()
        plt.tight_layout()

        p2 = torch.fft.fftshift(snr_ptycho_analytical).cpu().numpy()
        p2s = p2.shape[0] // 4
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        imax = ax.imshow(p2[p2s:-p2s, p2s:-p2s], cmap="magma")
        plt.colorbar(imax)
        plt.title("snr_ptycho_analytical")
        plt.tight_layout()
        plt.show()

    return snr_ptycho_analytical
