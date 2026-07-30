"""
Diffraction-space calibration from the scanned centre of mass.

Two things happen here. First, the centre of mass of every diffraction
pattern is measured and its slow drift across the scan (the descan) is fitted
away with a smooth surface -- see :func:`_calculate_intensities_center_of_mass`
and the ``plane`` / ``parabola`` / ``bezier_two`` models it can fit. Second,
the residual CoM field is used to recover the rotation between the scan axes
and the detector axes, by exploiting the fact that a correctly oriented field
is irrotational -- see
:func:`_solve_for_center_of_mass_relative_rotation`.

Everything is torch-native: functions accept and return ``torch.Tensor``,
take a ``device`` argument (defaulting to CUDA when present), and the angle
sweep and the surface fits are batched rather than looped.
"""

from __future__ import annotations

import warnings
from inspect import signature
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(
    x,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = _DEFAULT_DEVICE,
) -> torch.Tensor:
    """Convert *x* (ndarray, tensor, scalar) to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# 2-D fitting (torch-native, GPU-capable)
# ---------------------------------------------------------------------------


# ---- fitting basis functions ----------------------------------------------
#
# Each one evaluates a surface model at the coordinates ``xy = (x, y)`` given
# its coefficients, and works elementwise on anything supporting arithmetic
# (float, ndarray, Tensor). All three are linear in their coefficients, so the
# fit itself goes through the design matrix in ``_linear_design_matrix``; these
# are the closed forms used to evaluate a model once its coefficients are known.
# The coefficient ORDER of each signature is the column order of that design
# matrix -- do not permute one without the other.


def plane(xy, mx, my, b):
    """Affine surface ``z = mx*x + my*y + b`` (gradient ``(mx, my)``, offset ``b``)."""
    x, y = xy[0], xy[1]
    return mx * x + my * y + b


def parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    """General 2-D quadratic surface, evaluated in Horner form.

    ``z = c0 + cx1*x + cx2*x^2 + cy1*y + cy2*y^2 + cxy*x*y`` -- the complete
    second-order polynomial in two variables, grouped as
    ``c0 + x*(cx1 + cx2*x + cxy*y) + y*(cy1 + cy2*y)`` so each variable is
    touched once.
    """
    x, y = xy[0], xy[1]
    return c0 + x * (cx1 + cx2 * x + cxy * y) + y * (cy1 + cy2 * y)


def _bernstein_quadratic(t):
    """The three degree-2 Bernstein polynomials ``B_k(t) = C(2,k) t^k (1-t)^(2-k)``.

    Returned in order ``k = 0, 1, 2``; they sum to one for any ``t``.
    """
    u = 1 - t
    return u * u, 2 * u * t, t * t


def bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):
    """Tensor-product quadratic Bezier (biquadratic Bernstein) surface.

    ``z = sum_{i,j} c_ij * B_i(x) * B_j(y)`` over the degree-2 Bernstein basis
    of :func:`_bernstein_quadratic`, i.e. a 3x3 grid of control values ``c_ij``
    with ``i`` indexing ``x`` and ``j`` indexing ``y``. Because the basis is a
    partition of unity the surface interpolates the corner coefficients
    (``c00`` at ``(0,0)``, ``c22`` at ``(1,1)``) and stays within the convex
    hull of all nine.
    """
    bx = _bernstein_quadratic(xy[0])
    by = _bernstein_quadratic(xy[1])
    control = ((c00, c01, c02), (c10, c11, c12), (c20, c21, c22))
    return sum(bx[i] * sum(control[i][j] * by[j] for j in range(3)) for i in range(3))


def _make_xy_flat(
    shape: Tuple[int, int],
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create flattened normalized coordinate grid matching NumPy fit_2D."""
    x = torch.linspace(0, 1, shape[0], device=device, dtype=dtype)
    y = torch.linspace(0, 1, shape[1], device=device, dtype=dtype)
    rx, ry = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((rx.reshape(-1), ry.reshape(-1)), dim=0)  # (2, N)


def _linear_design_matrix(function, xy: torch.Tensor) -> Optional[torch.Tensor]:
    """Return design matrix for linear-in-parameters basis functions."""
    x = xy[0]
    y = xy[1]
    if function is plane:
        cols = [x, y, torch.ones_like(x)]
    elif function is parabola:
        cols = [torch.ones_like(x), x, x * x, y, y * y, x * y]
    elif function is bezier_two:
        cols = [
            ((1 - x) ** 2) * ((1 - y) ** 2),
            2 * ((1 - x) ** 2) * (1 - y) * y,
            ((1 - x) ** 2) * (y**2),
            2 * (1 - x) * x * ((1 - y) ** 2),
            4 * (1 - x) * x * (1 - y) * y,
            2 * (1 - x) * x * (y**2),
            (x**2) * ((1 - y) ** 2),
            2 * (x**2) * (1 - y) * y,
            (x**2) * (y**2),
        ]
    else:
        return None
    return torch.stack(cols, dim=1)  # (N, P)


def _fit_linear_model(
    A_all: torch.Tensor,
    z_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    target_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares fit for linear basis functions."""
    A = A_all[mask_flat]
    z = z_flat[mask_flat]
    if A.shape[0] == 0:
        raise ValueError("No valid points to fit.")

    popt = torch.linalg.lstsq(A, z.unsqueeze(1)).solution.squeeze(1)
    fit_all = (A_all @ popt).reshape(target_shape)

    n_obs = int(A.shape[0])
    n_params = int(A.shape[1])
    dof = max(n_obs - n_params, 1)
    resid = z - A @ popt
    sigma2 = (resid.square().sum() / dof) if n_obs > 0 else torch.tensor(0.0, device=z.device, dtype=z.dtype)
    pcov = sigma2 * torch.linalg.pinv(A.T @ A)
    return popt, pcov, fit_all


def _fit_nonlinear_model(
    function,
    xy_all: torch.Tensor,
    z_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    target_shape: Tuple[int, int],
    p0: torch.Tensor,
    max_iter: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LBFGS fallback for generic nonlinear basis."""
    xy = xy_all[:, mask_flat]
    z = z_flat[mask_flat]
    if z.numel() == 0:
        raise ValueError("No valid points to fit.")

    p = p0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = function(xy, *list(p.unbind()))
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred, dtype=z.dtype, device=z.device)
        loss = torch.mean((pred - z) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    fit_all = function(xy_all, *list(p.detach().unbind()))
    if not isinstance(fit_all, torch.Tensor):
        fit_all = torch.as_tensor(fit_all, dtype=z_flat.dtype, device=z_flat.device)
    fit_all = fit_all.reshape(target_shape)

    # Covariance estimation for generic nonlinear fit is not robustly available here.
    pcov = torch.full(
        (p.numel(), p.numel()),
        float("nan"),
        dtype=p.dtype,
        device=p.device,
    )
    return p.detach(), pcov, fit_all


def fit_2D(
    function,
    data: torch.Tensor,
    data_mask: Optional[torch.Tensor] = None,
    popt=None,
    robust: bool = False,
    robust_steps: int = 3,
    robust_thresh: float = 2,
    device: Union[str, torch.device] = _DEFAULT_DEVICE,
):
    """
    2-D least-squares fit with torch-native optimization.

    Parameters / returns mirror the NumPy version in ``calibration.py``.
    """
    data = _to_tensor(data, dtype=torch.float32, device=device)
    data64 = data.to(torch.float64)
    shape = data64.shape
    if len(shape) != 2:
        raise ValueError(f"data must be 2D, got shape {shape}")

    xy = _make_xy_flat(shape, device=device, dtype=torch.float64)  # (2, N)
    z_flat = data64.reshape(-1)

    if data_mask is None:
        mask = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        mask = _to_tensor(data_mask, dtype=torch.bool, device=device)
        if mask.shape != shape:
            raise ValueError(f"data_mask must have shape {shape}, got {tuple(mask.shape)}")

    n_params = len(signature(function).parameters) - 1
    if popt is None:
        p0 = torch.zeros(n_params, dtype=torch.float64, device=device)
    else:
        p0 = _to_tensor(popt, dtype=torch.float64, device=device).reshape(-1)
        if p0.numel() != n_params:
            raise ValueError(
                f"Initial popt has {p0.numel()} params, expected {n_params}."
            )

    if not robust:
        robust_steps = 0

    current_mask = mask.clone()
    popt_t = p0
    pcov_t = torch.full((n_params, n_params), float("nan"), dtype=torch.float64, device=device)
    fit_ar = torch.zeros_like(data64)

    A_all = _linear_design_matrix(function, xy)

    for k in range(robust_steps + 1):
        mask_flat = current_mask.reshape(-1)
        if A_all is not None:
            popt_t, pcov_t, fit_ar = _fit_linear_model(
                A_all, z_flat, mask_flat, shape
            )
        else:
            popt_t, pcov_t, fit_ar = _fit_nonlinear_model(
                function, xy, z_flat, mask_flat, shape, popt_t
            )

        if k < robust_steps:
            fit_mse = (fit_ar - data64) ** 2
            thresh = fit_mse.mean() * (robust_thresh**2)
            current_mask = current_mask & (fit_mse <= thresh)
            popt_t = popt_t.detach()

    return popt_t, pcov_t, fit_ar.to(dtype=torch.float32), current_mask


#: Surface models :func:`fit_origin` will fit, keyed by the name callers pass.
_ORIGIN_SURFACES = {"plane": plane, "parabola": parabola, "bezier_two": bezier_two}


def fit_origin(
    data: Tuple[torch.Tensor, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    fit_function: str = "plane",
    return_fit_params: bool = False,
    robust: bool = False,
    robust_steps: int = 3,
    robust_thresh: float = 2,
    device: Union[str, torch.device] = _DEFAULT_DEVICE,
):
    """
    Fit the drift of the diffraction-space origin across the scan.

    The two components of the measured origin are fitted independently, each
    with the same smooth surface model, and the residuals returned alongside.

    Parameters
    ----------
    data : (Tensor, Tensor)
        Measured origin, as (x-component, y-component), each (Rx, Ry).
    mask : Tensor, optional
        Boolean (Rx, Ry); scan positions where False are excluded from the fit
        but still get a fitted value and a residual.
    fit_function : str
        ``'constant'`` (per-component mean) or one of the keys of
        :data:`_ORIGIN_SURFACES`.
    return_fit_params : bool
        Also return the coefficients and covariances of the two fits. They are
        ``None`` for ``fit_function='constant'``, which has no fitted model.
    robust, robust_steps, robust_thresh
        Outlier rejection, forwarded to :func:`fit_2D`.
    device : str or torch.device

    Returns
    -------
    (qx0_fit, qy0_fit, qx0_residuals, qy0_residuals)
        Each (Rx, Ry). If ``return_fit_params``, a 2-tuple of that and
        ``(popt_x, popt_y, pcov_x, pcov_y)``.
    """
    if not (isinstance(data, tuple) and len(data) == 2):
        raise ValueError("data must be a 2-tuple of (x, y) origin components")
    qx0_meas = _to_tensor(data[0], device=device)
    qy0_meas = _to_tensor(data[1], device=device)
    if qx0_meas.ndim != 2 or qx0_meas.shape != qy0_meas.shape:
        raise ValueError(
            "origin components must be 2-D and the same shape, got "
            f"{tuple(qx0_meas.shape)} and {tuple(qy0_meas.shape)}"
        )

    popt_x = popt_y = pcov_x = pcov_y = None

    if fit_function == "constant":
        qx0_fit = qx0_meas.mean() * torch.ones_like(qx0_meas)
        qy0_fit = qy0_meas.mean() * torch.ones_like(qy0_meas)
    else:
        try:
            surface = _ORIGIN_SURFACES[fit_function]
        except KeyError:
            allowed = ", ".join(("constant", *_ORIGIN_SURFACES))
            raise ValueError(
                f"unknown fit_function {fit_function!r}; expected one of {allowed}"
            ) from None

        # fit_2D treats data_mask=None as "use every point", so the masked and
        # unmasked cases are the same call.
        mask_t = None if mask is None else _to_tensor(mask, torch.bool, device)
        fits = [
            fit_2D(
                surface,
                measured,
                data_mask=mask_t,
                robust=robust,
                robust_steps=robust_steps,
                robust_thresh=robust_thresh,
                device=device,
            )
            for measured in (qx0_meas, qy0_meas)
        ]
        (popt_x, pcov_x, qx0_fit, _), (popt_y, pcov_y, qy0_fit, _) = fits

    fitted = (qx0_fit, qy0_fit, qx0_meas - qx0_fit, qy0_meas - qy0_fit)
    if return_fit_params:
        return fitted, (popt_x, popt_y, pcov_x, pcov_y)
    return fitted


# ---------------------------------------------------------------------------
# Centre-of-mass calculation
# ---------------------------------------------------------------------------


def _calculate_intensities_center_of_mass(
    intensities: torch.Tensor,
    reciprocal_sampling: Tuple[float, float],
    dp_mask: Optional[torch.Tensor] = None,
    fit_function: Optional[str] = "plane",
    com_shifts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    com_measured: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    vectorized_calculation: bool = True,
    device: Union[str, torch.device] = _DEFAULT_DEVICE,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute and fit diffraction-pattern centre-of-mass.

    Parameters
    ----------
    intensities : Tensor, shape (Rx, Ry, Qx, Qy)
    reciprocal_sampling : (dx, dy)
    dp_mask : Tensor, shape (Qx, Qy), optional
    fit_function : str or None
    com_shifts : (Tensor, Tensor), optional — pre-computed fitted CoM
    com_measured : (Tensor, Tensor), optional — pre-computed measured CoM
    device : str or torch.device

    Returns
    -------
    com_measured_x, com_measured_y,
    com_fitted_x, com_fitted_y,
    com_normalized_x, com_normalized_y
    """
    intensities = _to_tensor(intensities, device=device)

    if com_measured is not None:
        com_measured_x = _to_tensor(com_measured[0], device=device)
        com_measured_y = _to_tensor(com_measured[1], device=device)
    else:
        Rx, Ry, Qx, Qy = intensities.shape
        kx = torch.arange(Qx, dtype=torch.float32, device=device)
        ky = torch.arange(Qy, dtype=torch.float32, device=device)
        kxa, kya = torch.meshgrid(kx, ky, indexing="ij")  # (Qx, Qy)

        if dp_mask is not None:
            dp_mask = _to_tensor(dp_mask, device=device)
            if dp_mask.shape != (Qx, Qy):
                raise ValueError(
                    f"dp_mask must match the detector shape {(Qx, Qy)}, "
                    f"got {tuple(dp_mask.shape)}"
                )
            intensities_mask = intensities * dp_mask[None, None]
        else:
            intensities_mask = intensities

        if vectorized_calculation:
            intensities_sum = intensities_mask.sum(dim=(-2, -1))  # (Rx, Ry)
            com_measured_x = (
                (intensities_mask * kxa[None, None]).sum(dim=(-2, -1)) / intensities_sum
            )
            com_measured_y = (
                (intensities_mask * kya[None, None]).sum(dim=(-2, -1)) / intensities_sum
            )
        else:
            com_measured_x = torch.zeros((Rx, Ry), dtype=torch.float32, device=device)
            com_measured_y = torch.zeros((Rx, Ry), dtype=torch.float32, device=device)
            for rx in range(Rx):
                for ry in range(Ry):
                    masked_intensity = intensities_mask[rx, ry]
                    summed_intensity = masked_intensity.sum()
                    com_measured_x[rx, ry] = (masked_intensity * kxa).sum() / summed_intensity
                    com_measured_y[rx, ry] = (masked_intensity * kya).sum() / summed_intensity

    # Fit origin
    if com_shifts is None:
        if fit_function is not None:
            finite_mask = torch.isfinite(com_measured_x)
            com_shifts = fit_origin(
                (com_measured_x, com_measured_y),
                fit_function=fit_function,
                mask=finite_mask,
                device=device,
            )
            com_fitted_x = com_shifts[0].to(dtype=torch.float32)
            com_fitted_y = com_shifts[1].to(dtype=torch.float32)
        else:
            com_fitted_x = com_measured_x.clone()
            com_fitted_y = com_measured_y.clone()
    else:
        com_fitted_x = _to_tensor(com_shifts[0], device=device)
        com_fitted_y = _to_tensor(com_shifts[1], device=device)

    com_normalized_x = torch.nan_to_num(com_measured_x - com_fitted_x) * reciprocal_sampling[0]
    com_normalized_y = torch.nan_to_num(com_measured_y - com_fitted_y) * reciprocal_sampling[1]

    return (
        com_measured_x, com_measured_y,
        com_fitted_x, com_fitted_y,
        com_normalized_x, com_normalized_y,
    )


# ---------------------------------------------------------------------------
# 180° disambiguation helper
# ---------------------------------------------------------------------------


def _disambiguate_rotation_180(
    com_x: torch.Tensor,
    com_y: torch.Tensor,
    verbose: bool = True,
) -> bool:
    """
    Determine whether the corrected CoM field needs a sign flip.

    The curl-based rotation search has a 180° ambiguity: both θ and θ+180°
    yield the same |curl|, but only one gives the correct sign of the
    phase gradient.  For a thin sample with positive projected atomic
    potential, the integrated phase φ (obtained from the CoM = ∇φ) should
    be **positively skewed** (sharp positive peaks at atom sites, shallow
    negative background).

    This function integrates (com_x, com_y) in Fourier space to recover φ
    and checks its skewness.  If the skewness is negative, the CoM vectors
    should be negated while the geometric rotation is left unchanged.

    Parameters
    ----------
    com_x, com_y : Tensor (Rx, Ry)
        Corrected (rotation-applied) centre-of-mass components.
    verbose : bool

    Returns
    -------
    needs_flip : bool
        True if the corrected CoM vectors should be negated.
    """
    Rx, Ry = com_x.shape
    device = com_x.device

    # Frequency grids (real-space pixel units → reciprocal pixels)
    kx = torch.fft.fftfreq(Rx, device=device).unsqueeze(1)   # (Rx, 1)
    ky = torch.fft.fftfreq(Ry, device=device).unsqueeze(0)   # (1, Ry)

    k_sq = kx ** 2 + ky ** 2
    k_sq[0, 0] = 1.0  # avoid division by zero at DC

    # Fourier transform of CoM components
    com_x_fft = torch.fft.fft2(com_x)
    com_y_fft = torch.fft.fft2(com_y)

    # Integration: φ = F^{-1}[ (-i kx F[com_x] - i ky F[com_y]) / (kx²+ky²) ]
    # Since F[∂φ/∂x] = i·kx·F[φ] ⇒ F[φ] = F[com_x]/(i·kx) etc.
    # Using both components for robustness via least-squares in Fourier space:
    phi_fft = (-1j * kx * com_x_fft - 1j * ky * com_y_fft) / k_sq
    phi_fft[0, 0] = 0.0  # zero mean

    phi = torch.fft.ifft2(phi_fft).real

    # Skewness = E[(x-μ)³] / σ³  — positive for right-tailed (atom peaks)
    phi_mean = phi.mean()
    phi_std = phi.std()
    if phi_std < 1e-12:
        if verbose:
            print("  [disambiguate_180] phase field has near-zero std — "
                  "skipping disambiguation.")
        return False

    skewness = ((phi - phi_mean) ** 3).mean() / (phi_std ** 3)
    skew_val = float(skewness.item())

    needs_flip = skew_val < 0

    if verbose:
        print(f"  [disambiguate_180] integrated-phase skewness = {skew_val:+.4f}  "
              f"→ {'adding 180°' if needs_flip else 'no flip needed'}")

    return needs_flip


# ---------------------------------------------------------------------------
# Rotation / transpose search  (vectorised on GPU)
# ---------------------------------------------------------------------------

#: Emitted when the handedness test prefers the swapped-axes hypothesis. A swap
#: is not a rotation, so the caller has to apply it to the data itself.
_TRANSPOSE_ADVICE = (
    "Detector axes look swapped relative to the scan axes: transpose the "
    "diffraction patterns before using this rotation."
)


def _solve_for_center_of_mass_relative_rotation(
    _com_measured_x: torch.Tensor,
    _com_measured_y: torch.Tensor,
    _com_normalized_x: torch.Tensor,
    _com_normalized_y: torch.Tensor,
    rotation_angles_deg: Optional[torch.Tensor] = None,
    plot_rotation: bool = True,
    plot_center_of_mass: str = "default",
    maximize_divergence: bool = False,
    force_com_rotation: Optional[float] = None,
    force_com_transpose: Optional[bool] = None,
    disambiguate_180: bool = False,
    scan_sampling: Tuple[float, float] = (1.0, 1.0),
    scan_units: Tuple[str, str] = ("pixels", "pixels"),
    verbose: bool = True,
    device: Union[str, torch.device] = _DEFAULT_DEVICE,
    **kwargs,
) -> Tuple[float, bool, torch.Tensor, torch.Tensor]:
    r"""
    Recover the rotation between the scan axes and the detector axes.

    For a thin specimen the diffraction centre of mass is proportional to the
    gradient of the projected potential, so as a vector field it must be
    irrotational. Rotating the measured field by the wrong angle mixes the two
    components and injects a spurious curl. Sweeping the trial angle and taking
    the one whose rotated field has the smallest mean :math:`|\nabla \times
    \mathrm{CoM}|` therefore recovers the true scan/detector rotation; the mean
    :math:`|\nabla \cdot \mathrm{CoM}|`, which is *largest* at the same angle,
    is available as an alternative objective. A detector whose axes are swapped
    relative to the scan cannot be undone by any rotation, so both handednesses
    are searched and the better one reported as ``transpose``.

    This estimator is due to Savitzky et al., *Ultramicroscopy* **231**, 113633
    (2021) (py4DSTEM), building on the CoM/DPC analysis of Ophus,
    *Microsc. Microanal.* **25**, 563 (2019); the implementation here is
    independent, batching every trial angle into one tensor operation.

    Both objectives are evaluated with centred differences on the interior of
    the scan, which is why one pixel is dropped at each edge.

    Parameters
    ----------
    _com_measured_x, _com_measured_y : Tensor (Rx, Ry)
        Raw CoM components along the scan-x and scan-y axes. Used for plotting
        only; the search runs on the normalized pair.
    _com_normalized_x, _com_normalized_y : Tensor (Rx, Ry)
        CoM components with the fitted origin removed and scaled to reciprocal
        units -- the field whose curl is minimized.
    rotation_angles_deg : Tensor, optional
        Trial angles in degrees. Defaults to a 1-degree sweep over
        [-89, 90), which is a half-turn because the objective is invariant
        under a 180-degree rotation (see ``disambiguate_180``).
    plot_rotation : bool, optional
        Plot the objective against trial angle, marking the chosen one.
    plot_center_of_mass : str, optional
        ``'default'`` plots the corrected CoM pair; ``'all'`` also plots the
        measured and normalized pairs; anything else plots nothing.
    maximize_divergence : bool, optional
        Select the angle by maximizing mean |divergence| instead of
        minimizing mean |curl|.
    force_com_rotation : float, optional
        Skip the angle sweep and use this angle (degrees).
    force_com_transpose : bool, optional
        Skip the handedness test and use this value.
    disambiguate_180 : bool, optional
        The objective cannot distinguish an angle from that angle plus 180
        degrees, since negating the whole field leaves both curl and
        divergence magnitudes untouched. When True, integrate the corrected
        field to a phase and negate the field if that phase comes out
        negatively skewed -- a thin specimen's positive projected potential
        gives sharp positive peaks on a shallow background, hence positive
        skew. The reported rotation is left alone (see the comment at the
        call site). Default False, which preserves the historical behaviour.
    scan_sampling, scan_units : tuple, optional
        Real-space pixel size and its unit, for plot axes only.
    verbose : bool, optional
        Emit the chosen angle and any forced overrides as ``UserWarning``.
    device : str or torch.device

    Returns
    -------
    _rotation_best_rad : float
        Chosen rotation in radians, wrapped to [-pi, pi).
    _rotation_best_transpose : bool
        True if the detector axes must be swapped before the rotation.
    _com_x, _com_y : Tensor (Rx, Ry)
        The normalized CoM field with the transpose and rotation applied.
    """

    _com_measured_x = _to_tensor(_com_measured_x, device=device)
    _com_measured_y = _to_tensor(_com_measured_y, device=device)
    _com_normalized_x = _to_tensor(_com_normalized_x, device=device)
    _com_normalized_y = _to_tensor(_com_normalized_y, device=device)

    assert _com_normalized_x.ndim == 2 and _com_normalized_y.ndim == 2

    if rotation_angles_deg is None:
        rotation_angles_deg = torch.arange(-89.0, 90.0, 1.0, device=device)
    else:
        rotation_angles_deg = _to_tensor(rotation_angles_deg, device=device)

    # ------------------------------------------------------------------
    # Helper: compute curl or div for a given (com_x, com_y) batch
    # com_x, com_y: (A, Rx, Ry) or (Rx, Ry)
    # ------------------------------------------------------------------
    def _curl(cx, cy):
        gxy = cx[..., 1:-1, 2:] - cx[..., 1:-1, :-2]
        gyx = cy[..., 2:, 1:-1] - cy[..., :-2, 1:-1]
        return torch.mean(torch.abs(gyx - gxy), dim=(-2, -1))

    def _div(cx, cy):
        gxx = cx[..., 2:, 1:-1] - cx[..., :-2, 1:-1]
        gyy = cy[..., 1:-1, 2:] - cy[..., 1:-1, :-2]
        return torch.mean(torch.abs(gxx + gyy), dim=(-2, -1))

    metric = _div if maximize_divergence else _curl

    def _note(message: str) -> None:
        """Report a choice this call made; silent unless ``verbose``."""
        if verbose:
            warnings.warn(message, UserWarning)

    def _override_note(value) -> str:
        return f"caller supplied {value}, so that step was skipped"

    def _objective_note(chosen_deg: float) -> str:
        objective = "largest |divergence|" if maximize_divergence else "smallest |curl|"
        return f"CoM field has {objective} at a rotation of {chosen_deg:.0f} deg."

    # ------------------------------------------------------------------
    if force_com_rotation is not None:
        _rotation_best_rad = float(np.deg2rad(force_com_rotation))
        _note(f"No angle sweep: {_override_note(f'{force_com_rotation:.0f} deg')}.")

        if force_com_transpose is not None:
            _rotation_best_transpose = force_com_transpose
            _note(f"No handedness test: {_override_note(force_com_transpose)}.")
        else:
            cos_r = float(np.cos(_rotation_best_rad))
            sin_r = float(np.sin(_rotation_best_rad))

            # untransposed
            cx = cos_r * _com_normalized_x - sin_r * _com_normalized_y
            cy = sin_r * _com_normalized_x + cos_r * _com_normalized_y
            val = metric(cx, cy)

            # transposed
            cx_t = cos_r * _com_normalized_y - sin_r * _com_normalized_x
            cy_t = sin_r * _com_normalized_y + cos_r * _com_normalized_x
            val_t = metric(cx_t, cy_t)

            if maximize_divergence:
                _rotation_best_transpose = bool(val_t > val)
            else:
                _rotation_best_transpose = bool(val_t < val)

            if _rotation_best_transpose:
                _note(_TRANSPOSE_ADVICE)

    else:
        # ---- rotation unknown ----
        rotation_angles_rad = torch.deg2rad(rotation_angles_deg)[:, None, None]

        if force_com_transpose is not None:
            _rotation_best_transpose = force_com_transpose
            _note(f"No handedness test: {_override_note(force_com_transpose)}.")

            if _rotation_best_transpose:
                cx = (
                    torch.cos(rotation_angles_rad) * _com_normalized_y[None]
                    - torch.sin(rotation_angles_rad) * _com_normalized_x[None]
                )
                cy = (
                    torch.sin(rotation_angles_rad) * _com_normalized_y[None]
                    + torch.cos(rotation_angles_rad) * _com_normalized_x[None]
                )
            else:
                cx = (
                    torch.cos(rotation_angles_rad) * _com_normalized_x[None]
                    - torch.sin(rotation_angles_rad) * _com_normalized_y[None]
                )
                cy = (
                    torch.sin(rotation_angles_rad) * _com_normalized_x[None]
                    + torch.cos(rotation_angles_rad) * _com_normalized_y[None]
                )

            vals = metric(cx, cy)  # (A,)

            if maximize_divergence:
                idx = int(torch.argmax(vals).item())
            else:
                idx = int(torch.argmin(vals).item())

            rotation_best_deg = float(rotation_angles_deg[idx].item())
            _rotation_best_rad = float(torch.deg2rad(rotation_angles_deg[idx]).item())

            _note(_objective_note(rotation_best_deg))

            if plot_rotation:
                _plot_rotation_curve(
                    rotation_angles_deg.cpu().numpy(),
                    vals.cpu().numpy(),
                    None,
                    rotation_best_deg,
                    _rotation_best_transpose,
                    maximize_divergence,
                    kwargs,
                )

        else:
            # ---- both unknown ----
            # untransposed
            cx = (
                torch.cos(rotation_angles_rad) * _com_normalized_x[None]
                - torch.sin(rotation_angles_rad) * _com_normalized_y[None]
            )
            cy = (
                torch.sin(rotation_angles_rad) * _com_normalized_x[None]
                + torch.cos(rotation_angles_rad) * _com_normalized_y[None]
            )
            vals = metric(cx, cy)

            # transposed
            cx_t = (
                torch.cos(rotation_angles_rad) * _com_normalized_y[None]
                - torch.sin(rotation_angles_rad) * _com_normalized_x[None]
            )
            cy_t = (
                torch.sin(rotation_angles_rad) * _com_normalized_y[None]
                + torch.cos(rotation_angles_rad) * _com_normalized_x[None]
            )
            vals_t = metric(cx_t, cy_t)

            if maximize_divergence:
                idx = int(torch.argmax(vals).item())
                idx_t = int(torch.argmax(vals_t).item())
                if vals[idx] >= vals_t[idx_t]:
                    rotation_best_deg = float(rotation_angles_deg[idx].item())
                    _rotation_best_rad = float(
                        torch.deg2rad(rotation_angles_deg[idx]).item()
                    )
                    _rotation_best_transpose = False
                else:
                    rotation_best_deg = float(rotation_angles_deg[idx_t].item())
                    _rotation_best_rad = float(
                        torch.deg2rad(rotation_angles_deg[idx_t]).item()
                    )
                    _rotation_best_transpose = True
            else:
                idx = int(torch.argmin(vals).item())
                idx_t = int(torch.argmin(vals_t).item())
                if vals[idx] <= vals_t[idx_t]:
                    rotation_best_deg = float(rotation_angles_deg[idx].item())
                    _rotation_best_rad = float(
                        torch.deg2rad(rotation_angles_deg[idx]).item()
                    )
                    _rotation_best_transpose = False
                else:
                    rotation_best_deg = float(rotation_angles_deg[idx_t].item())
                    _rotation_best_rad = float(
                        torch.deg2rad(rotation_angles_deg[idx_t]).item()
                    )
                    _rotation_best_transpose = True

            _note(_objective_note(rotation_best_deg))
            if _rotation_best_transpose:
                _note(_TRANSPOSE_ADVICE)

            if plot_rotation:
                _plot_rotation_curve(
                    rotation_angles_deg.cpu().numpy(),
                    vals.cpu().numpy(),
                    vals_t.cpu().numpy(),
                    rotation_best_deg,
                    _rotation_best_transpose,
                    maximize_divergence,
                    kwargs,
                )

    # ------------------------------------------------------------------
    # Corrected CoM
    # ------------------------------------------------------------------
    cos_r = float(np.cos(_rotation_best_rad))
    sin_r = float(np.sin(_rotation_best_rad))

    if _rotation_best_transpose:
        _com_x = cos_r * _com_normalized_y - sin_r * _com_normalized_x
        _com_y = sin_r * _com_normalized_y + cos_r * _com_normalized_x
    else:
        _com_x = cos_r * _com_normalized_x - sin_r * _com_normalized_y
        _com_y = sin_r * _com_normalized_x + cos_r * _com_normalized_y

    # ------------------------------------------------------------------
    # 180° disambiguation (optional)
    # ------------------------------------------------------------------
    # The curl-based rotation search has a 180° ambiguity in the *sign*
    # of the CoM gradient field.  When a flip is needed we negate the CoM
    # vectors but keep _rotation_best_rad unchanged, because:
    #   • The rotation is the *geometric* mapping between scan and
    #     detector coordinates — it does not depend on the sign of the
    #     phase gradient.
    #   • Adding π to the rotation would negate Q inside the direct-
    #     ptychography Warp kernel, flipping the SSB transfer function
    #     γ = A*(K)·A(K−Q) − A(K)·A*(K+Q)  and thereby inverting
    #     the reconstructed phase contrast.
    # ------------------------------------------------------------------
    if disambiguate_180:
        needs_flip = _disambiguate_rotation_180(_com_x, _com_y, verbose=verbose)
        if needs_flip:
            _com_x = -_com_x
            _com_y = -_com_y

    # Normalize angle to [-π, π)
    _rotation_best_rad = float(((_rotation_best_rad + np.pi) % (2 * np.pi)) - np.pi)

    if disambiguate_180 and needs_flip and verbose:
        warnings.warn(
            f"180° disambiguation applied — CoM vectors negated.  "
            f"Rotation unchanged = {np.rad2deg(_rotation_best_rad):.1f} degrees.",
            UserWarning,
        )

    # ------------------------------------------------------------------
    # Optionally plot CoM
    # ------------------------------------------------------------------
    if plot_center_of_mass == "all":
        _plot_com(
            [
                _com_measured_x, _com_measured_y,
                _com_normalized_x, _com_normalized_y,
                _com_x, _com_y,
            ],
            [
                "CoM_x", "CoM_y",
                "Normalized CoM_x", "Normalized CoM_y",
                "Corrected CoM_x", "Corrected CoM_y",
            ],
            nrows=3,
            scan_sampling=scan_sampling,
            scan_units=scan_units,
            shape=_com_measured_x.shape,
            **kwargs,
        )
    elif plot_center_of_mass == "default" or plot_center_of_mass is True:
        _plot_com(
            [_com_x, _com_y],
            ["Corrected CoM_x", "Corrected CoM_y"],
            nrows=1,
            scan_sampling=scan_sampling,
            scan_units=scan_units,
            shape=_com_x.shape,
            **kwargs,
        )

    return _rotation_best_rad, _rotation_best_transpose, _com_x, _com_y


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_rotation_curve(
    angles_deg,
    vals,
    vals_t,
    best_deg,
    best_transpose,
    maximize_divergence,
    extra_kwargs,
):
    figsize = extra_kwargs.get("figsize", (8, 2))
    fig, ax = plt.subplots(figsize=figsize)

    if vals_t is not None:
        ax.plot(angles_deg, vals, label="CoM")
        ax.plot(angles_deg, vals_t, label="CoM after transpose")
    else:
        label = "CoM after transpose" if best_transpose else "CoM"
        ax.plot(angles_deg, vals, label=label)

    yr = ax.get_ylim()
    ax.plot(np.ones(2) * best_deg, yr, color=(0, 0, 0, 1))
    ax.legend(loc="best")
    ax.set_xlabel("Rotation [degrees]")

    ylabel = "Mean Absolute Divergence" if maximize_divergence else "Mean Absolute Curl"
    ax.set_ylabel(ylabel)

    if vals_t is not None:
        ptp = max(np.ptp(vals), np.ptp(vals_t))
    else:
        ptp = np.ptp(vals)
    if ptp > 0:
        ax.set_aspect(np.ptp(angles_deg) / ptp / 4)
    fig.tight_layout()


def _plot_com(
    arrays,
    titles,
    nrows,
    scan_sampling,
    scan_units,
    shape,
    **kwargs,
):
    ncols = len(arrays) // nrows
    if nrows == 1:
        figsize = kwargs.pop("figsize", (8, 4))
    else:
        figsize = kwargs.pop("figsize", (8, 12))
    cmap = kwargs.pop("cmap", "RdBu_r")
    extent = [0, scan_sampling[1] * shape[1], scan_sampling[0] * shape[0], 0]

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=(0.25, 0.5))

    for ax, arr, title in zip(grid, arrays, titles):
        arr_np = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
        ax.imshow(arr_np, extent=extent, cmap=cmap, **kwargs)
        ax.set_ylabel(f"x [{scan_units[0]}]")
        ax.set_xlabel(f"y [{scan_units[1]}]")
        ax.set_title(title)
