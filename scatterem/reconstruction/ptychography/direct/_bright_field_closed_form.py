"""Closed-form low-order aberrations (C1, C12) and scan rotation from bright-field shifts.

Shared by the LBFGS fit in :mod:`.bright_field`, which uses it as its seed.
The fit also returns the covariance of its parameters.
"""

import math

import numpy as np
import torch

from scatterem.utils.spectral_mesh import dual_mesh


# A local 2x2 rotation keeps this module's imports small.
def _rotation_matrix_2x2(angle: torch.Tensor) -> torch.Tensor:
    """2x2 rotation matrix R(angle) for a scalar-tensor angle in radians."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _huber_weights(
    rn: torch.Tensor, huber_c: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Huber weights for per-pixel residual magnitudes, scale from their MAD."""
    delta = huber_c * (1.4826 * torch.median(rn) + 1e-12)
    inlier = rn <= delta
    return torch.where(inlier, torch.ones_like(rn), delta / (rn + 1e-30)), inlier


def _huber_irls(solve, per_point, n_iter: int, huber_c: float):
    """Huber-weighted IRLS on the per-pixel shift residual.

    ``solve(w)`` is the least-squares fit for per-pixel weights ``w`` (None:
    unweighted), ``per_point(fit)`` its per-pixel residual magnitude. Down-weights
    outlier shift vectors (e.g. cross-correlation mis-locks on a repetitive
    lattice). Returns the fit, the weighted RMS residual and the inlier fraction.
    """
    fit = solve(None)
    for _ in range(int(n_iter)):
        fit = solve(_huber_weights(per_point(fit), huber_c)[0])
    rn = per_point(fit)
    w, inlier = _huber_weights(rn, huber_c)
    residual = ((w * rn.square()).sum() / (w.sum() + 1e-30)).sqrt()
    return fit, residual, float(inlier.float().mean().item())


def _parameter_covariance(
    X: torch.Tensor, r: torch.Tensor, robust: bool, huber_c: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Covariance of a linear fit to 2-vector shifts, the shift covariance and dof.

    ``X`` is the design per pixel, shape ``(n, 2, p)``, and ``r`` the residual
    per pixel at the fit, shape ``(n, 2)``. The shift errors of different
    pixels are taken as independent, with one 2×2 covariance ``Σ`` for all
    pixels. The result is the sandwich ``A⁻¹ B A⁻¹`` with
    ``B = Σᵢ Xᵢᵀ Σ̂ Xᵢ``. Least squares has ``A = XᵀX`` and
    ``Σ̂ = rᵀr / (n - p/2)``. The Huber fit (loss on the residual norm) has
    the exact Hessian ``A = Σᵢ Xᵢᵀ Hᵢ Xᵢ``, ``Hᵢ = I`` for an inlier and
    ``wᵢ (I - r̂ᵢ r̂ᵢᵀ)`` for a pixel down-weighted to ``wᵢ``, and
    ``Σ̂ = ψᵀψ / (n - p/2)`` with ``ψᵢ = wᵢ rᵢ``. The MAD scale is held fixed.
    """
    n, _, p = X.shape
    eye = torch.eye(2, dtype=X.dtype, device=X.device)
    if robust:
        rn = r.norm(dim=1)
        w, inlier = _huber_weights(rn, huber_c)
        unit = r / (rn[:, None] + 1e-30)
        radial = (~inlier).to(X.dtype)[:, None, None] * unit[:, :, None] * unit[:, None]
        hessian = w[:, None, None] * (eye - radial)
        psi = w[:, None] * r
    else:
        hessian = eye.expand(n, 2, 2)
        psi = r
    shift_cov = psi.T @ psi / (n - p / 2)
    a_inv = torch.linalg.inv(torch.einsum("nji,njk,nkl->il", X, hessian, X))
    b = torch.einsum("nji,jk,nkl->il", X, shift_cov, X)
    return a_inv @ b @ a_inv, shift_cov, 2 * n - p


def _low_order_from_map(m: torch.Tensor, lam: float, flipped: bool) -> torch.Tensor:
    """(C1, C12a, C12b, rotation in degrees) from the map ``shift = M @ k``.

    ``m`` is ``M`` flattened row-major, shape ``(4,)``. ``M = P @ R(alpha)``
    with the canonical ``alpha`` (``C1 >= 0``), or the other branch
    ``(-P, alpha + 180 deg)`` when ``flipped``. The rotation is not wrapped,
    so the function is smooth for the Jacobian.
    """
    M = m.reshape(2, 2)
    alpha = torch.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    if flipped:
        alpha = alpha + math.pi
    P = M @ _rotation_matrix_2x2(alpha).mT
    return torch.stack(
        [
            (P[0, 0] + P[1, 1]) / (2 * lam),
            (P[1, 1] - P[0, 0]) / (2 * lam),
            (P[0, 1] + P[1, 0]) / (2 * lam),
            torch.rad2deg(alpha),
        ]
    )


def _low_order_covariance(
    K64, S64, Mt, lam: float, flipped: bool, robust: bool, huber_c: float
):
    """Covariance of the four low-order parameters, the shift covariance and dof.

    The linear fit ``S = K @ Mt`` gives the covariance of ``vec(M)`` (row-major,
    ``M = Mt.T``); the Jacobian of :func:`_low_order_from_map` maps it to the
    parameters (delta method).
    """
    zeros = torch.zeros_like(K64)
    X = torch.stack(
        [torch.cat([K64, zeros], dim=1), torch.cat([zeros, K64], dim=1)], dim=1
    )  # (n, 2, 4): shift_i = X_i @ vec(M)
    cov_map, shift_cov, dof = _parameter_covariance(X, K64 @ Mt - S64, robust, huber_c)
    jac = torch.func.jacrev(_low_order_from_map)(Mt.mT.reshape(4), lam, flipped)
    return jac @ cov_map @ jac.mT, shift_cov, dof


def _fit_linear_map(K64, S64, robust: bool, robust_iter: int, huber_c: float):
    """``S = K @ Mt`` by least squares, or by Huber IRLS when ``robust``."""

    def solve(w):
        sw = 1.0 if w is None else w.sqrt()[:, None]
        return torch.linalg.lstsq(K64 * sw, S64 * sw).solution

    if robust:
        return _huber_irls(
            solve, lambda Mt: (K64 @ Mt - S64).norm(dim=1), robust_iter, huber_c
        )
    Mt = solve(None)
    return Mt, (K64 @ Mt - S64).square().mean().sqrt(), 1.0


def _low_order_at_fixed_rotation(
    K64: torch.Tensor,
    S64: torch.Tensor,
    wavelength: float,
    rotation_deg: torch.Tensor | float,
    well_posed: bool,
    robust: bool,
    robust_iter: int,
    huber_c: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(C1, C12a, C12b) by least squares at a given scan rotation, in degrees.

    ``shift = P @ R(rotation) @ k`` with ``P = lambda [[C1 - C12a, C12b], [C12b, C1 + C12a]]``
    is linear in the three coefficients once the rotation is known.
    """
    deg = (
        float(rotation_deg.item())
        if torch.is_tensor(rotation_deg)
        else float(rotation_deg)
    )
    R = _rotation_matrix_2x2(
        torch.tensor(math.radians(deg), dtype=torch.float64, device=K64.device)
    )
    u = K64 @ R.mT  # R @ k for every bright-field pixel
    A = wavelength * torch.cat(
        [
            torch.stack([u[:, 0], -u[:, 0], u[:, 1]], dim=1),
            torch.stack([u[:, 1], u[:, 1], u[:, 0]], dim=1),
        ]
    )
    b = torch.cat([S64[:, 0], S64[:, 1]])[:, None]
    n = K64.shape[0]

    def solve(w):
        sw = 1.0 if w is None else w.sqrt().repeat(2)[:, None]
        return torch.linalg.lstsq(A * sw, b * sw).solution

    def per_point(c):
        r = (A @ c - b)[:, 0]
        return torch.hypot(r[:n], r[n:])

    if robust:
        c, residual, inlier_frac = _huber_irls(solve, per_point, robust_iter, huber_c)
    else:
        c, inlier_frac = solve(None), 1.0
        residual = (A @ c - b).square().mean().sqrt()
    X = torch.stack([A[:n], A[n:]], dim=1)  # (n, 2, 3)
    covariance, shift_cov, dof = _parameter_covariance(
        X, (A @ c - b)[:, 0].reshape(2, n).T, robust, huber_c
    )
    aberrations = torch.zeros(12, device=K64.device, dtype=torch.float32)
    aberrations[:3] = c[:, 0].to(torch.float32)
    rotation = torch.tensor(
        [((deg + 180.0) % 360.0) - 180.0], dtype=torch.float32, device=K64.device
    )
    info = {
        "residual": float(residual.item()),
        "well_posed": well_posed,
        "branch_flipped": False,
        "inlier_frac": inlier_frac,
        **_covariance_info(covariance, shift_cov, dof, robust, ("C1", "C12a", "C12b")),
    }
    return aberrations, rotation, info


def _covariance_info(
    covariance: torch.Tensor,
    shift_cov: torch.Tensor,
    dof: int,
    robust: bool,
    parameters: tuple[str, ...],
) -> dict:
    """The covariance entries of the fit's ``info``."""
    return {
        "parameters": parameters,
        "covariance": covariance,
        "covariance_method": "huber_sandwich" if robust else "lsq",
        "shift_covariance": shift_cov,
        "dof": dof,
    }


def low_order_to_detector_frame(
    aberrations: torch.Tensor,
    rotation_deg: torch.Tensor | float,
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a closed-form fit's aberrations and covariance into the detector frame.

    The detector frame turns the A1 pair ``C12a + i C12b`` by
    ``exp(+2i rot)``, as ``scatterem.utils.transfer.shift_model_to_detector_frame``
    does; C1 and the rotation do not change. The covariance goes through the
    Jacobian of that map. With a free rotation it includes the rotation
    column ``2 J₊ C12_det π/180`` (``J₊`` the 90° rotation), so the A1 block
    and the A1-rotation terms both change.

    Parameters
    ----------
    aberrations : torch.Tensor
        Length-12 shift-model vector from
        :func:`fit_low_order_aberrations_and_rotation_closed_form`; only
        C1, C12a, C12b may be nonzero.
    rotation_deg : torch.Tensor or float
        The fitted (or held) rotation, in degrees.
    covariance : torch.Tensor
        ``info["covariance"]`` of the fit: `(4, 4)` for (C1, C12a, C12b,
        rotation) or `(3, 3)` for (C1, C12a, C12b).

    Returns
    -------
    aberrations : torch.Tensor
        The detector-frame vector, same dtype and device as the input.
    covariance : torch.Tensor
        The detector-frame covariance, same shape, float64.
    """
    theta = math.radians(float(rotation_deg))
    c, s = math.cos(2 * theta), math.sin(2 * theta)
    a1 = aberrations[1:3].to(torch.float64)
    turn = torch.tensor([[c, -s], [s, c]], dtype=torch.float64, device=a1.device)
    a1_det = turn @ a1
    out = aberrations.clone()
    out[1:3] = a1_det.to(aberrations.dtype)
    p = covariance.shape[0]
    jac = torch.eye(p, dtype=torch.float64, device=covariance.device)
    jac[1:3, 1:3] = turn.to(jac.device)
    if p == 4:
        jac[1:3, 3] = (2 * math.pi / 180) * torch.stack([-a1_det[1], a1_det[0]])
    return out, jac @ covariance.to(torch.float64) @ jac.mT


def _pick_branch(alpha0, P0, rotation_prior, well_posed: bool):
    """Resolve (P, alpha) <-> (-P, alpha + 180 deg) with the rotation prior."""
    if rotation_prior is None or not well_posed:
        return alpha0, P0, False
    prior = (
        float(rotation_prior.item())
        if torch.is_tensor(rotation_prior)
        else float(rotation_prior)
    )
    rot0 = math.degrees(float(alpha0.item()))

    def _ang_dist(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    if _ang_dist(rot0 + 180.0, prior) < _ang_dist(rot0, prior):
        return alpha0 + math.pi, -P0, True
    return alpha0, P0, False


def fit_low_order_aberrations_and_rotation_closed_form(
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray | torch.Tensor,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    include_in_fit_mask: torch.Tensor | None = None,
    rotation_prior: torch.Tensor | float | None = None,
    well_posed_rtol: float = 1e-3,
    robust: bool = False,
    robust_iter: int = 5,
    huber_c: float = 1.345,
    rotation_fixed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Closed-form low-order aberration + scan-rotation estimate from bright-field shifts.

    Solves the linear map ``shift = M @ q`` over bright-field pixels, decomposes
    ``M = P @ R(alpha)`` in closed form, resolves the (P, alpha) <-> (-P, alpha +- 180 deg)
    degeneracy with ``rotation_prior``, and extracts (C1, C12a, C12b, rotation).
    With ``rotation_fixed`` the rotation is ``rotation_prior`` and (C1, C12a, C12b)
    are the least-squares solution at that rotation.

    Parameters
    ----------
    target_shifts : torch.Tensor
        Bright-field shifts in Angstrom, shape `(N, 2)`, row-major over
        `bright_field_mask`.
    k_sampling : numpy.ndarray
        Real-space sampling of the detector grid, ``dataset.dr`` (Angstrom,
        ``1 / (n dk)`` per axis for ``n`` detector pixels of reciprocal size
        ``dk = dataset.sampling[-2:]``, 1/Angstrom), shape `(2,)` -- NOT
        ``dk`` itself: the q-grid is ``fftfreq(shape, k_sampling)``, as in the
        shift model. Passing ``dk`` multiplies the fitted C1 and C12 by
        ``n dk**2`` on a square detector with square pixels (otherwise the
        q-grid is stretched unequally: no single factor undoes it, a spurious
        C12 appears, and with a real C12 the rotation moves too).
    wavelength : float
        Electron wavelength, in Angstrom.
    bright_field_mask : torch.Tensor
        Boolean mask of bright-field pixels, shape `(H, W)`.
    include_in_fit_mask : torch.Tensor, optional
        Boolean subset of BF pixels to include, shape `(N,)`. Default is
        None, which includes all BF pixels.
    rotation_prior : torch.Tensor or float, optional
        Scan rotation prior in degrees (e.g. `meta.rotation`); used only to
        pick the 180-degree branch. Default is None, which returns the
        canonical `C1 >= 0` branch.
    well_posed_rtol : float, optional
        Threshold on `2*lambda*|C1| / ||M||`; below this, rotation is
        unrecoverable (near-zero defocus) and `well_posed` is False.
        Default is 1e-3.
    robust : bool, optional
        If True, run Huber-weighted IRLS instead of a plain least-squares
        fit. Default is False.
    robust_iter : int, optional
        Number of IRLS iterations when `robust` is True. Default is 5.
    huber_c : float, optional
        Huber threshold multiplier on the residual MAD scale, used when
        `robust` is True. Default is 1.345.
    rotation_fixed : bool, optional
        If True, hold the rotation at `rotation_prior` (required) and solve
        only (C1, C12a, C12b). `well_posed` still reports whether the shifts
        alone would determine the rotation. Default is False.

    Returns
    -------
    aberrations : torch.Tensor
        Length-12 vector; only indices 0, 1, 2 (C1, C12a, C12b) are
        populated. In the SHIFT-MODEL frame of
        ``scatterem.utils.transfer.aberrations_to_image_shifts``, like
        :func:`.bright_field.fit_aberrations_and_rotation_to_bright_field_shifts`
        (convert with
        ``scatterem.utils.transfer.shift_model_to_detector_frame``).
    rotation_deg : torch.Tensor
        Fitted rotation, in degrees, shape `(1,)`.
    info : dict
        `residual` (RMS shift residual, Angstrom), `well_posed` (bool),
        `branch_flipped` (bool), `inlier_frac` (float), and the
        covariance of the fitted parameters:

        - `parameters`: their names, ``("C1", "C12a", "C12b", "rotation")``,
          or ``("C1", "C12a", "C12b")`` with `rotation_fixed`;
        - `covariance`: float64 tensor, shape `(p, p)`, in Angstrom² and
          degrees²: the predicted covariance of the estimate over repeated
          acquisitions, to first order. It is in the SHIFT-MODEL frame, like
          `aberrations`; :func:`low_order_to_detector_frame` turns both
          into the detector frame together;
        - `covariance_method`: ``"lsq"`` or ``"huber_sandwich"`` (when
          `robust`);
        - `shift_covariance`: float64 tensor `(2, 2)`, Angstrom²: the
          covariance of the shift errors per pixel, from the residuals
          (for Huber, of the down-weighted residuals ``w r``);
        - `dof`: residual degrees of freedom, ``2 N - p_linear``.

        The covariance is ``A⁻¹ B A⁻¹`` with ``B = Σᵢ Xᵢᵀ Σ̂ Xᵢ`` over the
        linear parameters (the map ``M``, or C1, C12a, C12b at a held
        rotation). Least squares has ``A = XᵀX``; Huber has the exact
        Hessian of its loss. With a free rotation, the delta method maps
        the covariance of ``M`` to the four parameters. The covariance
        assumes shift errors that are independent between pixels and share
        one 2×2 covariance, and a model with only C1, C12 and the rotation:
        higher-order aberrations enter `shift_covariance` as noise. With
        `rotation_fixed`, it is conditional on `rotation_prior`: an error in
        the held rotation is not in it. When `well_posed` is False and the
        rotation is free, the covariance is not meaningful.

    Raises
    ------
    ValueError
        If `rotation_fixed` is True and `rotation_prior` is None.
    """
    if rotation_fixed and rotation_prior is None:
        raise ValueError("rotation_fixed=True needs a rotation_prior")
    device = target_shifts.device
    shape = tuple(bright_field_mask.shape)
    if bright_field_mask.device != device:
        bright_field_mask = bright_field_mask.to(device)

    sampling = (
        k_sampling.tolist() if isinstance(k_sampling, torch.Tensor) else k_sampling
    )
    q = torch.fft.fftshift(
        dual_mesh(shape, sampling, staggered=False, device=device), dim=(-2, -1)
    )  # (2, H, W) = [k0, k1]
    K = q[:, bright_field_mask].T  # (N, 2)
    S = target_shifts
    if include_in_fit_mask is not None:
        # Accept a (N,) per-pixel mask or the existing (N, 2) per-component default; reduce to
        # per-pixel (a pixel is whole-in/whole-out in practice, so column 0 suffices).
        m = include_in_fit_mask.to(device).reshape(K.shape[0], -1)[:, 0].bool()
        K, S = K[m], S[m]

    # Solve in float64 for a robust 2x2 algebra, regardless of input dtype.
    K64, S64 = K.to(torch.float64), S.to(torch.float64)
    Mt, residual, inlier_frac = _fit_linear_map(K64, S64, robust, robust_iter, huber_c)
    M = Mt.mT  # so that shift = M @ k

    # Signed 2x2 polar: M = P @ R(alpha0), alpha0 canonical (forces trace(P0) >= 0 -> C1 >= 0).
    alpha0 = torch.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    P0 = M @ _rotation_matrix_2x2(alpha0).mT

    mag = torch.sqrt(
        (M[0, 0] + M[1, 1]) ** 2 + (M[1, 0] - M[0, 1]) ** 2
    )  # = 2*lam*|C1|
    well_posed = bool((mag / (M.norm() + 1e-30) > well_posed_rtol).item())
    if rotation_fixed and rotation_prior is not None:  # None was rejected above
        return _low_order_at_fixed_rotation(
            K64,
            S64,
            wavelength,
            rotation_prior,
            well_posed,
            robust,
            robust_iter,
            huber_c,
        )

    _, _, branch_flipped = _pick_branch(alpha0, P0, rotation_prior, well_posed)

    params = _low_order_from_map(M.reshape(4), wavelength, branch_flipped)
    aberrations = torch.zeros(12, device=device, dtype=torch.float32)
    aberrations[:3] = params[:3].to(torch.float32)  # C1, C12a, C12b
    rotation_deg = ((params[3] + 180.0) % 360.0) - 180.0

    covariance, shift_cov, dof = _low_order_covariance(
        K64, S64, Mt, wavelength, branch_flipped, robust, huber_c
    )
    info = {
        "residual": float(residual.item()),
        "well_posed": well_posed,
        "branch_flipped": branch_flipped,
        "inlier_frac": inlier_frac,
        **_covariance_info(
            covariance, shift_cov, dof, robust, ("C1", "C12a", "C12b", "rotation")
        ),
    }
    return aberrations, rotation_deg.reshape(1).to(torch.float32), info
