"""Block-bootstrap covariance of the closed-form bright-field fit.

The formal covariance of
:func:`._bright_field_closed_form.fit_low_order_aberrations_and_rotation_closed_form`
counts every bright-field pixel as an independent measurement. On dynamical
data the shift errors are smooth across neighbouring pixels, and the formal
covariance is then not calibrated (usually it is much too small). The block bootstrap here keeps that
correlation: it cuts the bright-field disk into square blocks of detector
pixels, draws blocks with replacement, fits again on every draw and takes the
covariance of the fitted parameters over the draws.

The estimate is valid when the block side is larger than the correlation
length of the shift errors and the disk holds many blocks (some tens). With
few blocks it is too small by about ``(n_blocks - 1) / n_blocks`` and has a
large scatter itself.
"""

from collections.abc import Callable

import numpy as np
import torch

from ._bright_field_closed_form import (
    _bright_field_frequencies,
    _fit_linear_map,
    _low_order_at_fixed_rotation,
    _low_order_from_map,
)

__all__ = [
    "block_bootstrap_covariance",
    "closed_form_bootstrap_covariance",
    "shift_fit_covariance",
]


def _block_members(pixel_index: np.ndarray, block_px: int) -> list[np.ndarray]:
    """Row indices of every block, for ``(n, 2)`` integer detector pixel indices."""
    keys = np.floor_divide(pixel_index, int(block_px))
    _, block_of = np.unique(keys, axis=0, return_inverse=True)
    block_of = block_of.reshape(-1)
    order = np.argsort(block_of, kind="stable")
    bounds = np.flatnonzero(np.diff(block_of[order])) + 1
    return np.split(order, bounds)


def block_bootstrap_covariance(
    fit: Callable[[torch.Tensor], torch.Tensor | None],
    pixel_index: np.ndarray,
    block_px: int,
    n_samples: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, int]:
    """Covariance of the parameters of ``fit`` by a block bootstrap over pixels.

    Parameters
    ----------
    fit : callable
        ``fit(rows)`` fits the model on the bright-field pixels ``rows`` (an
        int64 index tensor, repeats allowed) and returns the parameter
        vector, shape ``(p,)``, or None when the draw cannot fix the
        parameters. A draw that returns None is replaced by a new draw, up to
        ``10 * n_samples`` draws in total.
    pixel_index : numpy.ndarray
        ``(n, 2)`` integer detector pixel indices of the ``n`` pixels in the
        fit, in the row order that ``fit`` reads.
    block_px : int
        Side of a square block, in detector pixels (>= 1).
    n_samples : int, optional
        Number of bootstrap draws. Default is 200.
    seed : int, optional
        Seed of the block draws. Default is 0.

    Returns
    -------
    covariance : torch.Tensor
        ``(p, p)`` float64 covariance of the parameters over the draws.
    n_blocks : int
        Number of blocks in the disk.

    Raises
    ------
    ValueError
        If ``block_px`` < 1, ``n_samples`` < 2, the disk holds fewer than
        two blocks, or fewer than ``n_samples`` of the draws are usable.
    """
    if int(block_px) < 1:
        raise ValueError(f"block_px must be >= 1, got {block_px!r}")
    if int(n_samples) < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples!r}")
    blocks = _block_members(np.asarray(pixel_index), int(block_px))
    if len(blocks) < 2:
        raise ValueError(
            f"the bright-field disk holds {len(blocks)} block(s) of {block_px} px; "
            "use a smaller block_px"
        )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(10 * int(n_samples)):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        params = fit(torch.as_tensor(np.concatenate([blocks[i] for i in pick])))
        if params is not None:
            draws.append(params.to(torch.float64))
        if len(draws) == int(n_samples):
            break
    else:
        raise ValueError(
            f"only {len(draws)} of {10 * int(n_samples)} bootstrap draws fix the "
            f"parameters; use a smaller block_px or more bright-field pixels"
        )
    params = torch.stack(draws)
    centred = params - params.mean(dim=0)
    return centred.T @ centred / (params.shape[0] - 1), len(blocks)


def closed_form_bootstrap_covariance(
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray | torch.Tensor,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    info: dict,
    *,
    block_px: int,
    include_in_fit_mask: torch.Tensor | None = None,
    rotation_prior: torch.Tensor | float | None = None,
    robust: bool = False,
    robust_iter: int = 5,
    huber_c: float = 1.345,
    n_samples: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, int]:
    """Block-bootstrap covariance of a closed-form low-order fit.

    The empirical counterpart of ``info["covariance"]`` of
    :func:`._bright_field_closed_form.fit_low_order_aberrations_and_rotation_closed_form`.
    It repeats that fit on block-resampled bright-field pixels (see
    :func:`block_bootstrap_covariance`), so it keeps shift errors that are
    correlated between neighbouring pixels. The formal covariance does not.
    A draw whose frequencies do not span the plane cannot fix the model and
    is drawn again. With a free rotation, every draw's rotation is taken
    within 180 degrees of the rotation fitted on all pixels, so that the
    branch cut of the angle does not enter the covariance.

    Parameters
    ----------
    target_shifts, k_sampling, wavelength, bright_field_mask
        The inputs of the closed-form fit, unchanged.
    info : dict
        The ``info`` that the closed-form fit returned on these inputs. Its
        ``parameters`` select the free-rotation (4) or held-rotation (3)
        model, and ``branch_flipped`` the 180-degree branch.
    block_px : int
        Side of a square block, in detector pixels. Choose it larger than
        the correlation length of the shift errors.
    include_in_fit_mask, rotation_prior, robust, robust_iter, huber_c
        As given to the closed-form fit. A held rotation (3 parameters)
        needs ``rotation_prior``.
    n_samples : int, optional
        Number of bootstrap draws. Default is 200.
    seed : int, optional
        Seed of the block draws. Default is 0.

    Returns
    -------
    covariance : torch.Tensor
        ``(p, p)`` float64, Angstrom² and degrees², in the SHIFT-MODEL frame
        and parameter order of ``info["covariance"]``
        (:func:`._bright_field_closed_form.low_order_to_detector_frame` turns
        it into the detector frame).
    n_blocks : int
        Number of blocks in the disk.

    Raises
    ------
    ValueError
        If the model has a held rotation and ``rotation_prior`` is None, or
        as :func:`block_bootstrap_covariance`.
    """
    mask = bright_field_mask.to(target_shifts.device)
    K64 = _bright_field_frequencies(k_sampling, mask).to(torch.float64)
    S64 = target_shifts.to(torch.float64)
    pixels = torch.argwhere(mask)
    if include_in_fit_mask is not None:
        keep = include_in_fit_mask.to(mask.device).reshape(K64.shape[0], -1)[:, 0]
        K64, S64, pixels = K64[keep.bool()], S64[keep.bool()], pixels[keep.bool()]
    held = len(info["parameters"]) == 3
    if held and rotation_prior is None:
        raise ValueError("a held-rotation fit needs its rotation_prior")
    held_rotation = 0.0 if rotation_prior is None else rotation_prior

    def free_fit(k: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        mt = _fit_linear_map(k, s, robust, robust_iter, huber_c)[0]
        return _low_order_from_map(
            mt.mT.reshape(4), wavelength, bool(info["branch_flipped"])
        )

    # The free fit's rotation on all pixels; a held fit does not read it.
    centre = 0.0 if held else float(free_fit(K64, S64)[3])

    def refit(rows: torch.Tensor) -> torch.Tensor | None:
        rows = rows.to(K64.device)
        k, s = K64[rows], S64[rows]
        if int(torch.linalg.matrix_rank(k)) < 2:
            return None
        if held:
            out = _low_order_at_fixed_rotation(
                k, s, wavelength, held_rotation, True, robust, robust_iter, huber_c
            )
            return out[0][:3]
        params = free_fit(k, s)
        turn = torch.remainder(params[3] - centre + 180.0, 360.0) - 180.0
        return torch.cat([params[:3], (centre + turn)[None]])

    return block_bootstrap_covariance(
        refit, pixels.cpu().numpy(), block_px, n_samples, seed
    )


def shift_fit_covariance(
    seed_info: dict | None,
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray | torch.Tensor,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    include: torch.Tensor,
    rotation_prior: torch.Tensor,
    block_px: int | None,
) -> dict:
    """The covariance fields of a ``BrightFieldShiftFit``, from its closed-form seed.

    ``seed_info`` is the ``info`` of the closed-form seed, or None when the
    covariance does not apply to the refined fit (then the result is empty).
    With ``block_px`` the block-bootstrap covariance is added.
    """
    if seed_info is None:
        return {}
    out = {
        "covariance": seed_info["covariance"],
        "covariance_parameters": tuple(seed_info["parameters"]),
    }
    if block_px is not None:
        out["covariance_bootstrap"] = closed_form_bootstrap_covariance(
            target_shifts,
            k_sampling,
            wavelength,
            bright_field_mask,
            seed_info,
            block_px=int(block_px),
            include_in_fit_mask=include,
            rotation_prior=rotation_prior,
        )[0]
    return out
