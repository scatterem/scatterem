"""The image the sharpness autofocus scores, and its chunked evaluators.

``determine_aberrations`` builds a :class:`_TvFitContext` and asks
:func:`_build_tv_image_evaluator` (one candidate, differentiable) or
:func:`_build_tv_batched_evaluator` (a stack of candidates, forward only) for
the objective. The image is ``sum_k Im(ifft2(G_k corrected))`` over the
bright-field pixels ``k``, built chunk by chunk over ``k``, from ``G`` without
its scan DC ``G(K, 0)`` (:func:`_g_chunk_without_scan_dc`).
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from scatterem.nn.functional.ptychography import (
    build_aberration_bf_factor,
    build_aberration_bf_factor_planes,
    correct_aberrations_ksum,
    correct_aberrations_ksum_planes,
)
from scatterem.nn.regularizers.tv import total_variation_2d

from . import _config
from ._common import _chunk_ranges


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


def _image_sharpness(
    image: torch.Tensor, kind: str = "sparsity", *, stacked: bool = False
) -> torch.Tensor:
    """Differentiable image-sharpness metric (higher = sharper); maximised by the aberration fit.

    - ``"sparsity"`` (default): L4 / kurtosis-like ``mean(I^4) / mean(I^2)^2``. Rewards sparse,
      high-contrast atom-column spikes; aberration-induced ringing fills the background and
      *lowers* it, so the optimum sits at the true aberration. Empirically **unbiased for
      higher-order aberrations (coma, Cs) at adequate NA**, where total variation is biased (ringing
      inflates TV). Scale- and sign-invariant (even powers).
    - ``"tv"``: total variation (mean |gradient|). Legacy; gameable by high-frequency ringing, so its
      optimum is displaced for higher orders.

    Args:
        image: ``[..., H, W]``.
        kind: ``"sparsity"`` or ``"tv"``.
        stacked: reduce over the last TWO axes only, returning one metric per
            leading index, instead of over the whole tensor.  Used by the coarse
            multi-start sweep, which evaluates a STACK of candidate aberration
            vectors in one pass; a leading axis of length 1 is NOT the same
            reduction as no leading axis at all, so this is opt-in.
    """
    dims = (-2, -1) if stacked else None

    def _mean(t):
        return t.mean(dim=dims) if dims is not None else t.mean()

    if kind == "sparsity":
        i2 = _mean(image**2)
        i4 = _mean(image**4)
        return i4 / (i2 * i2 + 1e-12)
    if kind == "tv":
        return total_variation_2d(image, per_item=stacked)
    raise ValueError(f"sharpness_metric must be 'sparsity' or 'tv', got {kind!r}")


def _g_chunk_without_scan_dc(vBF, s: int, e: int) -> torch.Tensor:
    """``vBF.get_G_chunk(s, e)`` with each pixel's scan DC ``G(K, 0)`` set to 0.

    The image is linear in ``G``, so this is the autofocus image with the scan
    DC dropped, as in every acBF image. Upsampled, the tile reads
    ``G(K, 0)`` on every scan-lattice frequency, and kept, it put a grid that
    oscillates with C10 into the objective. At an upsampling of 1 its one
    frequency, ``Q = 0``, has weight 0, so the image does not change. With the
    earlier surrogate adjoint the gradient did: the backward differentiated an
    unnormalised surrogate, which gave that zero weight a nonzero derivative
    times the scan DC (on a planted truth d/dC10 -0.139 with it, -0.0068
    without, +0.0003 by a finite difference of the sharpness). The exact
    adjoint gives ``Q = 0`` no derivative and +0.00026 there, with or without
    the scan DC. The chunk is owned (``get_G_chunk``), so the write reaches no
    other reader.
    """
    chunk = vBF.get_G_chunk(s, e)
    chunk[0, 0, :] = 0
    return chunk


def _fetch_g_chunk(cache, vBF, s, e, upsample_int):
    """One ``G`` chunk, from the fit cache if it is populated and re-fetched if not.

    Both fitting loops need the identical sequence -- cache lookup, deterministic
    re-fetch on a miss, then tiling when the image is upsampled -- and carried
    two byte-identical copies of it.
    """
    Gc = cache.get((s, e))
    if Gc is None:
        Gc = _g_chunk_without_scan_dc(vBF, s, e)
    if upsample_int[0] > 1 or upsample_int[1] > 1:
        Gc = torch.tile(Gc, (int(upsample_int[0]), int(upsample_int[1]), 1))
    return Gc


def _prepare_tv_chunk_cache(
    vBF,
    upsample_int: tuple[int, int] | np.ndarray,
    n_batches: int,
    n_k_tv: int,
) -> tuple[int, dict[tuple[int, int], torch.Tensor]]:
    """Decide the autofocus stage's chunk count and pre-fetch its ``G`` cache
    (once, shared by ``evaluate_tv`` and ``evaluate_tv_stack``), budgeted
    below ``_TV_CHUNK_CACHE_BUDGET_BYTES`` and never for streaming
    residencies (``vbf_gpu`` / ``cpu_pinned``), excluded by name."""
    _cache_bytes = int(n_k_tv) * int(vBF.array.shape[0]) * int(vBF.array.shape[1]) * 8
    # The chunk COUNT the TV loop actually runs. ``_TV_CHUNK_MIN_BYTES`` names
    # the same quantity ``n_batches`` does, in the unit that makes it
    # comparable across configurations; ``min`` means this can only ever
    # coarsen a split that was finer than the budget requires. The tiling
    # factor is included because at ``upsample > 1`` the chunk is tiled to
    # ``U^2`` its size before anything reads it.
    _chunk_working_bytes = _cache_bytes * int(upsample_int[0]) * int(upsample_int[1])
    n_batches_tv = min(
        int(n_batches),
        max(1, math.ceil(_chunk_working_bytes / _config._TV_CHUNK_MIN_BYTES)),
    )
    g_chunks: dict[tuple[int, int], torch.Tensor] = {}
    if (
        _cache_bytes <= _config._TV_CHUNK_CACHE_BUDGET_BYTES
        and str(getattr(vBF, "_g_residency", "gpu")) == "gpu"
    ):
        for _s, _e in _chunk_ranges(n_k_tv, n_batches_tv):
            g_chunks[(_s, _e)] = _g_chunk_without_scan_dc(vBF, _s, _e)
    return n_batches_tv, g_chunks


@dataclass
class _TvFitContext:
    """Config invariant across chunks/iterations, shared by the autofocus
    evaluators and their per-chunk correct-and-reduce step."""

    vBF: object
    semiconvergence_angle: float
    wavelength: float
    Kx: torch.Tensor
    Ky: torch.Tensor
    Qx: torch.Tensor
    Qy: torch.Tensor
    rotation_t: torch.Tensor
    factor: float
    n_k_tv: int
    n_batches_tv: int
    upsample_int: tuple[int, int] | np.ndarray
    new_shape: tuple[int, int]
    roi_center: tuple[int, int]
    roi_shape: tuple[int, int]
    sharpness_metric: str
    n_grad_coeffs: int
    g_chunks: dict[tuple[int, int], torch.Tensor]


def _tv_correct_and_reduce_chunk(
    ctx: _TvFitContext, aberr: torch.Tensor, s: int, e: int, A_bf: torch.Tensor
) -> torch.Tensor:
    """Correct one ``[Ny, Nx, n_k]`` ``G`` chunk by ``aberr`` and reduce it to
    its real-space image contribution: ``sum_k Im(ifft2(G_k))``.

    The chunk fetch lives INSIDE this function so a gradient-checkpoint
    recompute (see ``_build_tv_image_evaluator``) re-fetches the identical
    chunk deterministically. FUSED correct-and-reduce: replaces
    ``correct_aberrations(...)`` + ``.sum(dim=-1)`` with the ``ik``
    accumulation happening in a register -- a reassociation, not a rewrite,
    of the SAME class/SIZE as the sum-first collapse below (rel 2.73e-07,
    float32's own error floor).
    """
    Gc = _fetch_g_chunk(ctx.g_chunks, ctx.vBF, s, e, ctx.upsample_int)
    Gprime_summed = correct_aberrations_ksum(
        Gc,
        aberr,
        ctx.rotation_t,
        ctx.semiconvergence_angle,
        ctx.wavelength,
        ctx.Qx,
        ctx.Qy,
        ctx.Kx[s:e],
        ctx.Ky[s:e],
        ctx.n_grad_coeffs,
        A_bf[s:e],
    )
    # Reduce the bright-field-pixel axis BEFORE the transform, not after.
    # ifft2 acts on the image axes (0, 1) and Im() is R-linear, so
    #     sum_k Im(ifft2(G_k)) == Im(ifft2(sum_k G_k))
    # and this chunk needs ONE two-dimensional transform instead of one per
    # bright-field pixel. NOT opt-in (unlike the reconstruction loop's
    # ``enable_chunk_sum_first``): this is the objective of a nonlinear TV
    # minimisation whose own reproducibility floor is far LARGER than this
    # reassociation (rel 2.6-2.7e-07 either way, float32's own error floor;
    # fitted coefficients move no more across this change than between two
    # runs of the unmodified code).
    return torch.fft.ifft2(Gprime_summed, dim=(0, 1), norm="ortho").imag * ctx.factor


def _build_tv_image_evaluator(ctx: _TvFitContext):
    """Build the differentiable per-candidate sharpness objective the LBFGS
    solve uses, plus the mutable ``last_image`` box the post-fit recompute
    writes into. Chunks over the detector-pixel axis (caps peak memory at
    one chunk); above ``_TV_CHUNK_CKPT_BUDGET_BYTES`` also checkpointed."""
    last_image: dict[str, torch.Tensor | None] = {"img": None}

    def evaluate_tv(aberr):
        from torch.utils.checkpoint import checkpoint

        # The ``ik``-only factor ``A = aperture(K) * exp(-i chi(K))`` is a
        # function of ``aberr`` alone, so it is built ONCE here and sliced
        # per chunk -- never inside ``_tv_correct_and_reduce_chunk`` (a per-
        # chunk rebuild would be a LOSS: 1.29x hoisting win at the published
        # Fig1 chunk, bit-exact). Safe under the checkpoint below:
        # ``build_aberration_bf_factor`` is ``@torch.no_grad()`` and passed
        # as an explicit ``checkpoint`` argument, so the recomputed call
        # binds the identical object the first forward used.
        A_bf = build_aberration_bf_factor(
            aberr.detach(),
            float(ctx.semiconvergence_angle),
            float(ctx.wavelength),
            ctx.Kx,
            ctx.Ky,
        )

        image_sum: torch.Tensor | None = None
        # Checkpoint only when the activation it drops is worth a second
        # forward pass: what it retains is one upsampled copy of ``vBF.G``
        # (the corrected G is never materialised, the ``ik`` sum happens
        # inside the kernel); what it COSTS is a whole extra forward, which
        # is why ``loss.backward`` measures 2.27x the forward it
        # differentiates. Below the budget, skip it.
        retained_bytes = (
            int(ctx.n_k_tv) * int(ctx.new_shape[0]) * int(ctx.new_shape[1]) * 8
        )
        use_ckpt = (
            torch.is_grad_enabled()
            and aberr.requires_grad
            and retained_bytes > _config._TV_CHUNK_CKPT_BUDGET_BYTES
        )
        for s, e in _chunk_ranges(ctx.n_k_tv, ctx.n_batches_tv):
            if use_ckpt:
                partial = cast(
                    torch.Tensor,
                    checkpoint(
                        _tv_correct_and_reduce_chunk,
                        ctx,
                        aberr,
                        s,
                        e,
                        A_bf,
                        use_reentrant=False,
                    ),
                )
            else:
                partial = _tv_correct_and_reduce_chunk(ctx, aberr, s, e, A_bf)
            image_sum = partial if image_sum is None else image_sum + partial
        if image_sum is None:
            raise ValueError("the virtual bright field has no bright-field pixels")
        image = image_sum / ctx.n_k_tv
        last_image["img"] = image
        roi = _windowed_roi(image, ctx.roi_center, ctx.roi_shape)
        return _image_sharpness(roi, ctx.sharpness_metric)

    return evaluate_tv, last_image


def _build_tv_batched_evaluator(ctx: _TvFitContext):
    """Build ``evaluate_tv`` for a STACK of ``[P, n_ab]`` candidate vectors,
    in ONE pass over the chunks.

    The coarse multi-start sweep in ``_optimize_aberrations_tv`` evaluates
    ``sweep_steps`` mutually-independent candidates per free coefficient, and
    is the bulk of the fit (measured 75.5% of ``determine_aberrations`` on
    the published Fig2_carbon config); serially each candidate re-issues a
    fixed ~110 us per-call HOST cost that does not shrink with problem size.
    Uses the same batched ops (and the same bit-identical-plane-by-plane
    argument) as a former correction
    sweep (replaced by the exact region sum, 2026-10-15). NOT claimed
    bit-exact: the stacked ``_image_sharpness`` reduction is a different
    block decomposition, but it only feeds a strict ``>`` comparison between
    distinct candidates, and the committed sweep point is verified identical.

    FORWARD-ONLY and ``no_grad``; does NOT write ``last_image`` (the caller
    re-evaluates at the chosen vector through ``evaluate_tv`` anyway).
    """

    def evaluate_tv_stack(aberr_stack):
        with torch.no_grad():
            aberr_stack = aberr_stack.detach().contiguous()
            n_planes = int(aberr_stack.shape[0])
            # Plane axis INNERMOST, so the per-chunk slice below is
            # contiguous for free -- ``correct_aberrations_ksum_planes``
            # documents ``A`` as ``[Nk, n_planes, 2]``.
            A_stack = (
                build_aberration_bf_factor_planes(
                    aberr_stack,
                    float(ctx.semiconvergence_angle),
                    float(ctx.wavelength),
                    ctx.Kx,
                    ctx.Ky,
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            image_sum: torch.Tensor | None = None
            for s, e in _chunk_ranges(ctx.n_k_tv, ctx.n_batches_tv):
                Gc = _fetch_g_chunk(ctx.g_chunks, ctx.vBF, s, e, ctx.upsample_int)
                summed = torch.empty(
                    (n_planes, int(Gc.shape[0]), int(Gc.shape[1])),
                    dtype=Gc.dtype,
                    device=Gc.device,
                )
                correct_aberrations_ksum_planes(
                    Gc,
                    aberr_stack,
                    ctx.rotation_t,
                    ctx.semiconvergence_angle,
                    ctx.wavelength,
                    ctx.Qx,
                    ctx.Qy,
                    ctx.Kx[s:e],
                    ctx.Ky[s:e],
                    A_stack[s:e],
                    summed,
                    # Explicit ``False``: the candidates carry whatever
                    # higher orders the sweep has already committed, so the
                    # condition cannot be established here, and ``None``
                    # would buy one device sync per call to ask.  This is
                    # also the branch the serial ``correct_aberrations_ksum``
                    # takes, which is what keeps the two paths comparable.
                    defocus_only=False,
                )
                partial = (
                    torch.fft.ifft2(summed, dim=(1, 2), norm="ortho").imag * ctx.factor
                )
                image_sum = partial if image_sum is None else image_sum + partial
            if image_sum is None:
                raise ValueError("the virtual bright field has no bright-field pixels")
            images = image_sum / ctx.n_k_tv
            rois = _windowed_roi(images, ctx.roi_center, ctx.roi_shape)
            return _image_sharpness(rois, ctx.sharpness_metric, stacked=True)

    return evaluate_tv_stack
