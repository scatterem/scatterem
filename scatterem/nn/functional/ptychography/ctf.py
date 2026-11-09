"""The phase-contrast transfer function of the direct-ptychography forward."""

import weakref

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp.ptychography import (
    _phase_contrast_transfer_function_forward,
)
from scatterem.physics import _warp_interop as wi

from ._ctf_unnormalized import normalize_pctf, pctf_unnormalized


def _launch_live_window(
    pctf,
    pctf_wp,
    Qx,
    Qy,
    Qx_wp,
    Qy_wp,
    Kx_wp,
    Ky_wp,
    ab_wp,
    sin_rot,
    cos_rot,
    semiconvergence_angle,
    wavelength,
    n_k,
    n_rows,
    limit,
    device,
):
    """Launch only the Q rows and columns that can carry signal.

    EVERYTHING BEYOND THE INFORMATION LIMIT IS IDENTICALLY ZERO. ``gamma`` needs
    ``K`` and at least one of ``K +- Q`` inside the aperture, and
    ``|Q| <= |K| + |K -+ Q| < 2R`` follows, so at ``|Q| >= 2 alpha / lambda``
    every term vanishes for every ``K`` -- the same cutoff
    ``double_and_triple_overlap_areas`` applies to the SSNR's noise term.

    The bound is taken per axis, so the disc's corners still launch; what this
    drops is whole dead rows and columns, which is what an oversampled scan has
    most of. Measured against the full-grid launch (i.e. including the even-in-Q
    halving), Nk 1137-2925 on an H200: 1.9x at a 0.2 A pixel where the grid
    barely reaches the limit, 2.6x at 0.1 A, and 6.0x at 0.07 A where 91 % of the
    grid is dead. The window is two slices per axis, never an index tensor: the
    advanced-indexing gather/scatter that replaces cost ~1.2 ms at 256 samples,
    more than the launch it saved.
    """
    n_qx = int(Qx.shape[0])
    n_live_rows = min(_live_axis_runs(Qy, limit)[0], n_rows)
    x_pos, x_neg = _live_axis_runs(Qx, limit)
    n_live_cols = x_pos + x_neg
    if n_live_rows and n_live_cols:
        whole = n_live_rows == n_rows and n_live_cols == n_qx
        target = (
            pctf
            if whole
            else torch.zeros(
                (n_live_rows, n_live_cols), dtype=torch.float32, device=pctf.device
            )
        )
        wp.launch(
            kernel=_phase_contrast_transfer_function_forward,
            dim=(n_k, n_live_rows, n_live_cols),
            inputs=[
                # The kernel's ``G`` parameter is unindexed; Warp packs ``None``
                # as a null array, so nothing is allocated or read for it.
                None,
                Qx_wp
                if whole
                else wi.from_torch(
                    torch.cat((Qx[:x_pos], Qx[n_qx - x_neg :])).contiguous()
                ),
                Qy_wp if whole else wi.from_torch(Qy[:n_live_rows].contiguous()),
                Kx_wp,
                Ky_wp,
                ab_wp,
                sin_rot,
                cos_rot,
                semiconvergence_angle,
                wavelength,
            ],
            outputs=[pctf_wp if whole else wi.from_torch(target)],
            device=device,
        )
        if not whole:
            pctf[:n_live_rows, :x_pos] = target[:, :x_pos]
            if x_neg:
                pctf[:n_live_rows, n_qx - x_neg :] = target[:, x_pos:]


_Q_WINDOW_CACHE: dict = {}
_Q_WINDOW_CACHE_MAX = 16


def _live_axis_runs(Q, limit):
    """``(n_pos, n_neg)``: the two contiguous runs of ``|Q| < limit`` on an axis.

    An ``fftfreq`` axis rises from 0 to the positive Nyquist and then continues
    from the negative Nyquist back to 0, so ``|Q|`` is unimodal and the live set
    is ALWAYS two runs -- a prefix ``[0, n_pos)`` and a suffix
    ``[len - n_neg, len)``. Returning counts rather than an index tensor keeps
    every use a slice, which matters: the advanced-indexing gather/scatter it
    replaces cost ~1.2 ms at 256 samples, more than the launch it saved.

    Cached per (axis object, limit), because the count has to reach the host and
    that is a device sync (measured 1.7 ms at 256 samples, 2.1 ms at 512). The Q
    axes come from ``_UPSAMPLED_GRID_CACHE`` and are the same objects on every
    call of a sweep, so the sync is paid once per grid, not once per plane. The
    entry holds a weak reference to the axis it was measured from and is
    recomputed once that is collected, so a reused ``id()`` cannot hand back
    another tensor's runs.
    """
    key = (id(Q), float(limit))
    hit = _Q_WINDOW_CACHE.get(key)
    if hit is not None and hit[0]() is not None:
        return hit[1]
    live = Q.abs() < limit
    n = int(Q.shape[0])
    counts = (
        int(live[: n // 2 + 1].sum()),
        int(live[n // 2 + 1 :].sum()),
    )
    if len(_Q_WINDOW_CACHE) >= _Q_WINDOW_CACHE_MAX:
        _Q_WINDOW_CACHE.clear()
    _Q_WINDOW_CACHE[key] = (weakref.ref(Q), counts)
    return counts


def _is_fftfreq_axis(Q):
    """Whether ``Q`` is laid out like ``torch.fft.fftfreq`` output.

    The half-grid mirror and the live window both rely on it: the mirror needs
    ``Q[n - i] == -Q[i]`` and the window needs ``|Q|`` rising to Nyquist and then
    falling back. The public wrapper accepts ANY axes (the reference tests pass
    ``linspace``), and on those both shortcuts return wrong values silently, so
    anything else takes the full-grid launch. Cached per axis object like
    :func:`_live_axis_runs`, because the answer has to reach the host.
    """
    key = ("fftfreq", id(Q))
    hit = _Q_WINDOW_CACHE.get(key)
    if hit is not None and hit[0]() is not None:
        return hit[1]
    n = int(Q.shape[0])
    if n < 2:
        is_fftfreq = n == 0 or float(Q[0]) == 0.0
    else:
        expected = torch.fft.fftfreq(n, device=Q.device, dtype=torch.float64) * (
            n * Q[1].double()
        )
        is_fftfreq = (
            bool(torch.allclose(Q.double(), expected, rtol=1e-5, atol=0.0))
            and float(Q[1]) > 0.0
        )
    if len(_Q_WINDOW_CACHE) >= _Q_WINDOW_CACHE_MAX:
        _Q_WINDOW_CACHE.clear()
    _Q_WINDOW_CACHE[key] = (weakref.ref(Q), is_fftfreq)
    return is_fftfreq


def _mirror_even_q_rows(
    pctf,
    Qx,
    Qy,
    Kx_wp,
    Ky_wp,
    ab_wp,
    sin_rot,
    cos_rot,
    semiconvergence_angle,
    wavelength,
    n_k,
    device,
    limit,
):
    """Fill the Q rows the half launch skipped, from the ones it computed.

    THE PCTF IS EVEN IN Q. Under ``Q -> -Q`` the two side bands of the double
    overlap exchange roles, so the double-overlap magnitude at ``-Q`` equals the
    one at ``Q``; the scan rotation commutes with the negation and the argument
    does not depend on the aberrations, so it holds for ANY coefficient vector --
    coma and spherical included. Rows ``n_qy // 2 + 1 ..`` are therefore the mirror of rows
    ``1 ..`` and are filled by a copy instead of by ``n_k`` threads each.
    Measured 2.1-2.3x on the whole call at Nk 1137-2925 and Q grids 256-512.

    ONE COLUMN CANNOT BE COPIED. ``fftfreq`` puts ``-N/2`` on an even-length axis
    and ``+N/2`` nowhere, so column ``n_qx // 2`` has no ``-Q`` partner on the
    grid; copying into it is wrong by ~4 absolute at a 16x16 grid with spherical
    aberration and a 30 degree rotation. It is launched instead, over
    ``n_k * n_mirrored`` threads, i.e. ``1/n_qx`` of what the mirror saved. The
    Nyquist ROW needs no fix: it is index ``n_qy // 2``, which the half launch
    computes directly.
    """
    n_qy, n_qx = int(Qy.shape[0]), int(Qx.shape[0])
    half = n_qy // 2
    if n_qy - half - 1 <= 0:
        return
    # ``out[i, j] = out[n_qy - i, n_qx - j]``: flip both axes, then roll the column
    # axis by one so index 0 maps to itself rather than to ``n_qx - 1``.
    pctf[half + 1 :] = torch.roll(
        torch.flip(pctf[1 : n_qy - half], dims=(0, 1)), shifts=1, dims=1
    )
    jn = n_qx // 2
    if n_qx % 2 or abs(float(Qx[jn])) >= limit:
        # An odd axis has no Nyquist bin, and a Nyquist column beyond the
        # information limit is zero in both halves, so the copy is already right.
        return
    column = torch.zeros((n_qy - half - 1, 1), dtype=torch.float32, device=pctf.device)
    wp.launch(
        kernel=_phase_contrast_transfer_function_forward,
        dim=(n_k, column.shape[0], 1),
        inputs=[
            None,
            wi.from_torch(Qx[jn : jn + 1].contiguous()),
            wi.from_torch(Qy[half + 1 :].contiguous()),
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            wavelength,
        ],
        outputs=[wi.from_torch(column)],
        device=device,
    )
    pctf[half + 1 :, jn : jn + 1] = column


def _launch_and_mirror(
    pctf,
    pctf_wp,
    Qx,
    Qy,
    Qx_wp,
    Qy_wp,
    Kx_wp,
    Ky_wp,
    ab_wp,
    sin_rot,
    cos_rot,
    semiconvergence_angle,
    wavelength,
    n_k,
    n_rows,
    device,
):
    """Launch the live window of the first ``n_rows`` Q rows, then mirror.

    The two halves of the saving: ``_launch_live_window`` drops the rows and
    columns beyond the information limit, ``_mirror_even_q_rows`` drops the ones
    the even-in-Q symmetry already determines.
    """
    limit = 2.0 * semiconvergence_angle / wavelength
    _launch_live_window(
        pctf,
        pctf_wp,
        Qx,
        Qy,
        Qx_wp,
        Qy_wp,
        Kx_wp,
        Ky_wp,
        ab_wp,
        sin_rot,
        cos_rot,
        semiconvergence_angle,
        wavelength,
        n_k,
        n_rows,
        limit,
        device,
    )
    _mirror_even_q_rows(
        pctf,
        Qx,
        Qy,
        Kx_wp,
        Ky_wp,
        ab_wp,
        sin_rot,
        cos_rot,
        semiconvergence_angle,
        wavelength,
        n_k,
        device,
        limit,
    )


@torch.library.custom_op(
    "scatterem::phase_contrast_transfer_function_fwd",
    mutates_args=(),
)
def _phase_contrast_transfer_function_fwd(
    aberrations: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    sin_rot: float,
    cos_rot: float,
    semiconvergence_angle: float,
    wavelength: float,
) -> Tensor:
    """Compute the un-normalized phase contrast transfer function.

    The returned tensor has shape ``[Nqy, Nqx]`` (float32) and is the
    per-(Qy, Qx) sum of the double-overlap magnitude over ``ik``. The bright-field
    normalization ``2 * A.sum()`` is applied by the public wrapper so the
    op body stays pure Warp + a single output allocation.

    This op takes **no** ``G``: the kernel declares one but never indexes it
    (gamma is a function of Q, K, the aberrations, the aperture and the
    wavelength alone), so ``G`` only ever supplied the launch grid
    ``[Nqy, Nqx, Nk]`` and the output shape -- and both are already carried by
    the coordinate arrays as ``(len(Qy), len(Qx), len(Kx))``. Passing a
    ``G``-shaped array instead forced callers to materialize a ``U^2``-tiled
    copy of it (up to 7.7 GiB per call at the Fig1 FF-STEM config) that the
    device then never read.
    """
    device = wi.device_from_torch(Kx.device)
    n_qy, n_qx, n_k = int(Qy.shape[0]), int(Qx.shape[0]), int(Kx.shape[0])
    pctf = torch.zeros((n_qy, n_qx), dtype=torch.float32, device=Kx.device)
    Qx_wp = wi.from_torch(Qx)
    Qy_wp = wi.from_torch(Qy)
    Kx_wp = wi.from_torch(Kx)
    Ky_wp = wi.from_torch(Ky)
    # Warp launches on the data device; the aberrations vector frequently comes
    # from ``meta.aberrations.array`` on the CPU (metadata is device-agnostic).
    # Align it to the launch device here, at the single torch->Warp boundary, so
    # every caller (direct ptychography, SSNR/PCTF, gradients) is device-safe.
    aberrations = aberrations.to(wp.device_to_torch(device))
    ab_wp = wi.from_torch(aberrations)
    pctf_wp = wi.from_torch(pctf)
    # ``(n_k, n_qy, n_qx)``, NOT ``(n_qy, n_qx, n_k)``: Warp makes the LAST
    # launch axis the fastest-varying thread index, and the kernel accumulates
    # with ``wp.atomic_add(pctf, iqy, iqx, ...)``.  With ``n_k`` last, all 32
    # lanes of a warp shared one ``(iqy, iqx)`` and hit ONE address, serialising
    # the add 32 ways -- half the launch at the Fig1_Gd2O3 config.  With
    # ``n_qx`` last the lanes hit 32 consecutive words instead.  The kernel body
    # unpacks ``wp.tid()`` in this order and carries the measurement; the two
    # must change together.
    # Only the first ``n_qy // 2 + 1`` rows are launched; the rest are mirrored
    # afterwards, because the PCTF is even in Q (see ``_mirror_even_q_rows``).
    n_rows = n_qy // 2 + 1
    # EVERYTHING BEYOND THE INFORMATION LIMIT IS IDENTICALLY ZERO. ``gamma`` needs
    # ``K`` and at least one of ``K +- Q`` inside the aperture, and
    # ``|Q| <= |K| + |K -+ Q| < 2R`` follows, so at ``|Q| >= 2 alpha / lambda``
    # every term vanishes for every ``K`` -- the same cutoff
    # ``double_and_triple_overlap_areas`` applies to the SSNR's noise term. Rows
    # and columns entirely outside it are dropped from the launch; one gather in,
    # one scatter out. The bound is per axis, so the disc's corners still launch:
    # this is 1.0x when the grid barely reaches the limit (1 A scan step at
    # upsample 2-3), 1.1x at upsample 5, 2.8x at upsample 8, and 9x at a 0.35 A
    # step and upsample 5, where 97 % of the grid is dead.
    if not (_is_fftfreq_axis(Qx) and _is_fftfreq_axis(Qy)):
        # Both shortcuts assume the fftfreq layout; any other axes get every
        # (Q, K) pair launched, exactly as before they existed.
        wp.launch(
            kernel=_phase_contrast_transfer_function_forward,
            dim=(n_k, n_qy, n_qx),
            inputs=[None, Qx_wp, Qy_wp, Kx_wp, Ky_wp, ab_wp, sin_rot, cos_rot]
            + [semiconvergence_angle, wavelength],
            outputs=[pctf_wp],
            device=device,
        )
        return pctf
    _launch_and_mirror(
        pctf,
        pctf_wp,
        Qx,
        Qy,
        Qx_wp,
        Qy_wp,
        Kx_wp,
        Ky_wp,
        ab_wp,
        sin_rot,
        cos_rot,
        semiconvergence_angle,
        wavelength,
        n_k,
        n_rows,
        device,
    )
    return pctf


@_phase_contrast_transfer_function_fwd.register_fake
def _(
    aberrations,
    Qx,
    Qy,
    Kx,
    Ky,
    sin_rot,
    cos_rot,
    semiconvergence_angle,
    wavelength,
):
    return torch.empty(
        (Qy.shape[0], Qx.shape[0]),
        dtype=torch.float32,
        device=Kx.device,
    )


@torch.no_grad()
def phase_contrast_transfer_function(
    G: torch.Tensor | None,
    aberrations: torch.Tensor,
    rotation: float | torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
):
    """Compute the phase contrast transfer function.

    Parameters
    ----------
    G : torch.Tensor or None
        The ``[Nqy, Nqx, Nk]`` complex64 G tensor. **Its values are not read
        and it may be None.** The kernel's gamma depends only on Q, K, the
        aberrations, the aperture and the wavelength; G only ever supplied
        the launch grid, which is ``(len(Qy), len(Qx), len(Kx))``. It is
        kept in the signature for API compatibility and, when given, is
        checked against that shape.
    aberrations : torch.Tensor
        Aberrations array.
    rotation : float or torch.Tensor
        Rotation in degrees (0-d tensor also accepted). Passing a tensor
        here causes a graph-break under ``torch.compile(fullgraph=True)``
        because the wrapper calls ``rotation.item()`` to materialize a
        Python float for the ``math.sin/cos`` call; pass a Python ``float``
        if you intend to compile this function.
    semiconvergence_angle : float
        Semiconvergence angle.
    wavelength : float
        Electron wavelength.
    Qx : torch.Tensor
        Qx coordinates.
    Qy : torch.Tensor
        Qy coordinates.
    Kx : torch.Tensor
        Kx coordinates.
    Ky : torch.Tensor
        Ky coordinates.

    Returns
    -------
    torch.Tensor
        Phase contrast transfer function, float32, shape ``[Nqy, Nqx]``.
    """
    pctf = pctf_unnormalized(
        G, aberrations, rotation, semiconvergence_angle, wavelength, Qx, Qy, Kx, Ky
    )
    return normalize_pctf(pctf, Kx, Ky, semiconvergence_angle, wavelength)
