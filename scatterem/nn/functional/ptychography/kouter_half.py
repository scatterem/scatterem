"""The half-spectrum k-outer correction.

``_direct_ptychography_kouter``'s ``reduce="sum"`` body reads the inverse
transform of the corrected chunk ONLY through ``.imag``, and for ANY complex
``C`` on an ``(Hq, W)`` grid ``Im(ifft2(C)) = irfft2(H, s=(Hq, W))`` with
``H(q) = -i (C(q) - conj C(-q)) / 2`` Hermitian.  So the correction kernel can
store ``H`` on the half grid ``(Nk, Hq, W//2 + 1)`` and the consumer's C2C
transform becomes a C2R -- no physics premise, the real part is simply unread.
``H`` itself is word-for-word the one spelled in torch from the full-grid
kernel's output; the image differs from the C2C route by ~3 ulp (the C2R rounds
differently), 3.16e-07 against complex128 where the C2C route is 2.26e-07.

Gated on eager + complex64 CUDA + an EVEN corrected last dim + the
squared-aperture threshold existing (see :mod:`._sq_aperture`; its kill switch
``SCATTEREM_KOUTER_HALF_C2R`` was deleted).  At an odd ``W`` (the 245-grid
rows) the C2R buys ~1.04x and the shipped path stays.
"""

import torch

from ._caches import _contiguous_memo, _sin_cos_rot
from ._sq_aperture import _pctf_sq_threshold
from .kouter import _correct_aberrations_kouter_impl


def kouter_half_supported(
    Gprime: torch.Tensor,
    full_shape: tuple,
    semiconvergence_angle: float,
    wavelength: float,
) -> bool:
    """Whether :func:`correct_aberrations_kouter_half` can serve this chunk.

    Parameters
    ----------
    Gprime : torch.Tensor
        ``[Nk, Nqy, Nqx]`` complex chunk, as for ``correct_aberrations_kouter``.
    full_shape : tuple
        ``(Nk, Hq, W)`` of the corrected grid; ``Hq``/``W`` must be integer
        Q-tiles of ``Gprime`` and ``W`` must be even.
    semiconvergence_angle : float
        Semiconvergence angle.
    wavelength : float
        Wavelength.

    Returns
    -------
    bool
        ``True`` when nothing is being traced, the chunk is a 3-D complex64 CUDA tensor, the shape conditions hold and the
        squared-aperture threshold is defined.
    """
    if torch.compiler.is_compiling():
        return False
    if Gprime.dtype != torch.complex64 or not Gprime.is_cuda or Gprime.ndim != 3:
        return False
    nk, hq, w = (int(v) for v in full_shape)
    if w < 2 or w % 2 != 0 or hq < 1 or nk != int(Gprime.shape[0]):
        return False
    if hq % int(Gprime.shape[1]) != 0 or w % int(Gprime.shape[2]) != 0:
        return False
    thresh = _pctf_sq_threshold(
        float(wavelength), float(semiconvergence_angle), Gprime.device
    )
    return thresh is not None


@torch.no_grad()
def correct_aberrations_kouter_half(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    A: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``correct_aberrations_kouter`` for a consumer of ``ifft2(C).imag`` only.

    Writes ``H = -i (C(q) - conj C(-q)) / 2`` on the half grid
    ``out = (Nk, Hq, W//2 + 1)``, so that
    ``irfft2(out, s=(Hq, W)) == ifft2(C).imag``, where ``C`` is the chunk
    ``correct_aberrations_kouter`` would write on the full ``(Nk, Hq, W)`` grid.
    The caller must have checked :func:`kouter_half_supported`.

    Parameters
    ----------
    Gprime : torch.Tensor
        ``[Nk, Nqy, Nqx]`` complex64 chunk, left untouched.
    aberrations : torch.Tensor
        Aberration coefficients.
    rotation : float
        Scan/detector rotation in degrees (float or device tensor).
    semiconvergence_angle : float
        Semiconvergence angle.
    wavelength : float
        Wavelength.
    Qx, Qy, Kx, Ky : torch.Tensor
        Coordinate arrays, as for ``correct_aberrations_kouter``.
    A : torch.Tensor
        ``[Nk, 2]`` float32 factor, or an empty ``(0, 2)`` tensor to build it.
    out : torch.Tensor
        Pre-allocated contiguous ``(Nk, Hq, W//2 + 1)`` complex64 tensor.

    Returns
    -------
    torch.Tensor
        ``out``.
    """
    out_real = torch.view_as_real(out)
    if not out_real.is_contiguous() or out.shape[0] != Gprime.shape[0]:
        raise ValueError("out must be a contiguous (Nk, Hq, W//2 + 1) complex tensor")
    _correct_aberrations_kouter_impl(
        torch.view_as_real(Gprime).contiguous(),
        aberrations.contiguous(),
        _sin_cos_rot(rotation, Gprime.device),
        _contiguous_memo(Qx),
        _contiguous_memo(Qy),
        _contiguous_memo(Kx),
        _contiguous_memo(Ky),
        A.contiguous(),
        out_real,
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
        False,
        True,
    )
    return out
