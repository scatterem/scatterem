"""The two-for-one packed k-outer correction.

``_direct_ptychography_kouter``'s ``reduce="sum"`` body reads the inverse
transform of the corrected chunk only through ``.imag``.  The half store
(:mod:`.kouter_half`) uses this with a C2R, but at an odd ``W`` (the 245-grid
rows) cuFFT's C2R is no cheaper than its C2C (1.04x).  This store packs
two planes into one complex plane instead::

    Z_j = H_{2j} + i H_{2j+1},   H_k(q) = -i (C_k(q) - conj C_k(-q)) / 2

``H_k`` is Hermitian, so ``ifft2(Z_j) = Im(ifft2(C_{2j})) + i Im(ifft2(C_{2j+1}))``
and a C2C over ``ceil(Nk/2)`` planes carries the whole chunk.  ``Z`` is
word-for-word the ``Z`` spelled in torch from the full-grid kernel's output.  The
image is NOT bit-exact against the full-grid C2C route (the packed transform
rounds differently); we measured 2.716e-07 against a complex128 reference
on ``r15_direct_ptychography/medium``.

Gated on eager + complex64 CUDA + integer Q-tiles + the squared-aperture
threshold existing.  The ``kouter_packed`` backend row asks for it only where
the ``kouter_half`` row declined.
"""

import torch

from ._caches import _contiguous_memo, _sin_cos_rot
from ._sq_aperture import HALF_PACKED, _pctf_sq_threshold
from .kouter import _correct_aberrations_kouter_impl


def kouter_packed_supported(
    Gprime: torch.Tensor,
    full_shape: tuple,
    semiconvergence_angle: float,
    wavelength: float,
) -> bool:
    """Whether :func:`correct_aberrations_kouter_packed` can serve this chunk.

    The conditions of :func:`.kouter_half.kouter_half_supported` without the
    even-``W`` one: the packed store is on the full grid, so any ``W`` works.

    Parameters
    ----------
    Gprime : torch.Tensor
        ``[Nk, Nqy, Nqx]`` complex chunk, as for ``correct_aberrations_kouter``.
    full_shape : tuple
        ``(Nk, Hq, W)`` of the corrected grid; ``Hq``/``W`` must be integer
        Q-tiles of ``Gprime``.
    semiconvergence_angle : float
        Semiconvergence angle.
    wavelength : float
        Wavelength.

    Returns
    -------
    bool
        ``True`` when nothing is being traced, the chunk is a 3-D complex64
        CUDA tensor, the shape conditions hold and the squared-aperture
        threshold is defined.
    """
    if torch.compiler.is_compiling():
        return False
    if Gprime.dtype != torch.complex64 or not Gprime.is_cuda or Gprime.ndim != 3:
        return False
    nk, hq, w = (int(v) for v in full_shape)
    if w < 1 or hq < 1 or nk < 1 or nk != int(Gprime.shape[0]):
        return False
    if hq % int(Gprime.shape[1]) != 0 or w % int(Gprime.shape[2]) != 0:
        return False
    thresh = _pctf_sq_threshold(
        float(wavelength), float(semiconvergence_angle), Gprime.device
    )
    return thresh is not None


@torch.no_grad()
def correct_aberrations_kouter_packed(
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
    """Write two planes of ``H`` per complex word onto the full grid.

    Writes ``Z_j = H_{2j} + i H_{2j+1}`` onto ``out = (ceil(Nk/2), Hq, W)``, so
    that ``ifft2(Z_j) = Im(ifft2(C_{2j})) + i Im(ifft2(C_{2j+1}))``, where ``C``
    is the chunk ``correct_aberrations_kouter`` would write.  For an odd ``Nk``
    the last word holds ``H_{Nk-1}`` alone.  The caller must have checked
    :func:`kouter_packed_supported`.

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
        Pre-allocated contiguous ``(ceil(Nk/2), Hq, W)`` complex64 tensor.

    Returns
    -------
    torch.Tensor
        ``out``.

    Raises
    ------
    ValueError
        If ``out`` is not contiguous or does not have ``ceil(Nk/2)`` planes.
    """
    out_real = torch.view_as_real(out)
    if not out_real.is_contiguous() or out.shape[0] != (Gprime.shape[0] + 1) // 2:
        raise ValueError("out must be a contiguous (ceil(Nk/2), Hq, W) complex tensor")
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
        HALF_PACKED,
    )
    return out
