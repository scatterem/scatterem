"""The ``ik``-split plan of the zero-aberration k-SUM forward.

:func:`.ksum._ksum_fwd_body` may split the ``ik`` axis of
``_direct_ptychography_forward_ksum_chi_zero`` over a leading tile axis
(``_direct_ptychography_forward_ksum_chi_zero_tiled`` plus a ``torch.sum`` over
``dim=0``), which reassociates the float32 ``ik`` sum -- 5.3e-07 relative to peak
on ``r15_direct_ptychography/medium/direct_ptycho_sumfirst``, where it is
**1.2589x on the row**.  The tiled body then takes the squared-radius aperture
(BIT-IDENTICAL to the tiled body, 1.1957x more on the row) whenever its threshold
exists; its kill switch ``SCATTEREM_KSUM_SQ_APERTURE`` was deleted.

Only the zero-aberration body is split.  The general body has the identical
defect, and the identical fix on it measured 4.468x on the leg, but
1.0042x (not disjoint) on the only row that can be timed; it is deliberately left
serial until a row exists that would cash it.
"""

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_forward_ksum,
    _direct_ptychography_forward_ksum_chi_zero,
    _direct_ptychography_forward_ksum_chi_zero_tiled,
    _direct_ptychography_forward_ksum_chi_zero_tiled_sq,
)

from ._sq_aperture import _pctf_sq_threshold

#: Tiles the ``ik`` axis is split over.  16 is where BOTH measured curves
#: saturate (on the chi-zero kernel and on the general sibling) -- a
#: saturation point, not a fitted optimum.
_KSUM_TILES_MAX = 16

#: Below this many ``(iqy, iqx, ik)`` triples the split is refused: it costs one
#: extra reduction over ``NT`` copies of the output, roughly fixed, where the
#: saving is proportional to the serial kernel's cost.  Placed an order of
#: magnitude under the smallest configuration measured to gain (9216 x 205 =
#: 1.9 M triples at 3.43x).
_KSUM_TILE_MIN_TRIPLES = 1 << 17

#: ``[tiled, serial]`` -- so the ``ik``-split gate can be AUDITED as reached or
#: refused rather than asserted.  One integer store per call.
_KSUM_TILE_STATS = [0, 0]


def _ksum_tiles(n_thread: int, n_k: int, is_cuda: bool) -> int:
    """How many ``ik`` tiles to split the zero-aberration k-SUM kernel over.

    Returns 1 for "do not split", which selects the shipped serial kernel.
    Refused off CUDA (Warp runs a single-threaded scalar loop on the host, so
    splitting only adds a reduction), with fewer than two bright-field pixels,
    and under the work floor.
    """
    if not is_cuda or n_k < 2 or n_thread * n_k < _KSUM_TILE_MIN_TRIPLES:
        return 1
    return min(_KSUM_TILES_MAX, n_k)


def _ksum_plan(dim: tuple[int, int], n_k: int, is_cuda: bool, chi_is_zero: bool):
    """``(n_tile, sq)`` for one forward call; counts it in ``_KSUM_TILE_STATS``.

    ``sq`` is only the REQUEST for the squared aperture; the threshold itself is
    derived on the record path (:func:`_ksum_kernel`), because it is a pure
    function of ``wavelength``, ``semiconvergence_angle`` and the device, all
    already in the ``(key, scalars)`` pair a replay must match.
    """
    n_tile = _ksum_tiles(dim[0] * dim[1], n_k, is_cuda) if chi_is_zero else 1
    _KSUM_TILE_STATS[0 if n_tile > 1 else 1] += 1
    return n_tile, n_tile > 1


def _ksum_kernel(
    n_tile: int,
    chi_is_zero: bool,
    sq: bool,
    wavelength: float,
    semiconvergence_angle: float,
    device,
):
    """``(kernel, tenth argument)`` for the k-SUM forward's RECORD path.

    The tenth argument is an angle for the original kernels and a squared-radius
    threshold for the ``_sq`` body, so the two are chosen together.  A ``None``
    threshold (degenerate geometry) keeps the original tiled kernel under the same
    key, so the decision replays consistently.
    """
    if n_tile > 1:
        if sq:
            s_thresh = _pctf_sq_threshold(
                float(wavelength), float(semiconvergence_angle), device
            )
            if s_thresh is not None:
                return _direct_ptychography_forward_ksum_chi_zero_tiled_sq, s_thresh
        kern = _direct_ptychography_forward_ksum_chi_zero_tiled
    elif chi_is_zero:
        kern = _direct_ptychography_forward_ksum_chi_zero
    else:
        kern = _direct_ptychography_forward_ksum
    return kern, float(semiconvergence_angle)
