"""The chunk COUNT ``_iter_chunk_images`` actually runs.

``n_batches`` is a count, not a memory unit.  Two of the loop's routes coarsen
an over-split request; both are COARSEN-ONLY (``min``) and both run only when
the coarsened chunk is still ``_g_chunk_kouter_view``'s free read-only view.
"""

import math

from . import _config


def _sumfirst_count(vBF, upsample_int):
    """The chunk-sum-first branch's BF-pixel cap, or ``None`` when it does not apply.

    On the chunk-sum-first branch ``n_batches`` buys NOTHING: the chunk is taken as
    ``_g_chunk_kouter_view``'s free read-only view (no ownership copy) and
    ``correct_aberrations_ksum`` reduces over ``ik`` in a register, returning
    [Nqy, Nqx] whatever the chunk holds -- so no allocation in the loop body grows
    with chunk width, and the count is pure host dispatch.  MEASURED at the
    registered medium geometry (717 BF px, 245^2): own peak 1.37 MiB at
    n_batches=8 and 1.15 MiB at n_batches=1, i.e. coarsening REDUCES peak, while
    wall goes 2.221 -> 1.961 ms (and 5.426 ms at the library-default
    n_batches=25, which is 2.8x the collapsed point).

    This is the ``_TV_CHUNK_MIN_BYTES`` argument for the OTHER chunk loop, but
    it must NOT be spelled as a byte budget here, because the quantity that
    bounds this one is not memory -- nothing is allocated -- it is the length of
    the float32 accumulation chain the register reduction runs.  Hence a cap on
    BF pixels reduced per chunk: at the 717 that the registered row collapses to
    one chunk, the returned phase moves 5.321e-07 peak-normalised against this
    variant's 1e-5 absolute gate (19x headroom, and the gate is absolute BY
    DESIGN for exactly this reassociation -- see the per-row tolerance
    overrides); 1024 is 1.43x that measured point.

    Two further conditions, each a measurement rather than a precaution.  (b) No
    upsampling tile: at ``upsample_int > 1`` the body materialises
    ``torch.tile(Gc, (uy, ux, 1))``, which DOES scale with chunk width, so the
    "nothing grows" premise fails.  (c) The coarsened view is actually free:
    ``_g_chunk_kouter_view`` returns None for an alignment overlay or a provider
    with no resident G, and the body then falls back to an OWNED
    ``get_G_chunk`` -- at the coarse width that would be the blowup.  Probed at
    the width about to be used; a slice + permute + ``is_contiguous``, so no copy
    and no launch.
    """
    if bool((upsample_int > 1).any()):
        return None
    n_bf_total = int(vBF.n_bright_field)
    return max(1, math.ceil(n_bf_total / _config._SUMFIRST_CHUNK_MAX_BF))


def _kouter_sum_count(vBF, Qy, Qx):
    """The k-outer ``reduce="sum"`` route's byte floor, or ``None``.

    A BF-pixel cap was rejected on this route, measured at ``r15/medium``
    (245^2, ~41 MiB of corrected chunk per 90 BF px), where coarsening the COUNT
    is an 8x memory blowup for 1.037x.  Stated in bytes, the floor is inert
    there (8 chunks of ~41 MiB are already above it) and at the 1024^2
    ``r14/gd2o3`` scan; it bites only where ``n_batches`` was never a memory
    budget: ``r14/carbon`` (96^2, 1785 BF px) splits ~126 MiB into 25 chunks of
    ~5 MiB, so the call is host-dispatch bound (7.85 ms of host issue for 2.16 ms
    of device work) and the floor coarsens it to 8 -- 1.83x on
    ``r14_ffstem/carbon/ptycho``.  The peak it can add is bounded by
    ``_KOUTER_SUM_CHUNK_MIN_BYTES`` (own peak 8.3 -> 24.9 MiB there, a labelled
    memory trade).  Working bytes = the complex64 corrected chunk at the
    upsampled grid, the buffer that scales with width.

    NOT bit-exact: the caller sums per-chunk images, so the count reassociates a
    float32 sum (<= 2.3e-07 peak-normalised at carbon, against the oracle's
    5.9e-06 absolute gate).  ``reduce="none"`` never reaches this: it feeds the
    bit-exact empirical-SSNR digest and keeps its split.
    """
    n_bf_total = int(vBF.n_bright_field)
    working = n_bf_total * int(Qy.shape[0]) * int(Qx.shape[-1]) * 8
    return max(1, math.ceil(working / _config._KOUTER_SUM_CHUNK_MIN_BYTES))


def _effective_n_batches(
    vBF, n_batches, reduce, use_kouter, upsample_int, Qy, Qx, kouter_view
):
    """``n_batches``, coarsened where a chunk count buys nothing but host dispatch.

    ``kouter_view`` is ``_g_chunk_kouter_view``.  ``n_batches is None`` is legal
    and means one chunk (``_chunk_ranges``), already the coarsest split, so it is
    returned unchanged.  COARSEN-ONLY: a caller who asks for a split COARSER than
    the cap keeps it -- nothing here is ever made finer than requested, which is
    the property that matters, since a finer split is what costs host dispatch.
    """
    if n_batches is None:
        return n_batches
    if not use_kouter:
        n_coarse = _sumfirst_count(vBF, upsample_int)
    elif reduce == "sum":
        n_coarse = _kouter_sum_count(vBF, Qy, Qx)
    else:
        n_coarse = None
    if n_coarse is None or n_coarse >= int(n_batches):
        return n_batches
    n_bf_total = int(vBF.n_bright_field)
    probe_e = max(1, int(math.ceil(n_bf_total / n_coarse)))
    if kouter_view(vBF, 0, min(probe_e, n_bf_total)) is None:
        return n_batches
    return n_coarse
