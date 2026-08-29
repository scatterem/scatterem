"""
Warp kernels for direct ptychography operations.
"""
import warp as wp

from scatterem.utils.warp.aberrations import aberration_function_cartesian
from scatterem.utils.warp import (
    aperture,
    cabs,
    cconj,
    cexp,
    cmul,
)


@wp.func
def minus_i(z: wp.vec2) -> wp.vec2:
    """
    Multiplication of a complex number by -i.
    """
    # (-i)*(x+iy) =  y + i*(-x)
    return wp.vec2(z[1], -z[0])


@wp.func
def ip_real_conj(a: wp.vec2, b: wp.vec2) -> wp.float32:
    """
    Inner product of the real part of the conjugate of a and b.
    """
    # Re{ conj(a) * b } for (ax+i ay)(bx+i by) = ax*bx + ay*by
    return a[0] * b[0] + a[1] * b[1]


@wp.func
def dchi_cartesian_aberrations(
    qy: wp.float32, qx: wp.float32, wavelength: wp.float32, j: int
) -> wp.float32:
    """
    Derivative of the aberration array with respect to the qy and qx.
    Args:
        qy: wp.float32 - qy coordinate
        qx: wp.float32 - qx coordinate
        wavelength: wp.float32 - wavelength
        j: int - index of the aberration

    Returns:
        wp.float32 - derivative of the aberration array with respect to the qy and qx
    """

    u = qx * wavelength
    v = qy * wavelength
    u2 = u * u
    v2 = v * v
    u3 = u2 * u
    v3 = v2 * v
    u4 = u3 * u
    v4 = v3 * v
    base = wp.float32(0.0)

    if j == 0:
        base = 0.5 * (u2 + v2)
    elif j == 1:
        base = 0.5 * (u2 - v2)
    elif j == 2:
        base = u * v
    elif j == 3:
        base = (1.0 / 3.0) * (u3 + u * v2)
    elif j == 4:
        base = (1.0 / 3.0) * (v3 + u * u * v)
    elif j == 5:
        base = (1.0 / 3.0) * (u3 - 3.0 * u * v2)
    elif j == 6:
        base = (1.0 / 3.0) * (3.0 * u * u * v - v3)
    elif j == 7:
        base = 0.25 * (u4 + v4 + 2.0 * u2 * v2)
    elif j == 8:
        base = 0.25 * (u4 - v4)
    elif j == 9:
        base = 0.25 * (2.0 * u3 * v + 2.0 * u * v3)
    elif j == 10:
        base = 0.25 * (u4 - 6.0 * u2 * v2 + v4)
    elif j == 11:
        base = 0.25 * (4.0 * u3 * v - 4.0 * u * v3)

    return base * (2.0 * wp.pi / wavelength)


@wp.kernel
def _direct_ptychography_forward(
    G: wp.array(dtype=wp.vec2, ndim=3),
    # The four coordinate arrays carry an explicit ``dtype`` so this kernel is
    # NOT generic. A bare ``wp.array(ndim=1)`` has dtype ``Any``, which makes
    # Warp re-run ``infer_argument_types`` + ``add_overload`` on EVERY launch
    # (measured: 37.0 us of host time per launch against 15.9 us for the same
    # body typed) and makes it refuse ``array_t`` descriptors outright
    # ("Unable to infer the type of argument"), so the cheap
    # ``wp.from_torch(..., return_ctype=True)`` path was unavailable to the
    # caller. float32 is the only dtype that ever worked here: the body feeds
    # these values to float32-typed ``wp.func``s, so a float64 array failed at
    # Warp codegen before this annotation existed (verified, not assumed --
    # ``benchmarks/lab/_probe_r9f_exact.py`` check 3).
    Qx_all: wp.array(dtype=wp.float32, ndim=1),
    Qy_all: wp.array(dtype=wp.float32, ndim=1),
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    G_out: wp.array(dtype=wp.vec2, ndim=3),
) -> None:
    """
    Forward kernel for the direct ptychography forward pass.

    Args:
        G: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]
        Qx_all: wp.array(ndim=1) - [Qx]
        Qy_all: wp.array(ndim=1) - [Qy]
        Kx_all: wp.array(ndim=1) - [Kx]
        Ky_all: wp.array(ndim=1) - [Ky]
        aberrations: wp.array(dtype=wp.float32, ndim=1)
        sin_rot: wp.float32 - sin(rotation)
        cos_rot: wp.float32 - cos(rotation)
        semiconvergence_angle: wp.float32 - semiconvergence angle
        eps: wp.float32 - epsilon
        wavelength: wp.float32 - wavelength
        G_out: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]

    Returns:
        None (G_out is modified in place) - output is the corrected G
    """

    iqy, iqx, ik = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    # ``aperture()`` is a hard 0/1 step, so ``gamma_complex`` below is
    # IDENTICALLY zero unless ``a1 != 0 and (a2 != 0 or a3 != 0)``: ``a1 == 0``
    # makes ``A`` exactly ``(0,0)`` and kills both products, and ``a2 == a3 ==
    # 0`` makes ``A_plus`` and ``Am`` exactly ``(0,0)`` and kills both again.
    # ``a1 * (a2 + a3) > 0`` is precisely that union, and it is written as the
    # kernel's OWN aperture predicate rather than as a ``|Q| > 2 sin(a)/l``
    # radius test on purpose: the bright-field set is THRESHOLDED, not
    # apertured, so ``max|K|`` need not respect ``sin(a)/l`` and a radius test
    # is not exact on real data (R9c-pctf measured ``max|K| = 0.6262 >
    # sin(a)/l = 0.6164`` at Fig1/U=2).
    #
    # Hoisting the three apertures above the three aberration polynomials and
    # the three ``cexp`` skips them where they cannot matter.  Measured on the
    # captured published ``Fig1_Gd2O3`` fit chunk (512x512x43 complex64,
    # ``benchmarks/lab/_probe_d9_kernel.py``, arms alternated in one process):
    # **74.1 % of threads skippable**, kernel **0.619 -> 0.380 ms = 1.63x**,
    # 292 -> 475 GB/s against a 634 GB/s load+store ROOF -- i.e. this kernel was
    # 2.18x its own memory floor and is now 1.33x it.  The three ``aperture()``
    # calls are worth 0.008 ms of the 0.335 saved, so the guard is free.
    # It is 1.63x rather than a micro-optimisation because ``ik`` is the
    # fastest-varying launch axis: a warp is 32 consecutive bright-field pixels
    # at ONE ``(iqy, iqx)``, and outside the useful ``|Q|`` region the whole
    # warp takes the same branch instead of diverging.
    #
    # THE STORE STAYS UNCONDITIONAL, AND THAT IS LOAD-BEARING TWICE.  (1) The
    # caller allocates the destination with ``empty_like`` on the strength of
    # this kernel covering it exactly once (see the comment at
    # ``nn/functional/ptychography.py``); a guarded store would need
    # ``zeros_like`` back in the same commit, whose memset costs more than this
    # saves.  (2) Keeping the ``cmul`` preserves NaN/Inf propagation from ``G``
    # bit for bit (``0 * NaN`` is still NaN), which an explicit zero store would
    # silently change.
    #
    # Exactness, measured rather than argued: in the SKIPPED region the two
    # forms are BIT-IDENTICAL (0 of 16.7 M floats differ, max|diff| exactly
    # 0.0), because ``gamma_abs`` clamps to 1e-8 and ``gamma_phase`` is then
    # exactly ``(0,0)``.  In the LIVE region the arithmetic is unchanged but
    # ``ptxas`` contracts it differently inside a conditional (D6-run's
    # finding), which moves 4.2 % of the live floats by <= 2 ULP: max|diff|
    # 9.537e-07 on ``G'``, 1.366e-07 RELATIVE on the chunk image the caller
    # consumes -- half the perturbation of the sum-first collapse this same fit
    # already ships, and ~3 orders below the FF-STEM path's own run-to-run
    # floor.
    a1 = aperture(Ky, Kx, wavelength, semiconvergence_angle)
    a2 = aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle)
    a3 = aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle)

    gamma_conj = wp.vec2(0.0, 0.0)
    if a1 * (a2 + a3) > 0.0:
        chi1 = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
        apert1 = wp.vec2(a1, wp.float32(0.0))
        expichi1 = cexp(1.0, -chi1)
        A = cmul(apert1, expichi1)

        chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
        apert2 = wp.vec2(a2, wp.float32(0.0))
        expichi2 = cexp(1.0, -chi2)
        A_plus = cmul(apert2, expichi2)

        chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
        apert3 = wp.vec2(a3, wp.float32(0.0))
        expichi3 = cexp(1.0, -chi3)
        Am = cmul(apert3, expichi3)

        gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

        gamma_abs = cabs(gamma_complex)
        gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
        gamma_phase = wp.vec2(
            gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs
        )
        gamma_conj = cconj(gamma_phase)
    G_out[iqy, iqx, ik] = cmul(G[iqy, iqx, ik], gamma_conj)


@wp.kernel(enable_backward=False)
def _direct_ptychography_build_A(
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    A_out: wp.array(dtype=wp.vec2, ndim=1),
) -> None:
    """Build the ``ik``-only aperture-times-aberration factor ``A``.

    ``A[ik] = aperture(K) * exp(-i chi(K))`` depends only on the bright-field
    pixel, yet ``_direct_ptychography_forward`` recomputes it for every one of
    the ``Nqy * Nqx`` reciprocal-space pairs.  These six lines are copied
    VERBATIM from that kernel (lines marked ``chi1``/``apert1``/``expichi1``)
    so ``_direct_ptychography_forward_precomputed_A`` consumes exactly the
    float32 value the fused kernel would have computed in registers -- the
    hoist is a memoisation, not a reformulation, and is bit-exact.
    """
    ik = wp.tid()
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]
    chi1 = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
    apert1 = wp.vec2(
        aperture(Ky, Kx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi1 = cexp(1.0, -chi1)
    A_out[ik] = cmul(apert1, expichi1)


@wp.kernel(enable_backward=False)
def _direct_ptychography_build_A_planes(
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=2),  # [ip, n_ab]
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    A_out: wp.array(dtype=wp.vec2, ndim=2),  # [ip, ik]
) -> None:
    """:func:`_direct_ptychography_build_A` with a leading PLANE axis.

    ``direct_ptychography_depth_section`` builds this factor once per depth
    plane above its chunk loop, because ``A`` varies with the plane's defocus.
    The kernel is ``Nk`` threads -- 729 at the registered ``medium`` geometry --
    so its DEVICE half is ~2 us and every one of those calls is host cost:
    measured in situ at 108 us/call, of which ``wp.launch`` is 49.7, the four
    ``wp.from_torch`` views 30.1 and ``torch.empty`` 7.8
    (``benchmarks/lab/_probe_d346_split.py``).  Collapsing ``n_depths`` launches
    into one is therefore worth ``(n_depths - 1) x 108 us`` of host and nothing
    on the device -- and that, not the arithmetic, is the whole prize here.  An
    ablation that deletes six of the seven calls outright bounds the row at
    **1.0836x** (``_probe_d346_roof.py``).

    Same arithmetic, line for line, in the same order and the same float32
    precision as the per-plane kernel -- only the ``wp.tid()`` unpacking and two
    subscripts change.  Thread ``(ip, ik)`` reads ONE aberration row
    (``aberrations[ip]``, a 1-D sub-array view -- no copy, the same idiom
    :func:`_direct_ptychography_forward_ksum_planes` uses) and writes one output
    word, so ``A_out[ip]`` is **bit-identical** to what a per-plane launch writes
    into its own ``[Nk]`` buffer.  There is no reduction and no cross-thread
    interaction that could reassociate.

    ``ik`` is the fastest-varying launch axis, so the store stays unit-stride
    within a warp.  The output is ``[ip, ik]`` -- plane OUTERMOST -- which is the
    layout whose per-plane slice ``A_out[ip]`` is contiguous, i.e. the layout the
    per-plane consumers already expect.  The ``[ik, ip]`` packing the
    ``*_planes`` correction kernels want (whose CHUNK slice ``[s:e]`` is
    contiguous for free) is one transpose-copy away and the caller already paid
    exactly that copy as a ``torch.stack``.
    """
    ip, ik = wp.tid()
    ab = aberrations[ip]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]
    chi1 = aberration_function_cartesian(Ky, Kx, wavelength, ab)
    apert1 = wp.vec2(
        aperture(Ky, Kx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi1 = cexp(1.0, -chi1)
    A_out[ip, ik] = cmul(apert1, expichi1)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A(
    G: wp.array(dtype=wp.vec2, ndim=3),
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    A_all: wp.array(dtype=wp.vec2, ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    G_out: wp.array(dtype=wp.vec2, ndim=3),
) -> None:
    """``_direct_ptychography_forward`` with the ``ik``-only ``A`` read from
    ``A_all`` (built once by ``_direct_ptychography_build_A``) instead of
    recomputed per ``(iqy, iqx)``.

    Three aberration-polynomial evaluations per element become two, which is
    the whole speedup: the kernel is COMPUTE-bound (measured 267 GB/s against
    a 659 GB/s read+write floor on this access pattern), so removing a third
    of the transcendentals is a real saving rather than a memory trade.

    Kept as a separate kernel rather than folded into
    ``_direct_ptychography_forward`` on purpose: that kernel also backs the
    DIFFERENTIABLE ``scatterem::correct_aberrations_fwd`` op, whose
    hand-written analytic adjoint pairs with it, and this path is
    ``@torch.no_grad()``.

    NO LONGER in step with it line for line: D9-run added the
    ``a1 * (a2 + a3) > 0`` aperture early-out to ``_direct_ptychography_forward``
    only (1.63x on the captured published Fig1 fit chunk, 74.1 % of threads
    skippable, 292 -> 475 GB/s against a 634 GB/s ROOF), because that is the
    kernel the aberration fit runs and the one that was measured.  The same
    lever applies here, with TWO chi evaluations to skip instead of three and on
    the RECONSTRUCTION path's ``upsample``-dependent geometry -- a different skip
    fraction, hence a different measurement, hence not shipped unmeasured.  See
    ``benchmarks/lab/_probe_d9_kernel.py``, which re-derives that fraction from a
    captured call.
    """
    iqy, iqx, ik = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    A = A_all[ik]

    chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
    apert2 = wp.vec2(
        aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi2 = cexp(1.0, -chi2)
    A_plus = cmul(apert2, expichi2)

    chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
    apert3 = wp.vec2(
        aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi3 = cexp(1.0, -chi3)
    Am = cmul(apert3, expichi3)

    gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

    gamma_abs = cabs(gamma_complex)
    gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
    gamma_phase = wp.vec2(gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs)
    gamma_conj = cconj(gamma_phase)
    G_out[iqy, iqx, ik] = cmul(G[iqy, iqx, ik], gamma_conj)


@wp.kernel(enable_backward=False)
def _direct_ptychography_chi_is_zero(
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    flag_out: wp.array(dtype=wp.int32, ndim=1),
) -> None:
    """``flag_out[0] = 1`` iff every aberration coefficient is exactly ``+0.0``.

    A ONE-THREAD launch, and the point of it is that the answer lands in DEVICE
    memory.  ``_direct_ptychography_forward_precomputed_A_kouter`` needs the
    predicate, and the obvious alternative -- reading ``aberrations`` on the host
    -- is a ``cudaStreamSynchronize`` on this path.  Measured on
    ``r15_direct_ptychography/medium`` (``benchmarks/lab/_probe_d322_gate.py``):
    ``bool((ab != 0).any())`` costs 0.0245 ms on an empty queue and **1.0007 ms**
    behind eight chunk kernels, against the ~0.34 ms the predicate is worth.  D313
    removed a sync from this very row; this must not put one back.

    WHY ``+0.0`` AND NOT ``|c| == 0``.  What the caller needs is not "the
    coefficients are zero" but ``cexp(1.0, -chi) == (1.0, -0.0)`` BIT FOR BIT at
    every ``(K +- Q)``, and that follows from ``chi`` being exactly ``+0.0``:
    every term of :func:`aberration_function_cartesian` is
    ``coefficient * <finite monomial>``, so with all coefficients ``+0.0`` each
    term is ``+-0.0``; the sum STARTS at ``0.5 * c0 * r2`` with
    ``r2 = u*u + v*v >= +0.0``, which is ``+0.0`` and never ``-0.0``; and
    ``+0.0 + (-0.0) == +0.0`` in round-to-nearest, so the running sum stays
    ``+0.0`` however the later monomials are signed.  Then ``-chi`` is ``-0.0``,
    ``cos(-0.0) == 1.0`` and ``sin(-0.0) == -0.0``.

    A single ``-0.0`` coefficient breaks that chain (it can make ``chi`` come out
    ``-0.0``, whose exponential is ``(1.0, +0.0)``), and ``-0.0`` is reachable --
    a fit that returns ``-C1`` with ``C1 == 0.0`` writes one.  So it is excluded
    explicitly with ``1.0 / c < 0.0``, which is ``-inf`` for ``-0.0`` and ``+inf``
    for ``+0.0``; there is no trap on the device and the cost is one thread's
    twelve divisions.  NaN takes the general path for free, since ``c != 0.0``.
    """
    n_bad = wp.int32(0)
    for i in range(aberrations.shape[0]):
        c = aberrations[i]
        if c != 0.0:
            n_bad += 1
        else:
            if 1.0 / c < 0.0:
                n_bad += 1
    if n_bad == 0:
        flag_out[0] = 1
    else:
        flag_out[0] = 0


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_kouter(
    G: wp.array(dtype=wp.vec2, ndim=3),
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    A_all: wp.array(dtype=wp.vec2, ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    chi_is_zero: wp.array(dtype=wp.int32, ndim=1),
    G_out: wp.array(dtype=wp.vec2, ndim=3),
) -> None:
    """``_direct_ptychography_forward_precomputed_A`` on a ``[ik, iqy, iqx]``
    chunk instead of a ``[iqy, iqx, ik]`` one.

    Same arithmetic, in the same order and the same float32 precision.  The
    correction is a pure per-element multiply by a ``gamma_conj`` that depends on
    ``(iqy, iqx, ik)`` and on nothing else, so there is no reduction and no
    cross-thread interaction that could reassociate.

    NO LONGER line for line with that kernel, and the divergence is deliberate:
    this one carries D9's aperture early-out (D321) and D322's ``chi``-identity
    fold, and the twin carries neither.  Both are guards that skip work which is
    provably zero, so the OUTPUT is still bit-identical -- measured at 0 of
    86 075 850 words over the registered ``medium`` row, compared as int32 -- but
    the two bodies are no longer interchangeable as source and a fix to one does
    not reach the other.  See the comment on the guard below, and the ``NOTICED``
    list of ``docs/perf-lab/reports/2026-08-26-d322-chi-identity-fold.md``, which
    records both ports as un-taken.

    Exists because the CONSUMER wants the bright-field axis outermost.
    ``torch.fft.ifft2(G, dim=(0, 1))`` on a contiguous ``[Nqy, Nqx, Nk]`` chunk
    transforms the OUTER two axes, so each 1-D transform strides ``Nqx * Nk``
    elements; ``ifft2(dim=(-2, -1))`` on ``[Nk, Nqy, Nqx]`` is the natural
    batched form and measures **1.66x** faster at the registered chunk shapes,
    bit-for-bit identical (D49/D224).  Giving the correction this layout is what
    lets a caller pay the transposing copy ONCE and transform many times --
    see :func:`~scatterem.reconstruction.direct_ptychography.direct_ptychography_depth_section`,
    which corrects the same chunk under ``n_depths`` different defocus values.

    The launch grid is ``[Nk, Nqy, Nqx]``, so the fastest-varying thread index is
    ``iqx`` and both the load and the store are unit-stride within a warp -- the
    same coalescing the ``[iqy, iqx, ik]`` kernel gets from ``ik``.
    """
    ik, iqy, iqx = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    A = A_all[ik]

    # D9's aperture hoist, ported to the PRECOMPUTED-``A`` layout.  The sibling
    # kernel above (``_direct_ptychography_forward``) carries the derivation:
    # ``aperture()`` is a hard 0/1 step, so ``gamma_complex`` is IDENTICALLY zero
    # unless ``a1 != 0 and (a2 != 0 or a3 != 0)``, and skipping the two aberration
    # polynomials and the two ``cexp`` where they cannot matter is free.
    #
    # Here ``a1`` is not available: it has already been folded into the caller's
    # precomputed ``A = aperture(K) * exp(-i chi(K))``.  It is recovered EXACTLY
    # rather than re-derived -- ``a1 == 0`` makes ``A`` exactly ``(+-0, +-0)``
    # (``cmul((0,0), z)``), so ``A[0]^2 + A[1]^2 == 0`` iff ``a1 == 0``, and it is
    # nonzero otherwise since ``|exp(-i chi)|`` is ~1.  ``a1sq * (a2 + a3) > 0`` is
    # therefore the same predicate the sibling uses, spelled from what this kernel
    # has.  It is NOT a ``|Q|`` radius test, for the reason the sibling records:
    # the bright-field set is THRESHOLDED, not apertured, so ``max|K|`` need not
    # respect ``sin(a)/l``.
    #
    # THE STORE STAYS UNCONDITIONAL, for both of the sibling's reasons: the caller
    # allocates ``out_real`` with ``torch.empty`` on the strength of this kernel
    # covering it exactly once, and keeping the ``cmul`` preserves NaN/Inf
    # propagation from ``G`` bit for bit.
    #
    # Measured over the registered ``r15_direct_ptychography/medium`` row (eight
    # chunks, (90, 245, 245) x7 + (87, 245, 245)): **65.75 % of the 43.0 M threads
    # skippable**, counted on the device with this same predicate and the op's own
    # arguments (``benchmarks/lab/_probe_d321_skip.py``); the live fraction rises
    # monotonically with the chunk index, 25.4 % -> 39.1 %.  Kernel
    # **2.029 -> 1.416 ms** and the row **6.496 -> 5.886 ms = 1.104x**, five
    # interleaved arms in one process (``benchmarks/lab/_probe_d321_guard.py``);
    # **1.1078x**, 6/6 paired and disjoint, over six alternating separately-launched
    # arms (``_probe_d321_ab.py``), peak CUDA unchanged at 4409.4 MiB.
    #
    # BIT-EXACT, measured and not argued: ``torch.equal`` on the reconstructed phase
    # image (0 of 60025 words differ) and the ``empirical_phase_sha`` sha256 contract
    # on the ``reduce="none"`` consumer both hold, at both registered sizes and on
    # ``r20_depth_section/small`` (``stack_normed`` max rel 0.000e+00).  Note this is
    # STRONGER than the sibling manages -- D6 measured its version moving 4.2 % of the
    # LIVE floats by <= 2 ULP, because ``ptxas`` contracts the same arithmetic
    # differently inside a conditional.  Why this port escapes that is NOT
    # established here; only that it does, at these shapes, on this toolkit.  Treat a
    # re-spelling of the guarded block as needing the exactness check re-run.
    a2 = aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle)
    a3 = aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle)
    a1sq = A[0] * A[0] + A[1] * A[1]

    # THE IDENTITY FOLD, and it is where the remaining time on this kernel was.
    # A delete-one-piece ablation on the row's own captured arguments
    # (``benchmarks/lab/_probe_d322_ablate.py``, arms interleaved in one process)
    # decomposed the post-D321 kernel exactly:
    #
    #     G_out = G, nothing else   1.150 ms   558 GB/s   <- compulsory traffic
    #     shipped                   1.585 ms   405 GB/s
    #     the two chi polynomials removed
    #                               1.654 ms              <- FREE, saves NOTHING
    #     the two ``cexp`` removed  1.184 ms   542 GB/s   <- AT ROOF
    #
    # i.e. the whole 0.43 ms above the memory floor is the two transcendental
    # pairs and none of it is the polynomial that feeds them.  Two respellings
    # are CLOSED with numbers and must not be re-taken: a ``sincosf`` fused pair
    # via ``wp.func_native`` is BYTE-identical and 0.97x (nvrtc already shares the
    # argument reduction), and splitting the guard into independent ``a2 > 0`` /
    # ``a3 > 0`` tests -- exact, since a hard 0/1 aperture makes the other term
    # vanish -- is only 1.0992x here because 14.67 % of live threads are in the
    # double-overlap region and the other 85 % already take both branches.
    #
    # What is left is not to pay them.  ``chi`` is identically ``+0.0`` when every
    # aberration coefficient is ``+0.0`` (see
    # ``_direct_ptychography_chi_is_zero``, which proves it term by term), so both
    # exponentials are the compile-time constant ``(1.0, -0.0)``: ``cos(-0.0)`` is
    # ``1.0`` and ``sin(-0.0)`` is ``-0.0``.  The signed zero is written out
    # deliberately -- it is what the general body produces and it survives into
    # the output's BYTES.
    #
    # The predicate is read from a ONE-ELEMENT DEVICE array rather than taken as a
    # launch scalar, because deriving a host scalar from ``aberrations`` costs a
    # sync worth 1.0 ms on this row.  That choice is priced: a launch scalar
    # measures 1.3336x and the device array 1.3213x, against 1.3980x for a
    # separate kernel with only the identity body in it (both bodies share one
    # register budget here) and 1.2099x for deriving the predicate per thread.
    # Measured over the eight registered ``medium`` chunks, arms interleaved in
    # one process (``benchmarks/lab/_probe_d322_arms.py``).
    #
    # BIT-EXACT, measured and not argued: 0 of 86 075 850 output words differ from
    # the shipped body over the whole row, compared as INT32 so that a difference
    # in the sign of a zero could not hide -- which is the failure mode this fold
    # actually has, and why the constant is spelled ``-0.0``.
    gamma_conj = wp.vec2(0.0, 0.0)
    if a1sq * (a2 + a3) > 0.0:
        if chi_is_zero[0] != 0:
            unit = wp.vec2(wp.float32(1.0), -wp.float32(0.0))
            A_plus = cmul(wp.vec2(a2, wp.float32(0.0)), unit)
            Am = cmul(wp.vec2(a3, wp.float32(0.0)), unit)
        else:
            chi2 = aberration_function_cartesian(
                Ky + Qy, Kx + Qx, wavelength, aberrations
            )
            apert2 = wp.vec2(a2, wp.float32(0.0))
            expichi2 = cexp(1.0, -chi2)
            A_plus = cmul(apert2, expichi2)

            chi3 = aberration_function_cartesian(
                Ky - Qy, Kx - Qx, wavelength, aberrations
            )
            apert3 = wp.vec2(a3, wp.float32(0.0))
            expichi3 = cexp(1.0, -chi3)
            Am = cmul(apert3, expichi3)

        gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

        gamma_abs = cabs(gamma_complex)
        gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
        gamma_phase = wp.vec2(
            gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs
        )
        gamma_conj = cconj(gamma_phase)
    G_out[ik, iqy, iqx] = cmul(G[ik, iqy, iqx], gamma_conj)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_precomputed_A_planes(
    G: wp.array(dtype=wp.vec2, ndim=3),
    # Explicit ``dtype`` for the same reason ``_direct_ptychography_forward``
    # carries one (see its own comment): a bare ``wp.array(ndim=1)`` has dtype
    # ``Any``, so Warp re-runs ``infer_argument_types`` + ``add_overload`` on
    # EVERY launch and refuses ``array_t`` descriptors outright ("Unable to infer
    # the type of argument"), which put the cheap
    # ``wp.from_torch(..., return_ctype=True)`` path out of the caller's reach.
    # float32 is the only dtype that ever worked here -- the body feeds these
    # values to float32-typed ``wp.func``s, so anything else failed at Warp
    # codegen before this annotation existed.  The generated CUDA for the float32
    # instantiation is unchanged, which is what keeps the correction bit-exact.
    Qx_all: wp.array(dtype=wp.float32, ndim=1),
    Qy_all: wp.array(dtype=wp.float32, ndim=1),
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    A_all: wp.array(dtype=wp.vec2, ndim=2),
    aberrations: wp.array(dtype=wp.float32, ndim=2),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    G_out: wp.array(dtype=wp.vec2, ndim=4),
) -> None:
    """``_direct_ptychography_forward_precomputed_A_kouter`` with a leading PLANE
    axis: one launch corrects the same ``[Nk, Nqy, Nqx]`` chunk under
    ``n_planes`` different aberration vectors.

    Same arithmetic, line for line, in the same order and the same float32
    precision as the ``kouter`` kernel -- only the ``wp.tid()`` unpacking and the
    subscripts change.  Every thread reads ONE aberration row (``aberrations[ip]``,
    a 1-D sub-array view -- no copy) and ONE ``A`` entry, and writes one output
    word, so there is no reduction and no cross-thread interaction that could
    reassociate; output word ``[ip, ik, iqy, iqx]`` is **bit-identical** to what a
    per-plane ``kouter`` launch writes at ``[ik, iqy, iqx]``.

    Exists because the per-CALL host cost of this correction does not shrink with
    problem size.  D274-run measured ``wp.launch`` at a fixed 32-124 us of host
    time per call and D275 split ``r20_depth_section/small``'s per-call cost into
    66 us of ``torch.library.custom_op`` dispatch plus ~54 us of Warp launch
    machinery; that row issues the correction ``n_chunks * n_depths`` = 40 times
    inside an 18 ms wall that is 91 % GPU-idle.  Folding the plane axis into the
    launch grid makes it ``n_chunks`` = 8, which is the only way to remove the
    dispatch half without removing the custom op itself (which exists to be opaque
    to Dynamo).  Measured 2.83x on that row, bit-exact.

    ``A_all`` is indexed ``[ik, ip]``, NOT ``[ip, ik]``: the caller holds one
    ``[n_bf_total, n_planes]`` factor built above the chunk loop, and that layout
    is the one whose chunk slice ``[s:e]`` is contiguous for free.

    The launch grid is ``[n_planes, Nk, Nqy, Nqx]``, so ``iqx`` is still the
    fastest-varying thread index and both the load and the store stay unit-stride
    within a warp.
    """
    ip, ik, iqy, iqx = wp.tid()

    ab = aberrations[ip]

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    A = A_all[ik, ip]

    chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, ab)
    apert2 = wp.vec2(
        aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi2 = cexp(1.0, -chi2)
    A_plus = cmul(apert2, expichi2)

    chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, ab)
    apert3 = wp.vec2(
        aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi3 = cexp(1.0, -chi3)
    Am = cmul(apert3, expichi3)

    gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

    gamma_abs = cabs(gamma_complex)
    gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
    gamma_phase = wp.vec2(gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs)
    gamma_conj = cconj(gamma_phase)
    G_out[ip, ik, iqy, iqx] = cmul(G[ik, iqy, iqx], gamma_conj)


@wp.kernel
def _phase_contrast_transfer_function_forward(
    G: wp.array(dtype=wp.vec2, ndim=3),
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    pctf: wp.array(dtype=wp.float32, ndim=2),
) -> None:
    """
    Forward kernel for the phase contrast transfer function forward pass.

    Args:
        G: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]
        Qx_all: wp.array(ndim=1) - [Qx]
        Qy_all: wp.array(ndim=1) - [Qy]
        Kx_all: wp.array(ndim=1) - [Kx]
        Ky_all: wp.array(ndim=1) - [Ky]
        aberrations: wp.array(dtype=wp.float32, ndim=1)
        sin_rot: wp.float32 - sin(rotation)
        cos_rot: wp.float32 - cos(rotation)
        semiconvergence_angle: wp.float32 - semiconvergence angle
        wavelength: wp.float32 - wavelength
        pctf: wp.array(dtype=wp.float32, ndim=2) - output phase contrast transfer function

    Returns:
        None (pctf is modified in place)
    """

    iqy, iqx, ik = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]

    Qx_rot = Qx * cos_rot - Qy * sin_rot
    Qy_rot = Qx * sin_rot + Qy * cos_rot

    Qx = Qx_rot
    Qy = Qy_rot

    # APERTURE FIRST.  ``aperture()`` is a hard 0/1 step (one sqrt, one asin, one
    # compare), while everything below it is three aberration polynomials and
    # three sin/cos.  ``gamma_complex = conj(A)*Am - A*conj(A_plus)`` is EXACTLY
    # (0, 0) in two cases: ``a1 == 0`` makes ``A`` exactly (0, 0), so both
    # products vanish; ``a2 == 0 and a3 == 0`` makes ``A_plus`` and ``Am``
    # exactly (0, 0), so both products vanish again.  ``cabs`` of that is 0.0,
    # ``pctf`` is zero-filled by the op and every contribution is non-negative,
    # so the skipped ``wp.atomic_add(..., 0.0)`` could not have changed the
    # accumulator -- the skip is value-preserving per thread, and the only
    # difference that survives is the atomic ORDER, which was already
    # nondeterministic.
    #
    # The reason this is a big win rather than a micro-optimisation: for
    # |Q| > max|K| + sin(alpha)/lambda the second case holds for EVERY
    # bright-field pixel, and ``ik`` is the fastest-varying launch axis, so a
    # whole warp exits together instead of diverging.  At the published
    # Fig1_Gd2O3 FF-STEM config (alpha 30 mrad, lambda 0.0487 A, U = 2,
    # (Nqy,Nqx,Nk) = (1024,1024,980)) 77.6 % of the Q grid is past that radius
    # and only 9.55 % of all threads can contribute; measured 312.3 -> 46.8 ms
    # per launch.  Do NOT "simplify" this into a |Q| radius test: the
    # bright-field mask is thresholded, not apertured, so max|K| exceeds
    # alpha/lambda and a radius test would need a margin to stay exact.
    a1 = aperture(Ky, Kx, wavelength, semiconvergence_angle)
    a2 = aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle)
    a3 = aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle)

    if a1 * (a2 + a3) > 0.0:
        chi1 = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
        apert1 = wp.vec2(a1, wp.float32(0.0))
        expichi1 = cexp(1.0, -chi1)
        A = cmul(apert1, expichi1)

        chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
        apert2 = wp.vec2(a2, wp.float32(0.0))
        expichi2 = cexp(1.0, -chi2)
        A_plus = cmul(apert2, expichi2)

        chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
        apert3 = wp.vec2(a3, wp.float32(0.0))
        expichi3 = cexp(1.0, -chi3)
        Am = cmul(apert3, expichi3)

        gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))
        gamma_abs = cabs(gamma_complex)
        wp.atomic_add(pctf, iqy, iqx, gamma_abs)


@wp.kernel(enable_backward=False)
def _direct_ptychography_backward_analytic(
    G: wp.array(dtype=wp.vec2, ndim=3),  # [Qy,Qx,ik]
    dL_dG: wp.array(dtype=wp.vec2, ndim=3),  # same shape
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    n_coeffs: int,  # <= 12
    out_grad: wp.array(dtype=wp.float32, ndim=1),  # length >= n_coeffs
):
    """
    Analytic gradient for the direct ptychography backward pass.

    Args:
        G: wp.array(dtype=wp.vec2, ndim=3) - [Qy,Qx,ik]
        dL_dG: wp.array(dtype=wp.vec2, ndim=3) - same shape
        Qx_all: wp.array(ndim=1) - [Qx]
        Qy_all: wp.array(ndim=1) - [Qy]
        Kx_all: wp.array(ndim=1) - [Kx]
        Ky_all: wp.array(ndim=1) - [Ky]
        aberrations: wp.array(dtype=wp.float32, ndim=1)
        sin_rot: wp.float32 - sin(rotation)
        cos_rot: wp.float32 - cos(rotation)
        semiconvergence_angle: wp.float32 - semiconvergence angle
        wavelength: wp.float32 - wavelength
        n_coeffs: int - <= 12 - number of coefficients
        out_grad: wp.array(dtype=wp.float32, ndim=1) - length >= n_coeffs - output gradient

    Returns:
        out_grad: wp.array(dtype=wp.float32, ndim=1) - length >= n_coeffs

    Notes:
        This kernel computes the analytic gradient of the direct ptychography forward pass.
        It is used to compute the gradient of the aberrations with respect to the loss function.
        It is a per-block tile reduction kernel.
    """

    iqy, iqx, ik = wp.tid()

    # coords
    qx = Qx_all[iqx]
    qy = Qy_all[iqy]
    kx0 = Kx_all[ik]
    ky0 = Ky_all[ik]

    # rotate Q (don't clobber)
    qxr = qx * cos_rot - qy * sin_rot
    qyr = qx * sin_rot + qy * cos_rot

    kx_p = kx0 + qxr
    ky_p = ky0 + qyr
    kx_m = kx0 - qxr
    ky_m = ky0 - qyr

    # upstream adjoint present?
    adj = dL_dG[iqy, iqx, ik]
    has_adj = not ((adj[0] == 0.0) and (adj[1] == 0.0))

    # binary aperture tests (no sqrt)
    kcut = semiconvergence_angle / wavelength
    kcut2 = kcut * kcut

    inp = (kx_p * kx_p + ky_p * ky_p) <= kcut2
    inm = (kx_m * kx_m + ky_m * ky_m) <= kcut2
    active = has_adj

    g_in = G[iqy, iqx, ik]

    # forward terms only when active; unit amplitude (top-hat)
    A0 = wp.vec2(0.0, 0.0)
    Ap = wp.vec2(0.0, 0.0)
    Am = wp.vec2(0.0, 0.0)
    if active:
        chi0 = aberration_function_cartesian(ky0, kx0, wavelength, aberrations)
        A0 = cexp(1.0, -chi0)
        chip = aberration_function_cartesian(ky_p, kx_p, wavelength, aberrations)
        Ap = cexp(1.0, -chip)
        chim = aberration_function_cartesian(ky_m, kx_m, wavelength, aberrations)
        Am = cexp(1.0, -chim)

    C1 = cmul(cconj(A0), Am)  # A* * Am
    C2 = cmul(A0, cconj(Ap))  # A   * A+*

    # coefficient loop with per-block tile reduction
    for j in range(n_coeffs):
        contrib = wp.float32(0.0)

        if active:
            dchi0 = dchi_cartesian_aberrations(ky0, kx0, wavelength, j)
            dchip = (
                dchi_cartesian_aberrations(ky_p, kx_p, wavelength, j) if inp else 0.0
            )
            dchim = (
                dchi_cartesian_aberrations(ky_m, kx_m, wavelength, j) if inm else 0.0
            )

            # dγ/da = i [ C1*(dchi0 - dchim) + C2*(dchi0 - dchip) ]
            t1 = wp.vec2(C1[0] * (dchi0 - dchim), C1[1] * (dchi0 - dchim))
            t2 = wp.vec2(C2[0] * (dchi0 - dchip), C2[1] * (dchi0 - dchip))
            dgamma = minus_i(wp.vec2(t1[0] + t2[0], t1[1] + t2[1]))

            dGout = cmul(g_in, cconj(dgamma))
            contrib = ip_real_conj(adj, dGout)  # Re{ conj(adj) * dGout }

        # cooperative block sum → single global atomic per coeff per block
        t = wp.tile(contrib)  # one scalar per thread
        s = wp.tile_sum(t)  # block-wide sum
        wp.tile_atomic_add(out_grad, s, offset=(j,))


# ---------------------------------------------------------------------------
# Fused correct-and-reduce: the same arithmetic as
# ``_direct_ptychography_forward``, but summed over ``ik`` in a register
# instead of stored.
#
# Every caller that corrects a bright-field chunk in order to form an image
# reduces the corrected chunk over ``ik`` immediately afterwards
# (``reconstruction/direct_ptychography.py``: ``ifft2(G'.sum(dim=-1)).imag``,
# the sum-first collapse R18 introduced and D2 wired into the TV fit).  So the
# ``(Nqy, Nqx, Nk)`` destination -- 86.0 MiB per chunk at the published
# ``Fig1_Gd2O3`` fit config -- is written once and read back once purely to be
# collapsed to ``(Nqy, Nqx)`` = 1.0 MiB.  Accumulating in the kernel deletes
# both passes.
#
# TWO things change relative to the unfused pair, and both are deliberate:
#
# 1. THE SUMMATION IS REASSOCIATED.  ``torch.sum`` over the last axis chooses
#    its own order; this kernel adds ``ik`` ascending in one float32 register.
#    Measured on the captured Fig1 chunk (512, 512, 43) against a complex128
#    reduction of the SAME corrected ``G``: the unfused form sits at 3.95e-08
#    relative and this one at 1.33e-07, i.e. both are at float32's own error and
#    neither is "the accurate one"; the chunk image the caller consumes moves
#    2.73e-07 relative.  That is the same size as -- and the same class as --
#    the sum-first collapse this objective already ships.
#
# 2. THE ``ik``-ONLY FACTOR ``A`` IS HOISTED, exactly as
#    ``_direct_ptychography_forward_precomputed_A`` hoists it for the unfused
#    sibling: ``A = aperture(K) * exp(-i chi(K))`` varies only with the
#    bright-field pixel, yet the un-hoisted form evaluated it (and its aperture)
#    for every one of the ``Nqy * Nqx`` threads.  ``A_all`` is built once by
#    ``_direct_ptychography_build_A``, whose six lines are a verbatim copy of the
#    ones removed here, so the hoist is a MEMOISATION and it is bit-exact --
#    measured ``torch.equal`` on the reduced chunk at five aberration vectors
#    (the captured Fig1 one, zeros, and randn x1/x10/x100), 0 of 262144 floats
#    differing.  Removing it takes one ``aperture`` (an ``asin``) off 100 % of
#    the inner iterations and one polynomial plus one ``cexp`` off the ~26 % that
#    pass the guard: 0.3111 -> 0.2411 ms on the captured chunk = 1.290x.
#
#    The guard needs ``a1``, which this form no longer computes -- but it needs
#    only the TEST, not the value.  ``aperture`` returns exactly 1.0 or 0.0 and
#    ``cexp(1, -chi)`` has unit modulus, so ``A == (0, 0)`` if and only if
#    ``a1 == 0``, and ``(A != 0) and (a2 + a3) > 0`` is exactly
#    ``a1 * (a2 + a3) > 0``.
#
#    Build ``A_all`` ABOVE the loop over bright-field chunks, never per chunk:
#    the launch is ~0.042 ms for ~43 elements (it is launch-bound) against a
#    0.070 ms per-chunk saving, so per chunk it gives back 60 % of the prize.
#    See ``build_aberration_bf_factor``, which records the same rule.
#
# 3. THE ``G`` LOAD MOVES INSIDE THE APERTURE GUARD.  In the unfused kernel the
#    store is unconditional, so the 74.1 % of ``(iqy, iqx, ik)`` whose ``gamma``
#    is identically zero still pay a load and a store; here a zero ``gamma``
#    contributes exactly nothing to the accumulator, so the load can be skipped
#    outright.  That is what makes this pay despite the worse access pattern:
#    one thread per ``(iqy, iqx)`` reads ``G`` at a stride of ``Nk`` vec2s, and
#    the unguarded floor for that (read everything, store one vec2) measures
#    0.518 ms against the unfused kernel's own 0.379 -- i.e. strided-and-
#    complete is a LOSS, and only strided-and-guarded (0.155 ms floor) wins.
#    Consequence to know: a non-finite ``G`` in the skipped region no longer
#    propagates (``0 * NaN`` is NaN, but an un-taken load is not).  Inside the
#    aperture the propagation is unchanged.
# ---------------------------------------------------------------------------
@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum(
    G: wp.array(dtype=wp.vec2, ndim=3),  # [Qy,Qx,ik]
    Qx_all: wp.array(dtype=wp.float32, ndim=1),
    Qy_all: wp.array(dtype=wp.float32, ndim=1),
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    A_all: wp.array(dtype=wp.vec2, ndim=1),  # [ik] = aperture(K)*exp(-i chi(K))
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    S_out: wp.array(dtype=wp.vec2, ndim=2),  # [Qy,Qx] = sum_ik G' [Qy,Qx,ik]
) -> None:
    """``S_out[iqy, iqx] = sum_ik G[iqy, iqx, ik] * conj(gamma_phase)``."""
    iqy, iqx = wp.tid()

    Qx0 = Qx_all[iqx]
    Qy0 = Qy_all[iqy]
    Qx = Qx0 * cos_rot - Qy0 * sin_rot
    Qy = Qx0 * sin_rot + Qy0 * cos_rot

    acc = wp.vec2(0.0, 0.0)
    for ik in range(G.shape[2]):
        Kx = Kx_all[ik]
        Ky = Ky_all[ik]

        # ``A`` is the hoisted ``ik``-only factor; ``A != (0, 0)`` is exactly
        # ``aperture(K) > 0`` (see item 2 of the block comment above).
        A = A_all[ik]
        a2 = aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle)
        a3 = aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle)

        if (A[0] != 0.0 or A[1] != 0.0) and (a2 + a3) > 0.0:
            # Issue the ``G`` load FIRST, not at the accumulate.  Nothing
            # between here and the accumulate depends on it, and this thread's
            # loads are the one long-latency operation in the loop: it reads
            # ``G[iqy, iqx, ik]`` while its warp-neighbours read ``iqx +- 1``,
            # i.e. addresses ``Nk`` vec2s (344 B at the published Fig1 chunk)
            # apart, so each lane pulls its own sector and the latency is not
            # amortised across the warp.  Spelling the load before the two
            # aberration polynomials gives ptxas ~60 flops of independent work
            # to cover it with; written at the accumulate it had none.
            # Measured on the captured (512, 512, 43) Fig1 chunk: 0.2395 ->
            # 0.2238 ms = 1.070x, and BIT-EXACT (``torch.equal`` on the reduced
            # chunk at five aberration vectors -- it is the same load of the
            # same address, only earlier).
            #
            # Do NOT "fix" this by transposing ``G`` to ``[ik, Qy, Qx]`` so the
            # warp coalesces: measured on the same chunk, k-major is 0.9895x on
            # this kernel (i.e. slightly SLOWER) and 1.124x on its guarded
            # read+store roof, because the roof is latency- and issue-bound
            # rather than sector-bound -- and it would make the ANALYTIC
            # ADJOINT, which launches ``(Nqy, Nqx, Nk)`` with ``ik`` fastest and
            # is already coalesced, read uncoalesced instead.  See
            # docs/perf-lab/reports/2026-08-06-d22-*.md.
            g_in = G[iqy, iqx, ik]

            chi2 = aberration_function_cartesian(
                Ky + Qy, Kx + Qx, wavelength, aberrations
            )
            apert2 = wp.vec2(a2, wp.float32(0.0))
            expichi2 = cexp(1.0, -chi2)
            A_plus = cmul(apert2, expichi2)

            chi3 = aberration_function_cartesian(
                Ky - Qy, Kx - Qx, wavelength, aberrations
            )
            apert3 = wp.vec2(a3, wp.float32(0.0))
            expichi3 = cexp(1.0, -chi3)
            Am = cmul(apert3, expichi3)

            gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

            gamma_abs = cabs(gamma_complex)
            gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
            gamma_phase = wp.vec2(
                gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs
            )
            acc = acc + cmul(g_in, cconj(gamma_phase))
    S_out[iqy, iqx] = acc


@wp.kernel(enable_backward=False)
def _direct_ptychography_aperture_mask_planes(
    Qx_all: wp.array(dtype=wp.float32, ndim=1),
    Qy_all: wp.array(dtype=wp.float32, ndim=1),
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    mask: wp.array(dtype=wp.uint8, ndim=3),  # [Qy,Qx,ik]
) -> None:
    """The ``ip``-INDEPENDENT half of :func:`_direct_ptychography_forward_ksum_planes`.

    ``a2``/``a3`` are functions of ``(iqy, iqx, ik)`` only -- neither reads
    ``aberrations[ip]`` -- so the planes kernel evaluates them ``n_planes`` times
    for the same answer, and each evaluation is a ``wp.sqrt`` plus a ``wp.asin``.
    This kernel evaluates them ONCE per ``(iqy, iqx, ik)`` and packs the pair into
    one bit each.

    BIT-EXACT by construction, not by tolerance: ``aperture`` ends in
    ``wp.where(ktheta < semiconvergence_angle_max, 1.0, 0.0)``, so its result is
    exactly ``+0.0`` or ``+1.0``; ``wp.float32(m & 1)`` reproduces those two values
    exactly, and every arithmetic use of them downstream is unchanged.  The
    comparison itself is the SAME comparison on the SAME operands -- this is a
    memoisation, not the (non-exact) algebraic rewrite
    ``asin(q*lambda) < alpha  <=>  qx^2 + qy^2 < (sin(alpha)/lambda)^2``, which
    would move points that sit within an ULP of the aperture edge.
    """
    iqy, iqx, ik = wp.tid()
    Qx0 = Qx_all[iqx]
    Qy0 = Qy_all[iqy]
    Qx = Qx0 * cos_rot - Qy0 * sin_rot
    Qy = Qx0 * sin_rot + Qy0 * cos_rot
    Kx = Kx_all[ik]
    Ky = Ky_all[ik]
    a2 = aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle)
    a3 = aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle)
    b = 0
    if a2 != 0.0:
        b += 1
    if a3 != 0.0:
        b += 2
    mask[iqy, iqx, ik] = wp.uint8(b)


@wp.kernel(enable_backward=False)
def _direct_ptychography_forward_ksum_planes(
    G: wp.array(dtype=wp.vec2, ndim=3),  # [Qy,Qx,ik]
    Qx_all: wp.array(dtype=wp.float32, ndim=1),
    Qy_all: wp.array(dtype=wp.float32, ndim=1),
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    A_all: wp.array(dtype=wp.vec2, ndim=2),  # [ik, ip]
    aberrations: wp.array(dtype=wp.float32, ndim=2),  # [ip, n_ab]
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    aperture_mask: wp.array(dtype=wp.uint8, ndim=3),  # [Qy,Qx,ik]
    S_out: wp.array(dtype=wp.vec2, ndim=3),  # [ip,Qy,Qx]
) -> None:
    """:func:`_direct_ptychography_forward_ksum` with a leading PLANE axis: one
    launch reduces the same ``[Nqy, Nqx, Nk]`` chunk under ``n_planes`` different
    aberration vectors.

    This is :func:`_direct_ptychography_forward_precomputed_A_planes`' mechanism
    on the ``ik``-REDUCED kernel.  That one was built for the ``k``-outer branch of
    ``direct_ptychography_depth_section``; the ``_CHUNK_SUM_FIRST`` branch of the
    same sweep still issued one launch (and one ``torch.library.custom_op``
    dispatch) per ``(chunk, plane)`` pair.

    Same arithmetic, line for line, in the same order and the same float32
    precision as the per-plane kernel -- only the ``wp.tid()`` unpacking and two
    subscripts change.  Thread ``(ip, iqy, iqx)`` reads ONE aberration row
    (``aberrations[ip]``, a 1-D sub-array view -- no copy) and the ``ik`` loop
    accumulates in the identical ascending order into one register, so output word
    ``[ip, iqy, iqx]`` is **bit-identical** to what a per-plane launch writes at
    ``[iqy, iqx]``.  There is no cross-thread interaction that could reassociate.

    ``A_all`` is indexed ``[ik, ip]``, NOT ``[ip, ik]``, for the same reason the
    ``kouter`` planes kernel is: the caller holds one ``[n_bf_total, n_planes]``
    factor built above the chunk loop and that layout's chunk slice ``[s:e]`` is
    contiguous for free.

    The launch grid is ``[n_planes, Nqy, Nqx]``, so ``iqx`` stays the
    fastest-varying thread index and the store stays unit-stride within a warp;
    the strided ``G`` read is unchanged from the per-plane kernel (and per the
    block comment above, transposing it is a measured LOSS on this kernel).  The
    ``G`` load is issued once per ``(ip, ik)``, i.e. it is NOT deduplicated across
    planes -- the prize here is the per-call HOST cost, which D274-run measured at
    a fixed 32-124 us of ``wp.launch`` plus D279's 22.1 us of custom-op dispatch
    and which does not shrink with problem size.
    """
    ip, iqy, iqx = wp.tid()

    ab = aberrations[ip]

    Qx0 = Qx_all[iqx]
    Qy0 = Qy_all[iqy]
    Qx = Qx0 * cos_rot - Qy0 * sin_rot
    Qy = Qx0 * sin_rot + Qy0 * cos_rot

    acc = wp.vec2(0.0, 0.0)
    for ik in range(G.shape[2]):
        Kx = Kx_all[ik]
        Ky = Ky_all[ik]

        A = A_all[ik, ip]
        # ``a2``/``a3`` do not depend on ``ip``, so they come from the mask
        # ``_direct_ptychography_aperture_mask_planes`` builds once per chunk
        # instead of being recomputed (2 x sqrt + 2 x asin) once per PLANE.
        # Exactly ``+0.0``/``+1.0`` either way -- see that kernel's docstring.
        m = int(aperture_mask[iqy, iqx, ik])
        a2 = wp.float32(m & 1)
        a3 = wp.float32((m >> 1) & 1)

        if (A[0] != 0.0 or A[1] != 0.0) and (a2 + a3) > 0.0:
            g_in = G[iqy, iqx, ik]

            chi2 = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, ab)
            apert2 = wp.vec2(a2, wp.float32(0.0))
            expichi2 = cexp(1.0, -chi2)
            A_plus = cmul(apert2, expichi2)

            chi3 = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, ab)
            apert3 = wp.vec2(a3, wp.float32(0.0))
            expichi3 = cexp(1.0, -chi3)
            Am = cmul(apert3, expichi3)

            gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

            gamma_abs = cabs(gamma_complex)
            gamma_abs = wp.where(gamma_abs < 1e-8, 1e-8, gamma_abs)
            gamma_phase = wp.vec2(
                gamma_complex[0] / gamma_abs, gamma_complex[1] / gamma_abs
            )
            acc = acc + cmul(g_in, cconj(gamma_phase))
    S_out[ip, iqy, iqx] = acc


@wp.kernel(enable_backward=False)
def _direct_ptychography_backward_analytic_ksum(
    G: wp.array(dtype=wp.vec2, ndim=3),  # [Qy,Qx,ik]
    dL_dS: wp.array(dtype=wp.vec2, ndim=2),  # [Qy,Qx]  -- the REDUCED adjoint
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    sin_rot: wp.float32,
    cos_rot: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    n_coeffs: int,  # <= 12
    out_grad: wp.array(dtype=wp.float32, ndim=1),  # length >= n_coeffs
):
    """Adjoint of :func:`_direct_ptychography_forward_ksum`.

    LINE-FOR-LINE ``_direct_ptychography_backward_analytic`` with ONE change:
    the upstream adjoint is indexed ``[iqy, iqx]`` instead of
    ``[iqy, iqx, ik]``.  That is not an approximation -- it is the exact
    consequence of the fused forward: ``S = sum_ik G'``, so ``dL/dG'_{q,k} =
    dL/dS_q`` for every ``k``, i.e. the tensor this kernel used to read was a
    broadcast of ``dL_dS`` along ``ik`` that the caller had to materialise with
    ``expand(...).contiguous()`` -- 86.0 MiB of duplicated floats per chunk at
    the Fig1 fit config, allocated, written and read on every grad-enabled body.
    Per thread the arithmetic is IDENTICAL to the unfused kernel's; only the
    load address changes, so the two differ at most by the order of their
    ``wp.tile_atomic_add``s, which was already nondeterministic (R9d).

    Keep this in step with ``_direct_ptychography_backward_analytic``: it
    carries the same deliberate mismatch with the forward it adjoints (unit
    amplitudes, no derivative of the ``gamma`` normalisation -- DISCOVERED D7),
    so it must be gated against THAT kernel, never against autograd.
    """

    iqy, iqx, ik = wp.tid()

    # coords
    qx = Qx_all[iqx]
    qy = Qy_all[iqy]
    kx0 = Kx_all[ik]
    ky0 = Ky_all[ik]

    # rotate Q (don't clobber)
    qxr = qx * cos_rot - qy * sin_rot
    qyr = qx * sin_rot + qy * cos_rot

    kx_p = kx0 + qxr
    ky_p = ky0 + qyr
    kx_m = kx0 - qxr
    ky_m = ky0 - qyr

    # upstream adjoint present?  ``dL_dS`` is the reduced [Qy,Qx] adjoint; the
    # unfused kernel read the same value out of an ik-broadcast copy of it.
    adj = dL_dS[iqy, iqx]
    has_adj = not ((adj[0] == 0.0) and (adj[1] == 0.0))

    # binary aperture tests (no sqrt)
    kcut = semiconvergence_angle / wavelength
    kcut2 = kcut * kcut

    inp = (kx_p * kx_p + ky_p * ky_p) <= kcut2
    inm = (kx_m * kx_m + ky_m * ky_m) <= kcut2
    active = has_adj

    g_in = G[iqy, iqx, ik]

    # forward terms only when active; unit amplitude (top-hat)
    A0 = wp.vec2(0.0, 0.0)
    Ap = wp.vec2(0.0, 0.0)
    Am = wp.vec2(0.0, 0.0)
    if active:
        chi0 = aberration_function_cartesian(ky0, kx0, wavelength, aberrations)
        A0 = cexp(1.0, -chi0)
        chip = aberration_function_cartesian(ky_p, kx_p, wavelength, aberrations)
        Ap = cexp(1.0, -chip)
        chim = aberration_function_cartesian(ky_m, kx_m, wavelength, aberrations)
        Am = cexp(1.0, -chim)

    C1 = cmul(cconj(A0), Am)  # A* * Am
    C2 = cmul(A0, cconj(Ap))  # A   * A+*

    # coefficient loop with per-block tile reduction
    for j in range(n_coeffs):
        contrib = wp.float32(0.0)

        if active:
            dchi0 = dchi_cartesian_aberrations(ky0, kx0, wavelength, j)
            dchip = (
                dchi_cartesian_aberrations(ky_p, kx_p, wavelength, j) if inp else 0.0
            )
            dchim = (
                dchi_cartesian_aberrations(ky_m, kx_m, wavelength, j) if inm else 0.0
            )

            # dγ/da = i [ C1*(dchi0 - dchim) + C2*(dchi0 - dchip) ]
            t1 = wp.vec2(C1[0] * (dchi0 - dchim), C1[1] * (dchi0 - dchim))
            t2 = wp.vec2(C2[0] * (dchi0 - dchip), C2[1] * (dchi0 - dchip))
            dgamma = minus_i(wp.vec2(t1[0] + t2[0], t1[1] + t2[1]))

            dGout = cmul(g_in, cconj(dgamma))
            contrib = ip_real_conj(adj, dGout)  # Re{ conj(adj) * dGout }

        # cooperative block sum → single global atomic per coeff per block
        t = wp.tile(contrib)  # one scalar per thread
        s = wp.tile_sum(t)  # block-wide sum
        wp.tile_atomic_add(out_grad, s, offset=(j,))


