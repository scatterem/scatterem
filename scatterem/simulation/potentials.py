r"""Projected electrostatic potentials by finite-difference electrostatics (FDES).

Written from the published formulation of multislice electrostatics cited
below.

What it computes
----------------
For a periodic (orthorhombic) supercell sampled on ``(Ny, Nx)`` pixels and cut
into ``nss`` depth slices, the projected electrostatic potential of slice ``s``
is the superposition of single-atom projected potentials,

.. math::

    V_s(\mathbf r) \;=\; \sum_{m \in s} o_m\, w_m^{p}\;
                          v_{z, Z_m}(\mathbf r - \mathbf r_m)
    \qquad [\mathrm{V\,\AA}],

with :math:`o_m` the fractional site occupancy and :math:`w_m` an optional
per-atom weight (see ``weights``/``weight_power``).  Multiplied by the
relativistic interaction constant :math:`\sigma(E)` in rad/(V Å) this is exactly
the multislice phase grating argument, :math:`t_s = \exp(i\sigma V_s)`
(Cowley & Moodie, *Acta Cryst.* **10** (1957) 609; Kirkland, *Advanced Computing
in Electron Microscopy*, 2nd ed., Ch. 6).

The sum is evaluated by the convolution theorem rather than atom by atom in real
space.  Writing the slice's atomic density as a comb of delta functions,
:math:`\rho_s(\mathbf r) = \sum_m o_m w_m^p \delta(\mathbf r - \mathbf r_m)`,

.. math::

    V_s(\mathbf r) \;=\; \mathcal F^{-1}\!\big[\, f_e^{Z}(\mathbf q)\,
                          \hat\rho_{s,Z}(\mathbf q) \,\big],

summed over species :math:`Z`, where :math:`f_e^{Z}(q)` is the electron
scattering factor **in potential units** (V Å³) — precisely the 2-D Fourier
transform of the single-atom projected potential,
:math:`f_e(\mathbf q) = \int v_z(\mathbf r)\,e^{-2\pi i \mathbf q\cdot\mathbf r}\,
\mathrm d^2 r`, so that :math:`f_e(0) = \int v_z \,\mathrm d^2r`.

Scattering factors
------------------
:func:`~scatterem.simulation.scattering_factors.calculate_scattering_factors`
supplies :math:`f_e` from the Gaussian-free parametrisation of

    I. Lobato and D. Van Dyck, *Acta Cryst.* **A70** (2014) 636-649,

.. math::

    f_e(q) \;=\; \sum_{i=1}^{5} a_i \,\frac{2 + b_i q^2}{(1 + b_i q^2)^2},
    \qquad q^2 = q_y^2 + q_x^2 \;[\mathrm{\AA^{-2}}],

converted from scattering length (Å) to potential units by the Mott-Bethe / Born
prefactor :math:`h^2/(2\pi m_e e) = 2\pi a_0 e = 47.878\;\mathrm{V\,\AA^2}`
(Kirkland, Ch. 5, Eqs. 5.9-5.15).  That conversion is done upstream; this module
only consumes ``fe`` in V Å³.

FDES density: the bilinear splat
--------------------------------
A delta comb is not representable on a sampled grid, so each atom's mass is
scattered bilinearly onto the four pixels bracketing its sub-pixel position
("finite-difference electrostatics"; see e.g. W. Van den Broek, X. Jiang and
C. T. Koch, *Ultramicroscopy* **158** (2015) 89, and Kirkland Ch. 6 on
sub-pixel-accurate atom placement).  With :math:`u` the atom's pixel coordinate
along an axis and :math:`\theta = u - \lfloor u \rfloor`,

.. math::

    \lfloor u \rfloor \;\mathrm{gets}\; (1-\theta), \qquad
    \lceil  u \rceil  \;\mathrm{gets}\; \theta,

both indices reduced modulo the grid extent.  The two axes multiply.  This is
the triangular (hat) interpolation kernel
:math:`\Lambda(u - n) = \max(0, 1 - |u - n|)`.

Sinc deconvolution
------------------
Splatting convolves the ideal comb with :math:`\Lambda`, whose discrete-time
Fourier transform is, to leading order, :math:`\mathrm{sinc}^2(f)` with
:math:`f = k/N` in cycles per pixel.  This module divides the splat spectrum by
:math:`\mathrm{sinc}(f)` **once** per axis, not twice.  That is deliberate.  The
grid samples the potential at nodes, while the splat weights are a cell-overlap
integral; the two conventions differ by exactly one factor of the kernel
transform, and the single division is the compromise that minimises the worst
case over sub-pixel atom positions.  Measured against the analytic band-limited
sampled potential of one Si atom in a 5 Å cell at ``N = 512``:

===========================  ========  ==========  ==========
atom position                no sinc   sinc**1     sinc**2
===========================  ========  ==========  ==========
exactly on a pixel           0.000     0.045       0.100
generic sub-pixel            0.072     0.049       0.031
===========================  ========  ==========  ==========

i.e. ``sinc**1`` is roughly the geometric mean of the two extremes and is
uniformly acceptable, whereas either endpoint is exact for one case and worst
for the other.  Set ``sinc_deconvolution=False`` to disable it.

Band limit
----------
A soft error-function low-pass suppresses the aliasing wrap-around of the sharp
atomic cores: with :math:`\rho = 2\sqrt{(k_y/N_y)^2 + (k_x/N_x)^2}` (so
:math:`\rho = 1` at the Nyquist radius) the mask is
:math:`M = \mathrm{erf}(\max(0, 1-\rho)/\text{soft})`.  The cut is at the **full**
Nyquist radius with a soft edge, not the usual hard 2/3 aperture: the 2/3 rule
belongs to the propagate/transmit loop (it is applied there), whereas here the
only job is to stop the core's tail from folding back.  ``M = 1`` at DC to
within 1e-12, so the integral invariant below is unaffected.

Normalisation and the integral invariant
----------------------------------------
The splat carries dimensionless counts per pixel; ``norm = 1/(dy*dx)`` converts
that to an areal number density in Å⁻²; multiplying by ``fe`` in V Å³ gives V Å
per reciprocal cell; and the inverse DFT's implicit ``1/(Ny*Nx)`` completes the
Riemann sum of the continuum inverse Fourier integral with
:math:`\mathrm dq_y\,\mathrm dq_x = 1/(L_y L_x)`.  The checkable consequence,
used as the primary correctness test of this module, is

.. math::

    \sum_{\text{grid}} V\, \mathrm dy\, \mathrm dx
      \;=\; \sum_{m,\,\text{tiles}} o_m\, w_m^{p}\, f_e^{Z_m}(0).

Conventions (all load-bearing)
------------------------------
* FFT sign/normalisation: NumPy/torch default — forward
  :math:`e^{-2\pi i k n/N}` unnormalised, inverse carries :math:`1/N`.
* **No fftshift anywhere.**  Every reciprocal-space array (splat spectrum, sinc
  factors, caller-supplied ``fe``, band-limit mask) is corner-origin
  ``fftfreq``-ordered: ``k = 0`` at index 0.
* Real-FFT half plane: only ``fe[..., :Nx//2+1]`` is ever read.  This is legal
  because ``fe`` is an *even* function of ``g`` — ``fftfreq``'s Nyquist column
  carries ``-0.5`` and ``rfftfreq``'s carries ``+0.5``, and an even function
  cannot tell them apart.  Anything odd in ``g`` must not be passed as ``fe``.
* Axis mapping: ``atoms[:, 0]`` → grid axis 0 (length ``Ny``, cell edge
  ``unitcell[0]``); ``atoms[:, 1]`` → grid axis 1.
* Origin: no half-pixel offset.  Node ``(0, 0)`` is fractional coordinate
  ``(0, 0)``.
* Element order is ``list(set(atoms[:, 3].astype(int32)))`` — CPython small-int
  hash order, which is **not** sorted in general (``{8,42,16,1,92}`` gives
  ``[1, 8, 42, 16, 92]``).  This is an external contract: callers that
  precompute ``fe`` index it in exactly this order.
* ``device=None`` resolves to CUDA when available, not to CPU.

Deviations from the module this replaces (deliberate, documented)
-----------------------------------------------------------------
1. ``seed`` seeds a private :class:`torch.Generator` instead of calling the
   global :func:`torch.manual_seed`, so a seeded call no longer perturbs the
   caller's RNG stream.  Seeded results are self-consistent (same seed → bitwise
   identical output) but are not comparable with any other RNG source.
2. Thermal displacements for all lateral tile replicas are drawn in one batched
   sample rather than one sample per tile.  Statistically identical — each
   replica still vibrates independently — but a different stream ordering.
3. ``_find_equivalent_sites`` uses a k-d tree instead of a dense pairwise
   distance matrix (``O(N log N)`` instead of ``O(N^2)`` time *and* memory), and
   is skipped entirely when ``displacements=False``, where it is unused.

Not reimplemented here (independent upstream modules): the scattering factor
evaluation and the band-limit mask.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections import OrderedDict
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from ._grid import bandwidth_limit_array_torch
from .scattering_factors import calculate_scattering_factors

__all__ = ["make_potential", "share_splat"]


# Integer dtype for the atomic-number cast and the slice-index array.  The cast
# to a fixed-width integer is what makes the element ordering reproducible:
# numpy int32 scalars hash exactly like python ints, so
# ``list(set(...))`` yields the same order it would for python ints.
_int = np.int32

# Active splat cache, or None outside any ``share_splat`` block.  See
# :func:`share_splat`.
_SPLAT_SHARE: Optional[dict] = None

# Real dtype for every possibly-complex torch dtype this module accepts.
_REAL_OF_COMPLEX = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
    torch.float16: torch.float16,
}

_NUMPY_OF_REAL = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
}

_COMPLEX_OF_REAL = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
    torch.float16: torch.complex32,
}

# cuFFT cannot plan a batched transform with more than ~2**31 elements; keep a
# comfortable margin.  Also the granularity of the tile-batched splat.
_FFT_PLAN_ELEMENTS = 1 << 30
_SPLAT_BLOCK = 1 << 20

# Small bounded memo tables for the two pure, grid-only quantities that used to
# be rebuilt on every forward pass.
_SINC_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_FE_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_CACHE_MAX = 16
_CACHE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _complex_to_real_dtype_torch(dtype: torch.dtype) -> torch.dtype:
    """Map a possibly-complex torch dtype to the matching real dtype.

    ``complex64 -> float32``, ``complex128 -> float64``, real dtypes pass
    through.  This single choice fixes the accumulation dtype of the splat, the
    dtype of the sinc factors and the dtype of the returned potential: a request
    for ``complex64`` and a request for ``float32`` must produce byte-identical
    real output.
    """
    try:
        return _REAL_OF_COMPLEX[dtype]
    except KeyError:
        # Fallback for any dtype added to torch later.
        return torch.empty(0, dtype=dtype).abs().dtype


def _get_device(device_type: Union[None, str, torch.device] = None) -> torch.device:
    """Resolve a device specification.

    ``None`` means *CUDA if one is present*, **not** CPU — an unqualified call
    therefore lands on the GPU on a machine that has one.  Anything else is
    handed to :class:`torch.device` unchanged.
    """
    if device_type is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_type)


def _ensure_array(input: Any) -> Any:
    """Wrap a bare scalar in a length-1 array; pass sequences through.

    Lets ``subslices=1.0`` behave as ``subslices=[1.0]``.  Sequences are passed
    through *unconverted* (a list stays a list), which is fine because the only
    consumers are ``len``, :func:`numpy.searchsorted` and float indexing.
    """
    if hasattr(input, "__len__") and not isinstance(input, str):
        return input
    return np.asarray([input])


def _sinc(x: torch.Tensor) -> torch.Tensor:
    r"""Normalised sinc, :math:`\sin(\pi x)/(\pi x)`.

    The removable singularity at ``x = 0`` is handled by clamping ``|x| < 1e-20``
    to ``+1e-20`` before dividing, which returns exactly ``1.0`` in floating
    point.  The clamp does not preserve the sign of a tiny negative argument;
    that is irrelevant because it can only ever affect the DC bin, where the
    function is even anyway.

    Parameters
    ----------
    x : torch.Tensor
        Normalised frequency in cycles per pixel.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``x``.
    """
    safe = torch.where(x.abs() < 1e-20, torch.full_like(x, 1e-20), x)
    arg = math.pi * safe
    return torch.sin(arg) / arg


def _find_equivalent_sites(positions: np.ndarray, EPS: float = 1e-3) -> np.ndarray:
    """Map every atom onto a representative of the site it shares.

    Partially occupying atoms that sit on one crystallographic site are one
    vibrating site, not several independent ones, so in the Einstein
    frozen-phonon model they must receive the *same* thermal displacement.  This
    returns an index map that lets the caller re-index the drawn displacements.

    Coincidence rule, chosen to be reproducible rather than merely plausible::

        rep(j) = max{ i < j : |x_i - x_j| < EPS },   rep(j) = j if no such i

    i.e. the **largest** qualifying lower index.

    Parameters
    ----------
    positions : (natoms, 3) array_like
        Fractional coordinates (dimensionless).
    EPS : float, optional
        Coincidence radius in fractional coordinates, default 1e-3.

    Returns
    -------
    numpy.ndarray of shape (natoms,), dtype int32
        Identity where no coincidence is found.

    Notes
    -----
    Two known limitations, preserved so that the numerics do not shift:

    * The map is **not transitively closed**.  For three mutually coincident
      atoms 0, 1, 2 it gives ``rep(1) = 0`` and ``rep(2) = 1``, so atom 2
      inherits atom 1's own drawn displacement and the three do not move as a
      unit.
    * Distances are not wrapped periodically, so sites at fractional ``0.0`` and
      ``1 - 1e-9`` are not recognised as the same.

    Implemented with a k-d tree (``O(N log N)`` time, output-sized memory); a
    dense pairwise distance matrix would need ~40 GB at 1e5 atoms.
    """
    from scipy.spatial import cKDTree

    pos = np.ascontiguousarray(positions, dtype=np.float64)
    natoms = pos.shape[0]
    out = np.arange(natoms, dtype=_int)
    if natoms < 2:
        return out

    # query_pairs uses a closed ball (d <= r) and returns pairs with i < j.
    pairs = cKDTree(pos).query_pairs(r=EPS, output_type="ndarray")
    if pairs.size == 0:
        return out

    # Re-impose the strict inequality of the rule above.
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    pairs = pairs[d < EPS]
    if pairs.size == 0:
        return out

    # rep(j) = max over qualifying i.  maximum.at is order-independent, unlike a
    # fancy-index assignment with repeated targets.
    best = np.full(natoms, -1, dtype=np.int64)
    np.maximum.at(best, pairs[:, 1], pairs[:, 0])
    hit = best >= 0
    out[hit] = best[hit].astype(_int)
    return out


# --------------------------------------------------------------------------- #
# memoised, purely grid-dependent factors
# --------------------------------------------------------------------------- #
def _cache_get(cache: "OrderedDict[tuple, torch.Tensor]", key: tuple):
    with _CACHE_LOCK:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value


def _cache_put(cache: "OrderedDict[tuple, torch.Tensor]", key: tuple, value) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)


def _inverse_sinc_half_plane(
    Ny: int, Nx: int, realdtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    r"""Reciprocal of the separable sinc interpolation kernel on the rfft grid.

    Returns :math:`1 / [\,\mathrm{sinc}(f_y)\,\mathrm{sinc}(f_x)\,]` of shape
    ``(Ny, Nx//2+1)``, with :math:`f_y` the two-sided ``fftfreq(Ny)`` (signed,
    wrap-around order) and :math:`f_x` the one-sided ``rfftfreq(Nx)``, both in
    cycles per pixel.  One multiply by this array replaces two full-array
    divisions of the complex spectrum.

    The sinc is evaluated in float32 and then promoted, which bounds the
    correction's accuracy at ~1e-7 relative — far below the accuracy of the
    scattering-factor table itself.  Amplification is at most
    :math:`(\pi/2)^2 \approx 2.47` at the Nyquist corner, so this never
    amplifies noise appreciably.

    Pure function of ``(Ny, Nx, realdtype, device)``; memoised.
    """
    key = (int(Ny), int(Nx), str(realdtype), str(device))
    cached = _cache_get(_SINC_CACHE, key)
    if cached is not None:
        return cached

    # float32 explicitly, so the result does not depend on the process-wide
    # torch default dtype.
    fy = torch.fft.fftfreq(Ny, dtype=torch.float32)
    fx = torch.fft.rfftfreq(Nx, dtype=torch.float32)
    inv = (1.0 / _sinc(fy).to(realdtype)).reshape(Ny, 1) * (
        1.0 / _sinc(fx).to(realdtype)
    ).reshape(1, -1)
    inv = inv.to(device)
    _cache_put(_SINC_CACHE, key, inv)
    return inv


def _scattering_factor_half_plane(
    gridshape: tuple,
    gridsize: tuple,
    elements: Sequence[int],
    realdtype: torch.dtype,
    device: torch.device,
    pp: int,
    cache: bool = True,
) -> torch.Tensor:
    """Device-resident ``(M, Ny, Nx//2+1)`` scattering factors in V Å³.

    ``calculate_scattering_factors`` is a pure function of the grid and the
    element set, but it is by far the most expensive thing in a cold call (on a
    2048x2048 grid it dominates the total by ~50x), and reconstruction-style
    callers re-enter :func:`make_potential` every forward pass with an unchanged
    grid.  Memoise the *already halved, already cast, already uploaded* tensor,
    which also removes the wasted half of the host evaluation and of the PCIe
    transfer.
    """
    key = (
        tuple(int(g) for g in gridshape),
        tuple(float(g) for g in gridsize),
        tuple(int(e) for e in elements),
        str(realdtype),
        str(device),
        int(pp),
    )
    if cache:
        cached = _cache_get(_FE_CACHE, key)
        if cached is not None:
            return cached

    fe = calculate_scattering_factors(tuple(gridshape), tuple(gridsize), list(elements))
    fe_t = torch.from_numpy(
        np.ascontiguousarray(fe[..., :pp], dtype=_NUMPY_OF_REAL[realdtype])
    ).to(device)
    if cache:
        _cache_put(_FE_CACHE, key, fe_t)
    return fe_t


# --------------------------------------------------------------------------- #
# inverse transform
# --------------------------------------------------------------------------- #
def _irfft2(spectrum: torch.Tensor, Ny: int, Nx: int) -> torch.Tensor:
    """Inverse real FFT of a corner-origin half spectrum to ``(..., Ny, Nx)``.

    The output size must be given explicitly: from ``Nx//2+1`` retained columns
    the full width is only determined up to parity, and an odd ``Nx`` would
    otherwise come back as ``Nx - 1``.

    One code path only, on every device and whether or not gradients are
    required.  That is load-bearing, not incidental: callers compare a
    forward-only evaluation against a differentiable one (a model against its
    own zero-residual "observation"), and any grad-conditional numerical route
    would make that difference a nonzero ~1e-12 instead of exactly zero.
    """
    return torch.fft.irfft2(spectrum, s=(Ny, Nx))


# --------------------------------------------------------------------------- #
# per-fe tail
# --------------------------------------------------------------------------- #
def _apply_form_factors(
    P: torch.Tensor,
    fe: Optional[np.ndarray],
    elements: Sequence[int],
    psize: Sequence[int],
    gsize: Sequence[float],
    nelements: int,
    pixels_: Sequence[int],
    structure: Any,
    tiling: Sequence[int],
    bandwidthlimit: Optional[float],
    device: torch.device,
    copy: bool = False,
    cache_fe: bool = True,
) -> torch.Tensor:
    r"""Turn a deconvolved splat spectrum into a real projected potential.

    This is the cheap, form-factor-dependent tail of :func:`make_potential`:
    everything downstream of the atom splat, and the only part that changes when
    several potentials are built from one structure with different ``fe``.

    Steps:

    1. evaluate ``fe`` if it was not supplied;
    2. clone ``P`` if it is shared;
    3. multiply by ``fe[..., :Nx//2+1]``, broadcast over the subslice axis;
    4. sum over the element axis and scale by ``norm = 1/(dy*dx)``;
    5. apply the soft band limit;
    6. inverse real FFT to ``(nss, Ny, Nx)``.

    Parameters
    ----------
    P : torch.Tensor, shape (M, nss, Ny, Nx//2+1), complex
        Sinc-deconvolved half-plane spectrum of the per-(element, subslice) splat.
    fe : (M, Ny, Nx) numpy.ndarray or None
        Scattering factors in V Å³, corner-origin order, indexed along axis 0 in
        the element order of :func:`make_potential`.  Only the first
        ``Nx//2+1`` columns are read, so callers may leave the rest zero.  Any
        real dtype: it is cast to ``P``'s real dtype before the multiply, so a
        float64 ``fe`` neither changes the result nor promotes the output.
    bandwidthlimit : float or None
        Width of the erf taper; ``None`` disables the mask entirely.
    copy : bool
        Clone ``P`` before the in-place multiply.  Set whenever ``P`` is shared
        with another consumer.

    Returns
    -------
    torch.Tensor
        Real ``(nss, Ny, Nx)`` potential in V Å.

    Notes
    -----
    ``norm = Ny*Nx/(Ly*Lx) = 1/(dy*dx)`` is the inverse pixel area in Å⁻²; it
    converts the splat's dimensionless mass per pixel into an areal number
    density, which multiplied by ``fe`` in V Å³ gives V Å.  See the module
    docstring for the resulting integral invariant.
    """
    Ny, Nx = int(pixels_[0]), int(pixels_[1])
    pp = Nx // 2 + 1
    realdtype = _complex_to_real_dtype_torch(P.dtype)

    if fe is None:
        fe_t = _scattering_factor_half_plane(
            psize, gsize, elements, realdtype, device, pp, cache=cache_fe
        )
    else:
        fe_np = np.asarray(fe)
        # astype on the non-contiguous half-plane view produces a contiguous
        # array in one pass; casting here (rather than relying on an in-place
        # multiply to truncate) keeps the output dtype independent of fe's.
        fe_t = torch.from_numpy(
            np.ascontiguousarray(fe_np[..., :pp], dtype=_NUMPY_OF_REAL[realdtype])
        ).to(device)

    if copy:
        P = P.clone()

    # (M, 1, Ny, pp) broadcast over the subslice axis, in place: no (M, nss, Ny,
    # pp) temporary, and no risk of type promotion.
    P = P.mul_(fe_t.unsqueeze(1))

    a0 = float(structure.unitcell[0])
    a1 = float(structure.unitcell[1])
    # Only the first two tiling entries index the grid, but the cell area scales
    # with the full tiling product.
    norm = float(Ny) * float(Nx) / (a0 * a1) / float(np.prod(np.asarray(tiling)))

    # Reduce first, scale second: the scalar multiply then costs 1/M as much.
    V = P.sum(0) * norm

    if bandwidthlimit is not None:
        V = bandwidth_limit_array_torch(V, limit=1, soft=bandwidthlimit, rfft=True)

    return _irfft2(V, Ny, Nx)


# --------------------------------------------------------------------------- #
# scoped splat sharing
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def share_splat():
    """Reuse the structure-dependent half of :func:`make_potential` in a block.

    Inside the block, the first ``displacements=False`` call stores its
    sinc-deconvolved reciprocal-space splat spectrum, and any later call in the
    same block that differs *only* in ``fe`` and/or ``bandwidthlimit`` reuses it
    instead of re-splatting every atom and re-transforming.

    The motivating pattern is the inelastic/TDS channel, which builds several
    potentials from one structure and one weight vector — one detector-windowed
    emission form factor per detector, the all-angle absorptive limit, and the
    elastic potential — differing only in the form factor.  The splat and its
    transform are roughly 90% of the work.

    Results are bit-identical to the same calls made outside the block: the
    cached object is the pre-``fe`` intermediate, and the tail clones before its
    in-place multiply.

    Examples
    --------
    >>> with share_splat():                                  # doctest: +SKIP
    ...     V_a = make_potential(s, px, fe=fe_a, displacements=False)
    ...     V_b = make_potential(s, px, fe=fe_b, displacements=False)

    Notes
    -----
    Scoped and nestable: the previous cache is saved and restored around the
    ``yield``, so nothing leaks between blocks and no cached tensor — which may
    carry an autograd graph — outlives the block.  One cache is shared per
    thread of control; it is not thread-safe.

    Calls with ``displacements=True`` are neither served from nor written to the
    cache, since their splat carries fresh randomness.
    """
    global _SPLAT_SHARE
    previous = _SPLAT_SHARE
    _SPLAT_SHARE = {}
    try:
        yield
    finally:
        _SPLAT_SHARE = previous


# --------------------------------------------------------------------------- #
# the splat
# --------------------------------------------------------------------------- #
def _assign_subslices(zfrac: np.ndarray, subslices: Sequence[float]) -> np.ndarray:
    """Bin fractional depths into subslices.

    Atom ``m`` lands in slice ``i`` iff ``subslices[i-1] <= z_m < subslices[i]``,
    with an implied lower edge of 0 for ``i = 0``.  Atoms at or beyond the last
    boundary wrap to slice 0 — ``subslices`` is assumed to end at 1.0, and a list
    that does not is a caller error whose consequence is this wrap, not a clamp.
    NaN depths also land in slice 0.  Atoms exactly on an internal boundary go to
    the upper slice, so repeated boundaries (``[0.5, 0.5, 1.0]``) leave the
    zero-width slice empty.
    """
    nss = len(subslices)
    edges = np.asarray(subslices, dtype=np.float64).reshape(-1)

    if nss > 1 and not np.all(np.diff(edges) >= 0.0):
        # Non-monotonic boundaries have no searchsorted interpretation; fall back
        # to the literal sequential-bin definition, later bins overwriting.
        out = np.zeros(zfrac.shape, dtype=_int)
        low = 0.0
        for i in range(nss):
            high = float(edges[i])
            out[(zfrac >= low) & (zfrac < high)] = i
            low = high
        return out

    idx = np.searchsorted(edges, zfrac, side="right").astype(_int)
    idx[idx >= nss] = 0
    return idx


def _build_splat_spectrum(
    structure: Any,
    pixels_: tuple,
    subslices: Sequence[float],
    tiling: Sequence[int],
    displacements: bool,
    fractional_occupancy: bool,
    sinc_deconvolution: bool,
    device: torch.device,
    realdtype: torch.dtype,
    generator: Optional[torch.Generator],
    weights: Any,
    weight_power: int,
    elements: list,
) -> torch.Tensor:
    r"""Splat the atoms, transform, and deconvolve the interpolation kernel.

    Returns the ``(M, nss, Ny, Nx//2+1)`` complex spectrum of the
    per-(element, subslice) atomic density, divided by the separable sinc of the
    bilinear splat kernel.  This is the structure-dependent, form-factor-free
    half of :func:`make_potential`, and the object cached by
    :func:`share_splat`.

    Everything is vectorised over atoms **and** over lateral tile replicas: the
    tiles differ only by an integer pixel offset, so they are handled as one
    batched scatter-add rather than a Python loop of four scatter-adds each.
    """
    atoms = structure.atoms
    natoms = atoms.shape[0]
    Ny, Nx = pixels_
    t0, t1 = int(tiling[0]), int(tiling[1])
    ntiles = t0 * t1
    nss = len(subslices)
    nelem = len(elements)
    pp = Nx // 2 + 1

    # ---- host-side per-atom bookkeeping (all O(natoms), done once) ---------- #
    Z = np.asarray(atoms[:, 3], dtype=_int)
    slice_of = _assign_subslices(np.mod(np.asarray(atoms[:, 2], dtype=np.float64), 1.0),
                                subslices)

    # Element index = position in the `elements` list.  Built as a dense lookup
    # table over the (tiny) span of atomic numbers present and applied with one
    # vectorised gather: anything per-atom in Python dominates the whole call at
    # realistic supercell sizes (a generator expression here cost ~60 ms of a
    # ~140 ms call at 5e5 atoms).  The rank-by-sort fallback covers a pathological
    # Z column (garbage or negative values) where the dense table would be huge.
    Z64 = Z.astype(np.int64)
    zlo, zhi = int(Z64.min()), int(Z64.max())
    if zhi - zlo < (1 << 16):
        lut = np.empty(zhi - zlo + 1, dtype=np.int64)
        for i, e in enumerate(elements):
            lut[int(e) - zlo] = i
        elem_of = lut[Z64 - zlo]
    else:
        el = np.asarray(elements, dtype=np.int64)
        rank = np.argsort(el)
        elem_of = rank[np.searchsorted(el[rank], Z64)]

    # Flattened (element, subslice) plane index, times Ny: the row offset of each
    # atom's destination plane in the flattened accumulator.
    plane_row = torch.from_numpy(
        ((elem_of * nss + slice_of.astype(np.int64)) * Ny)
    ).to(device).reshape(1, natoms)

    # ---- per-atom splat mass: occupancy * weight**power --------------------- #
    use_occ = bool(fractional_occupancy) and bool(structure.fractional_occupancy)
    mass: Optional[torch.Tensor] = None
    if use_occ:
        mass = torch.from_numpy(
            np.ascontiguousarray(atoms[:, 4], dtype=_NUMPY_OF_REAL[realdtype])
        ).to(device)
    if weights is not None:
        if isinstance(weights, torch.Tensor):
            w = weights.to(device=device, dtype=realdtype)
        else:
            w = torch.as_tensor(weights, dtype=realdtype, device=device)
        w = w.reshape(-1)
        if weight_power == 1:
            wfac = w
        elif weight_power == 2:
            wfac = w * w
        else:
            wfac = w**weight_power
        mass = wfac if mass is None else mass * wfac
    if mass is not None:
        mass = mass.reshape(1, natoms)

    # ---- thermal displacements (Einstein model), in pixels ------------------ #
    disp: Optional[torch.Tensor] = None
    if displacements:
        # sigma_axis = sqrt(<u^2>) [A] * pixels-per-angstrom, per axis.
        pixperA = np.array(
            [Ny / (float(structure.unitcell[0]) * t0),
             Nx / (float(structure.unitcell[1]) * t1)],
            dtype=np.float64,
        )
        urms_np = (
            np.sqrt(np.asarray(atoms[:, 5], dtype=np.float64))[:, None]
            * pixperA[None, :]
        ).astype(_NUMPY_OF_REAL[realdtype])
        urms = torch.from_numpy(np.ascontiguousarray(urms_np)).to(device)
        disp = torch.randn(
            (ntiles, natoms, 2),
            dtype=realdtype,
            device=device,
            generator=generator,
        )
        disp.mul_(urms)
        if use_occ:
            # Partially occupying atoms sharing a site are one vibrating site and
            # must move together; re-index onto the site representative.
            eqv = _find_equivalent_sites(np.asarray(atoms[:, :3], dtype=np.float64))
            eqv_t = torch.from_numpy(eqv.astype(np.int64)).to(device)
            disp = disp.index_select(1, eqv_t)

    # ---- fractional coordinates, uploaded once in float64 ------------------- #
    # The pixel coordinate arithmetic (offset, divide by tiling, scale to pixels)
    # is done in float64 and only then rounded to the working dtype, so the
    # floor/ceil split is taken from a fully accurate coordinate.  Uploading once
    # and offsetting on the device removes ntiles host->device round trips.
    frac = torch.from_numpy(
        np.ascontiguousarray(atoms[:, :2], dtype=np.float64)
    ).to(device)
    fy = frac[:, 0].reshape(1, natoms)
    fx = frac[:, 1].reshape(1, natoms)

    tile_ids = torch.arange(ntiles, device=device)
    off0 = (tile_ids % t0).to(torch.float64).reshape(ntiles, 1)
    off1 = torch.div(tile_ids, t0, rounding_mode="floor").to(torch.float64).reshape(
        ntiles, 1
    )

    # ---- batched bilinear scatter ------------------------------------------ #
    flat = torch.zeros(nelem * nss * Ny * Nx, dtype=realdtype, device=device)
    block = max(1, _SPLAT_BLOCK // max(1, natoms))

    for lo in range(0, ntiles, block):
        hi = min(lo + block, ntiles)
        # u = ((frac + tile_offset) / tiling) * pixels, in pixels of the tiled grid
        u = (((fy + off0[lo:hi]) / t0) * Ny).to(realdtype)
        v = (((fx + off1[lo:hi]) / t1) * Nx).to(realdtype)
        if disp is not None:
            u = u + disp[lo:hi, :, 0]
            v = v + disp[lo:hi, :, 1]

        # Triangular (hat) kernel: floor gets 1-theta, ceil gets theta, both
        # reduced modulo the grid extent with a non-negative residue so that
        # atoms displaced past the origin wrap correctly.
        fu = torch.floor(u)
        fv = torch.floor(v)
        ty = u - fu
        tx = v - fv
        iy0 = fu.to(torch.int64) % Ny
        iy1 = torch.ceil(u).to(torch.int64) % Ny
        ix0 = fv.to(torch.int64) % Nx
        ix1 = torch.ceil(v).to(torch.int64) % Nx

        # The per-atom mass enters exactly once (folded into the axis-0 pair), so
        # the total splat mass of atom m is o_m * w_m**p, not its square.
        wy0 = 1.0 - ty
        wy1 = ty
        if mass is not None:
            wy0 = wy0 * mass
            wy1 = wy1 * mass
        wx0 = 1.0 - tx
        wx1 = tx

        row0 = plane_row + iy0
        row1 = plane_row + iy1
        for row, wy in ((row0, wy0), (row1, wy1)):
            base = row * Nx
            flat.index_add_(0, (base + ix0).reshape(-1), (wy * wx0).reshape(-1))
            flat.index_add_(0, (base + ix1).reshape(-1), (wy * wx1).reshape(-1))

    splat = flat.view(nelem * nss, Ny, Nx)

    # ---- forward real FFT --------------------------------------------------- #
    nbatch = nelem * nss
    if device.type == "cuda" and nbatch * Ny * Nx > _FFT_PLAN_ELEMENTS:
        # cuFFT cannot plan a batch this large; transform in place into a
        # preallocated output rather than building a list and concatenating.
        chunk = max(1, _FFT_PLAN_ELEMENTS // (Ny * Nx))
        P = torch.empty(
            (nbatch, Ny, pp), dtype=_COMPLEX_OF_REAL[realdtype], device=device
        )
        for lo in range(0, nbatch, chunk):
            P[lo : lo + chunk] = torch.fft.rfft2(splat[lo : lo + chunk])
    else:
        P = torch.fft.rfft2(splat)
    P = P.view(nelem, nss, Ny, pp)

    # ---- deconvolve the interpolation kernel -------------------------------- #
    if sinc_deconvolution:
        P = P * _inverse_sinc_half_plane(Ny, Nx, realdtype, device)

    return P


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def make_potential(
    structure,
    pixels,
    subslices=[1.0],
    tiling=(1, 1),
    displacements=True,
    fractional_occupancy=True,
    sinc_deconvolution=True,
    bandwidthlimit=0.2,
    fe=None,
    device=None,
    dtype=torch.float32,
    seed=None,
    weights=None,
    weight_power=1,
    *,
    cache_fe: bool = True,
) -> torch.Tensor:
    r"""Projected electrostatic potential of a structure on a pixel grid.

    Evaluates, for each depth slice ``s`` and grid node ``n``,

    .. math::

        V_s(\mathbf r_n) \;=\; \frac{1}{L_y L_x} \sum_{\mathbf k} M(\mathbf k)
            \Big[\sum_{Z} f_e^{Z}(\mathbf g_{\mathbf k})\,
                 \tilde S_{Z,s}(\mathbf k)\Big]\,
            e^{+2\pi i (k_y n_y/N_y + k_x n_x/N_x)},

    where :math:`\tilde S_{Z,s}` is the sinc-deconvolved DFT of the bilinear
    splat of the species-``Z`` atoms assigned to slice ``s``, :math:`f_e^Z` is
    the electron scattering factor in V Å³, :math:`M` is the soft band-limit
    mask, :math:`\mathbf g_{\mathbf k} = (k_y/L_y, k_x/L_x)` in Å⁻¹, and
    :math:`L_i` are the tiled supercell edges.  Equivalently, and this is what it
    is *for*: the convolution of the slice's atomic delta comb with the
    per-species projected atomic potential.  See the module docstring for the
    physics and the references.

    Parameters
    ----------
    structure : Structure
        Must expose ``unitcell`` — the length-3 **orthorhombic** edge vector in
        Å — and ``atoms``, an ``(natoms, 6)`` float array whose columns are
        ``[x, y, z, Z, occupancy, <u^2>]``: ``x, y, z`` **fractional**
        coordinates nominally in ``[0, 1)``, ``Z`` the atomic number, occupancy
        in ``(0, 1]``, and ``<u^2>`` the one-dimensional mean-square thermal
        displacement in Å² (**not** the Debye-Waller ``B``; ``B = 8 pi^2 <u^2>``).
        Also ``fractional_occupancy``, a bool.
    pixels : (2,) sequence of int
        Grid ``(Ny, Nx)`` covering the *tiled* supercell.  A bare int is not
        accepted.
    subslices : float or (nss,) array_like, optional
        Upper fractional depths of the slices along ``c``, non-decreasing, ending
        at 1.0.  A bare float is treated as a single slice.  Default ``[1.0]``.
    tiling : (2,) sequence of int, optional
        Lateral repetition ``(t0, t1)`` of the unit cell.  Default ``(1, 1)``.
    displacements : bool, optional
        Draw per-atom, per-replica Gaussian lateral displacements with
        ``sigma = sqrt(<u^2>)`` (Einstein frozen-phonon model).  ``False`` gives
        the deterministic static-lattice potential.  Default True.
    fractional_occupancy : bool, optional
        Honour the occupancy column, and make atoms sharing a site vibrate
        together.  Inert when the structure is fully occupied.  Default True.
    sinc_deconvolution : bool, optional
        Divide out one factor of the splat kernel's sinc per axis.  Default True.
    bandwidthlimit : float or None, optional
        Width of the erf taper at the Nyquist radius; ``None`` disables the mask.
        Default 0.2.
    fe : (M, Ny, Nx) numpy.ndarray or None, optional
        Precomputed scattering factors in V Å³, corner-origin (unshifted) order,
        **indexed along axis 0 in this function's element order**
        ``list(set(atoms[:, 3].astype(int32)))`` — see Notes.  Only the first
        ``Nx//2+1`` columns are read.  ``None`` (default) evaluates them.
    device : str, torch.device or None, optional
        ``None`` means CUDA if available, **not** CPU.
    dtype : torch.dtype, optional
        ``float32``/``complex64`` (default) or ``float64``/``complex128``; the
        output always carries the corresponding *real* dtype.
    seed : int or None, optional
        Seed for the displacement draw.  Seeds a private generator, so the
        caller's global RNG stream is untouched.
    weights : (natoms,) array_like or torch.Tensor, optional
        Per-atom weight multiplying the splat mass; may require grad.
    weight_power : int, optional
        Exponent ``p`` on ``weights``.  ``1`` (default) for the elastic
        potential, which is linear in the scattering amplitude; ``2`` for
        emission/TDS potentials, whose form factor is a product of two
        amplitudes and therefore scales as ``w^2``.

    Returns
    -------
    torch.Tensor
        ``(nss, Ny, Nx)`` real potential in volt-angstrom, on ``device``, with
        the real dtype derived from ``dtype``.  Multiply by the interaction
        constant ``sigma(E)`` in rad/(V Å) to get the multislice phase shift.

    Notes
    -----
    **Element order is an external contract.**  Species are enumerated as
    ``list(set(...))`` over the int32 atomic numbers, i.e. CPython's small-int
    hash order, which is *not* sorted: ``{8, 42, 16, 1, 92}`` enumerates as
    ``[1, 8, 42, 16, 92]``.  Callers that precompute ``fe`` build it against that
    order; replacing it with ``sorted()`` would silently permute form factors
    between species.

    **Integral invariant.**  ``sum(V) * dy * dx`` equals
    ``sum_m occ_m * w_m**p * fe_{Z_m}(0)`` over all atoms including tile
    replicas.  This is the cheapest end-to-end check of the normalisation, and
    the gradient with respect to ``w_m`` is correspondingly
    ``fe_{Z_m}(0)/(dy*dx)``.

    **Differentiability.**  Gradients flow through ``weights`` only; positions
    and the slice/element index maps are non-differentiable buffers.

    **Determinism.**  On CUDA, scatter-add into a float buffer accumulates in
    nondeterministic order, so two identical calls can differ at the ULP level
    even with ``displacements=False``.

    Examples
    --------
    >>> V = make_potential(structure, (256, 256), subslices=[0.5, 1.0],
    ...                    displacements=False, device="cpu")  # doctest: +SKIP
    >>> V.shape                                                # doctest: +SKIP
    torch.Size([2, 256, 256])
    """
    device = _get_device(device)
    realdtype = _complex_to_real_dtype_torch(dtype)
    pixels_ = tuple(int(p) for p in pixels)
    subslices = _ensure_array(subslices)
    nss = len(subslices)
    Ny, Nx = pixels_

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    # Species enumeration.  See Notes: hash order, not sorted, by contract.
    elements = list(set(np.asarray(structure.atoms[:, 3], dtype=_int)))

    # A vacuum tile has no species at all.  Short-circuit: the element-batched
    # real FFT would otherwise be a zero-size batch, which cuFFT rejects, and the
    # projected potential of vacuum is exactly the empty element sum.
    if len(elements) == 0:
        return torch.zeros((nss, Ny, Nx), dtype=realdtype, device=device)

    gsize = (
        float(structure.unitcell[0]) * int(tiling[0]),
        float(structure.unitcell[1]) * int(tiling[1]),
    )

    # ---- scoped reuse of the structure-dependent half ----------------------- #
    share = _SPLAT_SHARE
    share_key = None
    if share is not None and not displacements:
        share_key = (
            id(structure),
            pixels_,
            tuple(float(s) for s in subslices),
            tuple(int(t) for t in tiling),
            # With weights=None the weight multiply never happens, so the power
            # cannot influence the splat and must not split the key.
            (int(weight_power) if weights is not None else None),
            id(weights),
            bool(fractional_occupancy),
            bool(sinc_deconvolution),
            str(device),
            str(realdtype),
        )
        entry = share.get(share_key)
        if entry is not None:
            P = entry[0]
            return _apply_form_factors(
                P, fe, elements, pixels_, gsize, len(elements), pixels_,
                structure, tiling, bandwidthlimit, device, copy=True,
                cache_fe=cache_fe,
            )

    P = _build_splat_spectrum(
        structure,
        pixels_,
        subslices,
        tiling,
        displacements,
        fractional_occupancy,
        sinc_deconvolution,
        device,
        realdtype,
        generator,
        weights,
        weight_power,
        elements,
    )

    copy = False
    if share_key is not None:
        # Hold strong references to the keyed objects so their ids cannot be
        # recycled by a garbage collection inside the block.
        share[share_key] = (P, structure, weights)
        copy = True

    return _apply_form_factors(
        P, fe, elements, pixels_, gsize, len(elements), pixels_,
        structure, tiling, bandwidthlimit, device, copy=copy, cache_fe=cache_fe,
    )
