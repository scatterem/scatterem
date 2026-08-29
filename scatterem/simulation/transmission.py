"""Multislice transmission functions :math:`T_j = \\exp(i\\sigma V_j)`.

Written from the physics -- the equations below and the references they cite.

Physics
-------
A fast electron traversing a thin slice of material accumulates a phase
proportional to the electrostatic potential projected through that slice.
Writing :math:`V_j(x, y)` for the potential of slice *j* projected along the
beam direction (units V*A), the slice acts as a pure multiplicative
transmission function

.. math::

    T_j(x, y) = \\exp\\bigl(i\\,\\sigma\\,V_j(x, y)\\bigr),
    \\qquad
    V_j(x, y) = \\int_{\\text{slice }j} V(x, y, z)\\, dz ,

with :math:`\\sigma` the relativistic electron interaction parameter in
rad/(V*A).  This is the transmission half of the transmit--propagate cycle of
the multislice algorithm (Cowley & Moodie, *Acta Cryst.* **10** (1957) 609;
Kirkland, *Advanced Computing in Electron Microscopy*, 2nd ed., Ch. 6).

Thermal diffuse scattering may be folded in as an *absorptive* (imaginary)
addition to the potential, in the Hall--Hirsch sense (P. M. Hall & P. B.
Hirsch, *Proc. R. Soc. A* **286** (1965) 158; D. M. Bird & Q. A. King,
*Acta Cryst.* **A46** (1990) 202).  With :math:`V^{\\text{abs}}_j` the
absorptive potential *per unit thickness* and :math:`\\Delta z` the slice
thickness,

.. math::

    V^{\\text{tot}}_j = V^{\\text{DW}}_j + i\\,\\Delta z\\, V^{\\text{abs}}_j ,
    \\qquad
    T_j = e^{\\,i\\sigma V^{\\text{DW}}_j}\\;
          e^{-\\sigma\\,\\Delta z\\, V^{\\text{abs}}_j} ,

i.e. a pure phase multiplied by a real attenuation :math:`\\le 1`: the
Debye--Waller-smeared elastic channel loses exactly the flux that TDS removes
from it.

Because :math:`\\exp(i\\sigma V)` is a *nonlinear* function of :math:`V`, it
carries energy to spatial frequencies well beyond the band limit of the
potential itself.  On a periodic FFT grid that energy aliases back into the
physical band, so the transmission function is band-limited by a hard aperture
in reciprocal space at a fraction ``bandwidth_limit`` (2/3 by default) of the
Nyquist frequency -- Kirkland's "2/3 rule", which places the second-order
aliasing product of a band-limited wave outside the retained band
(Kirkland, Ch. 6.6).  Band-limiting necessarily breaks exact unitarity:
:math:`|T| = 1` holds before the aperture, and to within ~1% after it.

Public API
----------
:func:`make_transmission_functions`
    Structure -> band-limited complex transmission stack.
:func:`projected_potential_to_object`
    Projected potential -> scatterem reconstruction-object convention
    (:math:`O = \\sigma V`, no exponential, no band limit).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .potentials import make_potential
from .scattering_factors import interaction_constant

__all__ = ["make_transmission_functions", "projected_potential_to_object"]


# --------------------------------------------------------------------------- #
# Loop-invariants: sigma and the band-limit aperture are pure functions of a
# handful of scalars, and both sit inside per-slice / per-frozen-phonon loops in
# the callers.  Memoise them so a call in a loop pays for neither.
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _sigma(eV: float) -> float:
    """Interaction parameter sigma(E) in rad/(V*A), memoised on the energy.

    Pure function of the accelerating voltage, so caching is exact rather than
    approximate.  See :func:`scatterem.simulation.scattering_factors.interaction_constant`
    for the closed form (Kirkland Eq. 5.6).
    """
    return interaction_constant(eV)


@lru_cache(maxsize=64)
def _antialias_reject_mask(
    ny: int,
    nx: int,
    limit_y: float,
    limit_x: float,
    device: torch.device,
) -> Tensor:
    """Boolean mask of the frequencies **removed** by the antialias aperture.

    The aperture is the ellipse

    .. math::

        \\left(\\frac{f_y}{\\tfrac{1}{2} L_y}\\right)^2
        + \\left(\\frac{f_x}{\\tfrac{1}{2} L_x}\\right)^2 < 1 ,

    i.e. its semi-axis along each direction is ``limit_i`` times that axis'
    own Nyquist frequency (1/2 cycle/pixel).  Frequencies are the unit-sample
    ``fftfreq`` values ``f[k] = k/N`` for ``k < N/2`` and ``(k-N)/N`` otherwise,
    in **cycles per pixel** -- there is no dependence on the physical cell size,
    so on an anisotropic grid the aperture is isotropic in units of each axis'
    Nyquist and therefore elliptical in A^-1.

    The comparison is strict, so a frequency landing exactly on the ellipse is
    discarded.  The frequency grid is built in float64 and only the boolean
    outcome is kept, which removes any float32 knife-edge sensitivity for grid
    sizes that admit an exact tie.

    Returns the *complement* (the reject set) because that is what
    ``masked_fill_`` wants, and because a boolean mask costs a quarter of the
    memory of a complex64 multiplicand and needs no arithmetic.

    Notes
    -----
    The returned tensor is shared between calls and must never be mutated.
    """
    # Scale by the single factor ``2/limit`` rather than multiplying by 2 and
    # then dividing by ``limit``.  The two are equal in exact arithmetic but not
    # in binary floating point, and on a frequency that lands exactly on the
    # contour the difference decides which side of the strict inequality it
    # falls.  ``_grid._nyquist_normalised_axis`` associates it this way, so
    # doing the same keeps the two apertures in this package bit-identical.
    fy = torch.fft.fftfreq(ny, dtype=torch.float64, device=device)
    fx = torch.fft.fftfreq(nx, dtype=torch.float64, device=device)
    ry = fy.mul_(2.0 / limit_y).square_()
    rx = fx.mul_(2.0 / limit_x).square_()
    return (ry[:, None] + rx[None, :]) >= 1.0


def _resolve_limits(bandwidth_limit) -> Tuple[float, float]:
    """Normalise ``bandwidth_limit`` to a ``(limit_y, limit_x)`` pair of floats.

    A scalar is broadcast to both axes; a sequence contributes its **last two**
    entries as ``[limit_y, limit_x]``.

    Anything ``float()`` accepts counts as a scalar -- Python numbers, numpy
    scalars of any width (``np.float32``/``np.int64`` are *not* instances of
    ``float``/``int``), and 0-d arrays and tensors -- so the scalar path is
    selected by behaviour rather than by type membership.
    """
    try:
        limit = float(bandwidth_limit)
    except (TypeError, ValueError):
        pass
    else:
        return limit, limit
    seq = tuple(bandwidth_limit)
    return float(seq[-2]), float(seq[-1])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def projected_potential_to_object(V, eV):
    """Convert a projected potential to scatterem's reconstruction-object convention.

    scatterem's ptychographic forward model stores the object already scaled by
    the interaction parameter and applies the slice as ``exp(1j * object)``.
    This adapter performs exactly that one scaling,

    .. math::

        O_j(\\mathbf{r}) = \\sigma(E)\\, V_j(\\mathbf{r}) ,

    turning a potential in V*A into a phase in radians.  It deliberately does
    **nothing else**: it does not exponentiate, does not band-limit, does not
    move devices, does not change dtype and does not copy metadata.  Contrast
    :func:`make_transmission_functions`, which pre-applies sigma *and*
    band-limits, returning ``IFFT(M * FFT(exp(i*sigma*V)))``.

    Parameters
    ----------
    V : array_like
        Projected electrostatic potential in V*A, canonically of shape
        ``(nsubslices, Ny, Nx)`` as returned by
        :func:`scatterem.simulation.potentials.make_potential`.  Any real
        torch tensor or numpy array works; the operation is duck-typed.
    eV : float
        Accelerating voltage in volts (equivalently the electron kinetic
        energy in eV).  Must be positive.

    Returns
    -------
    array_like
        ``sigma * V``, a phase in radians, with exactly the same shape, dtype,
        device and layout as ``V``.  ``sigma`` is a Python float, so weak
        scalar promotion keeps a float32 input float32 under both NEP-50 and
        legacy numpy casting rules, and a float32 tensor float32.  Autograd
        transparent: if ``V`` requires grad so does the result, with
        ``dO/dV = sigma``.  ``V`` is never mutated.

    Raises
    ------
    ValueError
        If ``eV`` is not strictly positive (the interaction parameter divides
        by the energy, so a non-positive value would silently poison the whole
        array with inf/nan).

    References
    ----------
    Kirkland, *Advanced Computing in Electron Microscopy*, 2nd ed., Eq. 5.6.
    """
    eV = float(eV)
    if not eV > 0.0:
        raise ValueError(f"eV must be a positive accelerating voltage, got {eV!r}")
    return _sigma(eV) * V


def make_transmission_functions(
    structure,
    pixels,
    eV,
    subslices: Sequence[float] = (1.0,),
    tiling: Sequence[int] = (1, 1),
    fe: Optional[Tensor] = None,
    displacements: bool = True,
    fftout: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    fractional_occupancy: bool = True,
    seed: Optional[int] = None,
    bandwidth_limit: Optional[Union[float, Sequence[float]]] = 2 / 3,
    V_abs: Optional[Tensor] = None,
    dz: Optional[float] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Build the band-limited multislice transmission functions of a structure.

    Three stages:

    1. **Potential.** Delegate to
       :func:`scatterem.simulation.potentials.make_potential` for the real
       projected electrostatic potential ``V`` of shape
       ``(nsubslices, Ny, Nx)`` in V*A.  (That builder applies its own,
       separate soft antialias filter to the potential; ``bandwidth_limit``
       here is the later, harder aperture applied to the *transmission
       function*, and is not forwarded.)

    2. **Complex exponential.**  Pure-elastic path,

       .. math:: T_j(\\mathbf{r}) = \\exp\\bigl(i\\,\\sigma\\,V_j(\\mathbf{r})\\bigr),

       so :math:`|T| = 1` exactly.  With ``V_abs`` supplied, the optical
       potential picks up an imaginary part
       :math:`V^{\\text{tot}} = V^{\\text{DW}} + i\\,\\Delta z\\,V^{\\text{abs}}` and

       .. math::

           T_j(\\mathbf{r}) = e^{\\,i\\sigma V_j(\\mathbf{r})}\\,
                              e^{-\\sigma\\,\\Delta z\\,V^{\\text{abs}}_j(\\mathbf{r})},

       a pure phase times a real attenuation: the Debye--Waller absorptive
       channel.  Pass Debye--Waller-damped scattering factors through ``fe`` to
       make the elastic ``V`` the DW-smeared :math:`V^{\\text{DW}}`; the
       absorptive sink itself arrives ready-made in ``V_abs``.

    3. **Band limit and output space.**  Forward-FFT over the last two axes,
       zero everything outside the antialias ellipse, and either return that
       (``fftout=True``) or transform back (``fftout=False``, the default).
       In one line, for ``fftout=False``::

           T = IFFT2( M * FFT2( exp(i*sigma*V - sigma*dz*V_abs) ) )

    Parameters
    ----------
    structure : Structure
        scatterem structure; needs ``.unitcell`` and an ``.atoms`` table with
        columns ``[x, y, z, Z, occupancy, mean-square displacement]``.
    pixels : sequence of int
        Output grid ``(Ny, Nx)``.  Must be a length-2 iterable; a bare int is
        not accepted.
    eV : float
        Accelerating voltage in volts.  Must be positive.
    subslices : sequence of float, default ``(1.0,)``
        Fractional depths at which each slice **ends**, in fractional
        (untiled) unit-cell coordinates, increasing and ending at 1.0 to tile
        the cell exactly -- e.g. ``(0.5, 1.0)`` for two equal slices.  The
        number of output slices is ``len(subslices)``.  Not validated.
    tiling : sequence of int, default ``(1, 1)``
        Lateral repeat ``(ty, tx)`` of the unit cell.
    fe : Tensor, optional
        Precomputed electron scattering factors.  Pass to avoid recomputation,
        and as the mechanism for injecting Debye--Waller-damped factors.
    displacements : bool, default True
        Apply random frozen-phonon thermal displacements.  With ``seed=None``
        this makes the result nondeterministic.
    fftout : bool, default False
        Return the reciprocal-space (band-limited) transmission instead of the
        real-space one.
    dtype : torch.dtype, optional
        Working precision.  ``None`` is exactly equivalent to
        ``torch.float32``.  A complex dtype is accepted and maps to the
        matching real precision internally.
    device : str or torch.device, optional
        Compute device.  ``None`` resolves to CUDA when available, else CPU --
        so on a CUDA machine the default return is a **GPU** tensor.
    fractional_occupancy : bool, default True
        Weight each site by its occupancy.
    seed : int, optional
        Seeds the RNG used for the thermal displacements.  Note this seeds the
        *global* torch RNG, so a seeded call is reproducible but perturbs any
        subsequent unseeded torch randomness in the process.
    bandwidth_limit : float or sequence of float or None, default 2/3
        Antialias aperture radius as a fraction of the Nyquist frequency.  A
        scalar applies to both axes; a sequence contributes its last two
        entries as ``[limit_y, limit_x]``.  ``None`` disables the aperture
        entirely (and then, for ``fftout=False``, no transform is performed at
        all, so the result is the exponential exactly).
    V_abs : Tensor or array_like, optional
        Absorptive potential **per unit thickness** (units V), of shape
        ``(nsubslices, Ny, Nx)`` or anything broadcastable against it --
        ``(Ny, Nx)`` broadcasts over all slices.  Moved to the potential's
        device and cast to the real working precision.  Must be non-negative
        for the exponential to damp rather than amplify; this is not validated.
    dz : float, optional
        Slice thickness in Angstrom, restoring the V*A scaling of ``V_abs``.
        Required in spirit whenever ``V_abs`` is given; if omitted it is
        silently taken as 1.0 A.
    weights : Tensor, optional
        Per-atom scalar weights forwarded to the potential splat at power 1.
        May carry autograd gradients; the whole path stays differentiable.

    Returns
    -------
    Tensor
        Complex tensor of shape ``(len(subslices), Ny, Nx)``.  The complex
        dtype is the partner of the real working precision: float32 (or
        ``None``, or ``torch.complex64``) -> ``complex64``; float64 ->
        ``complex128``.  Axis order is ``(slice, y, x)``, row-major, with
        ``pixels[0]`` the slow axis.

        With ``fftout=True`` the array is in standard **corner-origin** FFT
        order -- element ``[j, 0, 0]`` is DC, positive frequencies first,
        negative frequencies in the upper half.  There is no ``fftshift``
        anywhere; a consumer wanting a centred q-grid must shift it itself.

    Raises
    ------
    ValueError
        If ``eV`` is not strictly positive.

    Notes
    -----
    The forward transform is unnormalised (``norm="backward"``) and the inverse
    carries the full ``1/(Ny*Nx)``, so Parseval reads
    ``sum_r |T|^2 = (1/(Ny*Nx)) * sum_q |T_hat|^2`` and the DC element equals
    ``sum_r T(r)``.  Do not switch to ``norm="ortho"``; it would rescale the
    ``fftout=True`` return by ``sqrt(Ny*Nx)``.

    ``|T| = 1`` holds only on the elastic path and only *before* the band
    limit.  After band-limiting it departs from unity by of order 1% for a
    realistic potential, and with ``V_abs`` it is strictly below 1 by
    construction.  A vacuum (atom-free) tile gives ``T = 1`` everywhere before
    the aperture, but the aperture turns that into the inverse transform of a
    masked delta -- a sinc-like kernel with DC = 1, not the constant 1.  Tests
    asserting unitarity or vacuum-identity must set ``bandwidth_limit=None``.

    References
    ----------
    Cowley & Moodie, *Acta Cryst.* **10** (1957) 609.
    Kirkland, *Advanced Computing in Electron Microscopy*, 2nd ed., Ch. 6
    (multislice; 2/3 antialias rule in Ch. 6.6) and Eq. 5.6 (sigma).
    Hall & Hirsch, *Proc. R. Soc. A* **286** (1965) 158; Bird & King,
    *Acta Cryst.* **A46** (1990) 202 (absorptive potentials).
    """
    eV = float(eV)
    if not eV > 0.0:
        raise ValueError(f"eV must be a positive accelerating voltage, got {eV!r}")

    # sigma is a loop-invariant scalar (rad/(V*A)); memoised on the energy.
    sigma = _sigma(eV)

    # ---- 1. projected electrostatic potential, real, (nsubslices, Ny, Nx) --- #
    V = make_potential(
        structure,
        pixels,
        subslices=list(subslices),
        tiling=tiling,
        fe=fe,
        displacements=displacements,
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
        fractional_occupancy=fractional_occupancy,
        seed=seed,
        weights=weights,
    )

    # ---- 2. complex exponential of the optical potential -------------------- #
    #
    # T = exp(i*sigma*V) [* exp(-sigma*dz*V_abs)] is built directly in polar
    # form from a REAL phase (and a REAL magnitude).  Forming the complex
    # argument i*sigma*V and calling a general complex exp would compute
    # exp(0) over the whole array for nothing and double the memory traffic;
    # torch.polar is one fused kernel over real inputs.
    phase = V * sigma  # radians

    if V_abs is None:
        # Pure-elastic: unit magnitude, so |T| = 1 exactly at this point.
        T = torch.polar(torch.ones_like(phase), phase)
    else:
        # exp(-sigma*dz*V_abs).  The two scalar factors are fused into one
        # Python float so the array is traversed once, and the attenuation
        # stays REAL (an all-zero imaginary half would be pure waste).
        V_abs_t = torch.as_tensor(V_abs, dtype=V.dtype, device=V.device)
        magnitude = torch.exp(V_abs_t * -(sigma * (1.0 if dz is None else float(dz))))
        T = torch.polar(magnitude, phase)  # broadcasts (Ny, Nx) over the slice axis

    # ---- 3. band limit and output space ------------------------------------ #
    if bandwidth_limit is None and not fftout:
        # No aperture and no reciprocal-space output: a forward and inverse
        # transform would provably cancel.  Skipping them is both faster and
        # strictly more accurate (no float32 FFT round-trip noise).
        return T

    T_hat = torch.fft.fft2(T, dim=(-2, -1))  # unnormalised, corner origin

    if bandwidth_limit is not None:
        reject = _antialias_reject_mask(
            T_hat.shape[-2], T_hat.shape[-1], *_resolve_limits(bandwidth_limit), T_hat.device
        )
        # T_hat is a freshly allocated temporary nobody else holds, so the
        # zero-fill can be done in place -- unless autograd needs the graph.
        # masked_fill touches only the ~65% of pixels being killed and needs no
        # float mask array at all.
        if T_hat.requires_grad:
            T_hat = T_hat.masked_fill(reject, 0)
        else:
            T_hat.masked_fill_(reject, 0)

    return T_hat if fftout else torch.fft.ifft2(T_hat, dim=(-2, -1))
