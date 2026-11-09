"""Sliced projected potential of a periodic field of view (Fourier route).

Literature: the Fourier series of the projected potential of a periodic
cell and the slices of multislice follow Kirkland (2020, *Advanced Computing
in Electron Microscopy*, 3rd ed., ch. 5 and 6); thermal draws follow Loane,
Xu and Silcox (1991) and Kirkland (2020, ch. 7). The atomic form spectra are
those of Lobato and Van Dyck (2014), Acta Cryst. A70, 636-649. The atoms go
on the grid with the cubic B-spline charge assignment, an interlaced grid
and the removal of the assignment smoothing of Hockney and Eastwood (1988),
*Computer Simulation Using Particles*, ch. 5 and 7. Written for scatterem
from this literature.

Units: lengths [Angstrom], frequencies [1/Angstrom], form spectra
[Volt Angstrom^3], projected potentials [Volt Angstrom], ``<u^2>``
[Angstrom^2].
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ._bspline_deposit import layout_factors, layout_spectrum
from ._stack_checks import (
    StackRequest,
    check_stack_request,
    resolve_stack_device,
    scale_values,
)
from .atom_amplitudes import potential_spectra
from .atom_set import FULL_OCCUPANCY_TOLERANCE, AtomSet, has_partial_occupancy
from .axial_shares import axial_shares, check_spread
from .beam_partition import BeamPartition, species_order
from .periodic_sampling import PeriodicGrid
from .potential_rolloff import DEFAULT_ROLLOFF, check_rolloff, rolloff_mask
from .row_replacements import RowReplacements, atom_data
from .spectrum_sweep import active_ledger, content_digest
from .thermal_jitter import (
    ThermalJitter,
    check_msd_derivative,
    colocated_rows,
    group_msd,
    offsets_from_data,
)

__all__ = [
    "StackRequest",
    "check_stack_request",
    "potential_stack",
    "resolve_stack_device",
]

_log = logging.getLogger(__name__)


def _check_switches(partition, edge_snap, ignore_fill, jitter, overrides, rows):
    if not isinstance(partition, BeamPartition):
        raise TypeError(f"partition: must be a BeamPartition; got {partition!r}")
    for name, value in (
        ("edge_snap", edge_snap),
        ("ignore_site_fraction", ignore_fill),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name}: must be a bool; got {value!r}")
    if jitter is not None and not isinstance(jitter, ThermalJitter):
        raise TypeError(f"jitter: must be None or a ThermalJitter; got {jitter!r}")
    if overrides is not None:
        if not isinstance(overrides, RowReplacements):
            raise TypeError(f"overrides: must be a RowReplacements; got {overrides!r}")
        overrides.checked_for(rows)


def _check_form(form_spectra, count: int, points) -> None:
    if form_spectra is None:
        return
    if isinstance(form_spectra, torch.Tensor):
        if form_spectra.is_complex() or form_spectra.dtype == torch.bool:
            raise TypeError(f"form_spectra: must be real; got {form_spectra.dtype}")
        plain = form_spectra.detach()
    else:
        arr = np.asarray(form_spectra)
        if arr.dtype.kind not in "iuf":
            raise TypeError(f"form_spectra: must be real; got dtype {arr.dtype}")
        plain = torch.from_numpy(arr.astype(np.float64))
    want = (count, *points)
    if tuple(plain.shape) != want:
        raise ValueError(
            f"form_spectra: must have shape {want}; got {tuple(plain.shape)}"
        )
    if not bool(torch.isfinite(plain).all()):
        raise ValueError("form_spectra: all values must be finite")


def _check_combinations(spread, partition, jitter, overrides, weighting):
    """Refuse combinations without a correct value or gradient."""
    if spread != "binned" and not partition.covers_cell:
        raise ValueError(
            f"beam_spread: {spread!r} needs a partition that covers the cell"
        )
    along_beam = jitter is not None and jitter.moves_along_beam
    if spread == "smeared" and along_beam:
        raise ValueError(
            "beam_spread: 'smeared' already holds the Debye-Waller smearing "
            "along the beam; ThermalJitter(transverse_only=False) would count "
            "it twice"
        )
    o = overrides or RowReplacements()
    if spread == "binned" and o.beam_fraction is not None:
        if o.beam_fraction.requires_grad:
            raise ValueError(
                "overrides.beam_fraction: requires grad, but the 'binned' share is "
                "piecewise constant; use beam_spread='tent' or 'smeared'"
            )
    msd_grad = o.msd is not None and o.msd.requires_grad
    if spread == "binned" and along_beam and msd_grad:
        raise ValueError(
            "overrides.msd: requires grad, but with 'binned' and draws along the "
            "beam the slice changes in steps; the gradient would miss that part"
        )
    if o.site_fraction is not None:
        if not weighting:
            raise ValueError(
                "overrides.site_fraction: given with ignore_site_fraction=True; "
                "the build would ignore it"
            )
        values = o.site_fraction.detach().cpu().numpy()
        if o.site_fraction.requires_grad and not has_partial_occupancy(values):
            raise ValueError(
                "overrides.site_fraction: requires grad, but no value is below "
                f"1 - {FULL_OCCUPANCY_TOLERANCE}; site fractions are then not in "
                "use and the gradient would be zero"
            )


def _prefactors(data, atom_scale, squared, weighting, device) -> torch.Tensor:
    """Return ``a_j = o_j w_j`` (N,) float64 with the graph of the inputs."""
    fill = data["site_fraction"]
    if weighting and has_partial_occupancy(fill.detach().cpu().numpy()):
        a = fill
    else:
        a = torch.ones_like(fill.detach())
    if atom_scale is not None:
        c = scale_values(atom_scale, fill.shape[0])
        if isinstance(c, np.ndarray):
            c = torch.from_numpy(c)
        c = c.to(device=device, dtype=torch.float64)
        a = a * (c * c if squared else c)
    return a


def _copies(data, sides, replicas, jitter, need_groups, device):
    """Return positions, beam coordinates, group msd and atom index per copy."""
    n = data["fractional"].shape[0]
    c_y, c_x = replicas
    u_g_rows = None
    if jitter is not None:
        offsets, labels, u_g = offsets_from_data(data, sides, replicas, jitter, device)
        u_g_rows = u_g[labels]
    else:
        offsets = torch.zeros((c_y, c_x, n, 3), dtype=torch.float64, device=device)
        if need_groups:
            labels_np = colocated_rows(data["fractional"].detach().cpu().numpy(), sides)
            labels = torch.from_numpy(labels_np).to(device)
            u_g = group_msd(data["msd"], labels, int(labels_np.max()) + 1)
            check_msd_derivative(data["msd"], u_g)
            u_g_rows = u_g[labels]
    f = data["fractional"]
    shift_y = torch.arange(c_y, dtype=torch.float64, device=device)[:, None, None]
    shift_x = torch.arange(c_x, dtype=torch.float64, device=device)[None, :, None]
    y = (f[None, None, :, 0] + shift_y) * float(sides[0]) + offsets[..., 0]
    x = (f[None, None, :, 1] + shift_x) * float(sides[1]) + offsets[..., 1]
    z = f[None, None, :, 2] + offsets[..., 2] / float(sides[2])
    z = z.expand(c_y, c_x, n)
    atom = torch.arange(n, device=device).repeat(c_y * c_x)
    msd = None if u_g_rows is None else u_g_rows.repeat(c_y * c_x)
    return y.reshape(-1), x.reshape(-1), z.reshape(-1), msd, atom


def _spectra_half(form_spectra, species, req: StackRequest, rolloff, device):
    """Return ``Fsym_e T / (h_y h_x)`` per element, half layout, build dtype."""
    ny, nx = req.points
    if form_spectra is None:
        grid = PeriodicGrid(req.points, req.lengths)
        form = torch.from_numpy(potential_spectra(grid, species))
    elif isinstance(form_spectra, torch.Tensor):
        form = form_spectra.to(device=device, dtype=torch.float64)
    else:
        form = torch.from_numpy(np.asarray(form_spectra, dtype=np.float64))
    home = form.device
    half = nx // 2 + 1
    flip_y = torch.remainder(-torch.arange(ny, device=home), ny)
    flip_x = torch.remainder(-torch.arange(half, device=home), nx)
    t_half = rolloff_mask(
        req.points, rolloff, half_layout=True, device=home, dtype=torch.float64
    )
    cell = (req.lengths[0] / ny) * (req.lengths[1] / nx)
    out = []
    for row in form:
        sym = (row[:, :half] + row[flip_y][:, flip_x]) / 2.0
        out.append((sym * t_half / cell).to(device=device, dtype=req.real_dtype))
    return out


def _slice_layouts(copies, shares, prefactor, elem, count_e, req, factors):
    """Yield, per slice, the list of layout spectra per element (or None)."""
    y, x, _, _, atom = copies
    ny, nx = req.points
    hy, hx = req.lengths[0] / ny, req.lengths[1] / nx
    slab = shares.slab
    bounds = torch.searchsorted(
        slab, torch.arange(factors[2] + 1, device=slab.device)
    ).tolist()
    for s in range(factors[2]):
        lo, hi = bounds[s], bounds[s + 1]
        item = shares.item[lo:hi]
        rows = atom[item]
        weight = shares.share[lo:hi] * prefactor[rows]
        kinds = elem[rows]
        layouts = []
        for e in range(count_e):
            pick = kinds == e
            if not weight.requires_grad:
                pick = pick & (weight != 0)
            sel = torch.nonzero(pick).reshape(-1)
            if sel.numel() == 0:
                layouts.append(None)
                continue
            i = item[sel]
            layouts.append(
                layout_spectrum(
                    y[i] / hy,
                    x[i] / hx,
                    weight[sel],
                    req.points,
                    req.real_dtype,
                    factors[:2],
                )
            )
        yield layouts


def _synthesis(layouts, spectra, points, dtype, device) -> torch.Tensor:
    """Return ``irfft2(sum_e Fh_e D_e)`` for one slice."""
    total = None
    for f_half, d in zip(spectra, layouts, strict=True):
        if d is None:
            continue
        term = d * f_half
        total = term if total is None else total + term
    if total is None:
        return torch.zeros(points, dtype=dtype, device=device)
    return torch.fft.irfft2(total, s=points)


def _sweep_key(atom_set, req, partition, options, atom_scale, overrides, device):
    o = overrides or RowReplacements()
    return (
        content_digest(atom_set.sides),
        content_digest(atom_set.fractional),
        content_digest(atom_set.element_z),
        content_digest(atom_set.site_fraction),
        content_digest(atom_set.msd),
        req.points,
        req.replicas,
        partition.edges,
        options,
        content_digest(atom_scale),
        content_digest(o.beam_fraction),
        content_digest(o.plane_fraction),
        content_digest(o.msd),
        content_digest(o.site_fraction),
        str(req.real_dtype),
        str(device),
    )


def _wants_grad(atom_scale, overrides) -> bool:
    given = [] if overrides is None else list(overrides.tensors)
    if isinstance(atom_scale, torch.Tensor):
        given.append(atom_scale)
    return any(t.requires_grad for t in given)


def potential_stack(
    partition: BeamPartition,
    atom_set: AtomSet,
    points,
    *,
    replicas=None,
    beam_spread: str = "binned",
    edge_snap: bool = True,
    ignore_site_fraction: bool = False,
    atom_scale=None,
    scale_squared: bool = False,
    form_spectra=None,
    overrides: RowReplacements | None = None,
    jitter: ThermalJitter | None = None,
    rolloff: float | None = DEFAULT_ROLLOFF,
    dtype=None,
    device=None,
) -> torch.Tensor:
    """Return the projected potential of each slice of a periodic field of view.

    Physics: on the field of view of ``(c_y, c_x)`` cell copies, area
    ``A = L_y L_x``, the Fourier coefficient of slice ``s`` is::

        V_s(q) = (1 / A) sum_{(j, c)} p_{j,c,s} a_j F_{Z_j}(q) exp(-2 pi i q . r_{j,c})

    with ``r_{j,c}`` the position of atom ``j`` in copy ``c`` (after the
    thermal offset), ``p`` its share in the slice (``beam_spread``), ``a_j =
    o_j w_j`` (``o_j`` the site fraction when site fractions are in use, else
    1; ``w_j`` the coefficient ``c_j`` or ``c_j**2``, else 1) and ``F_Z`` the
    form spectrum [Volt Angstrom^3]. A caller spectrum enters through its
    even part ``(F(q) + F(-q)) / 2``.

    Meaning of a pixel: the output at grid point ``n`` is the band-limited
    point value of the series over the FFT-order frequencies ``J``::

        v_s[n] = Re sum_{q in J} T(q) V_s(q) exp(2 pi i q . r_n)
               = Re(ifft2(Ny Nx T V_s))[n]

    with the rolloff ``T`` (1 without a rolloff). The mean of slice ``s`` is
    ``V_s(0)``.

    Method: each atom copy goes on two interlaced grids (points ``n h`` and
    ``n h + h/2``) with the cubic B-spline; the mean of the phase-corrected
    rfft2 spectra divided by ``sinc^4`` gives the layout spectrum (Hockney
    and Eastwood 1988). Positions stay exact (float64). Relative L2 error
    inside the alias-free ellipse: about 1e-4 for one atom on a 0.1 Angstrom
    grid. One slice is in work at a time.

    Parameters
    ----------
    partition : BeamPartition
        The slicing along the beam.
    atom_set : AtomSet
        The structure.
    points : tuple of int
        Grid point counts ``(Ny, Nx)`` on the field of view.
    replicas : tuple of int, optional
        Copy counts ``(c_y, c_x)``. Default None: one copy per axis.
    beam_spread : str, optional
        ``"binned"`` (default), ``"tent"`` or ``"smeared"``; see
        :func:`~scatterem.physics.axial_shares.axial_shares`.
    edge_snap : bool, optional
        Edge snap of atomic layers (default True); see
        :meth:`~scatterem.physics.beam_partition.BeamPartition.bin_index`.
    ignore_site_fraction : bool, optional
        If False (default), the build uses site fractions when the set has a
        partial site. In use, every site fraction enters as given, also
        ``0.99995``. If True, every prefactor has the site factor 1.
    atom_scale : array_like or torch.Tensor, optional
        Real, finite coefficient ``c_j`` per atom ``(N,)``.
    scale_squared : bool, optional
        Use ``c_j**2`` (default False).
    form_spectra : array_like or torch.Tensor, optional
        Real, finite ``(E, Ny, Nx)`` in FFT order [Volt Angstrom^3], rows in
        :func:`~scatterem.physics.beam_partition.species_order`. Default ``potential_spectra`` (Z <= 103).
        With autograd through it, the graph keeps ``S E`` half-layout grids.
    overrides : RowReplacements, optional
        Per-site replacement tensors.
    jitter : ThermalJitter, optional
        Thermal draws. Default None: static potential.
    rolloff : float or None, optional
        Rolloff depth (see :func:`.potential_rolloff.rolloff_mask`). Default
        ``DEFAULT_ROLLOFF``; None for no rolloff.
    dtype : torch.dtype, optional
        The float32 or float64 dtype; complex64/complex128 give their real partner.
        Default the torch default dtype.
    device : torch.device or str, optional
        Default: the device of the tensor inputs, else the CPU.

    Returns
    -------
    torch.Tensor
        Real ``(S, Ny, Nx)`` [Volt Angstrom]. Zeros for an empty set.

    Raises
    ------
    TypeError
        For an input of a wrong type.
    ValueError
        For a bad value, or a combination without a correct gradient: a
        beam coordinate with grad and ``"binned"``; an ``msd`` with grad,
        ``"binned"`` and draws along the beam; ``"smeared"`` with draws along
        the beam; a group with ``<u^2> = 0`` where a derivative is needed; a
        site fraction replacement with ``ignore_site_fraction=True``, or with
        grad and no partial value.
    """
    req = check_stack_request(
        points,
        atom_set,
        replicas=replicas,
        atom_scale=atom_scale,
        scale_squared=scale_squared,
        dtype=dtype,
    )
    rows = atom_set.row_total
    _check_switches(partition, edge_snap, ignore_site_fraction, jitter, overrides, rows)
    weighting = not bool(ignore_site_fraction)
    spread = check_spread(beam_spread)
    w = check_rolloff(rolloff)
    species = species_order(atom_set)
    _check_form(form_spectra, len(species), req.points)
    given = () if overrides is None else overrides.tensors
    dev = resolve_stack_device(device, form_spectra, atom_scale, *given)
    _check_combinations(spread, partition, jitter, overrides, weighting)
    out = torch.zeros((partition.count, *req.points), dtype=req.real_dtype, device=dev)
    if rows == 0:
        return out
    ledger = active_ledger()
    key = None
    if ledger is not None:
        if jitter is not None or _wants_grad(atom_scale, overrides):
            ledger.note_bypass()
            ledger = None
        else:
            options = (
                spread,
                bool(edge_snap),
                weighting,
                bool(scale_squared),
            )
            key = _sweep_key(
                atom_set, req, partition, options, atom_scale, overrides, dev
            )
    stored = None if ledger is None else ledger.lookup(key)
    if stored is None:
        layouts = _layout_stream(
            atom_set,
            req,
            partition,
            spread,
            edge_snap,
            weighting,
            atom_scale,
            scale_squared,
            overrides,
            jitter,
            dev,
        )
        if ledger is not None:
            stored = list(layouts)
            ledger.keep(key, stored)
            layouts = iter(stored)
    else:
        layouts = iter(stored)
    spectra = _spectra_half(form_spectra, species, req, w, dev)
    for s, per_element in enumerate(layouts):
        out[s] = _synthesis(per_element, spectra, req.points, req.real_dtype, dev)
    return out


def _layout_stream(
    atom_set,
    req,
    partition,
    spread,
    edge_snap,
    weighting,
    atom_scale,
    squared,
    overrides,
    jitter,
    dev,
):
    """Return the iterator of layout spectra per slice and element."""
    data = atom_data(atom_set, dev, overrides)
    need_groups = spread == "smeared"
    copies = _copies(data, atom_set.sides, req.replicas, jitter, need_groups, dev)
    _, _, z, msd, _ = copies
    if msd is None:
        msd = torch.zeros_like(z.detach())
    beam = None if overrides is None else overrides.beam_fraction
    snap_dtype = torch.float64 if beam is None else beam.dtype
    if need_groups and beam is not None and beam.requires_grad:
        if bool((msd.detach() == 0).any()):
            raise ValueError(
                "overrides.beam_fraction: requires grad, but a row group has "
                "<u^2> = 0 and uses the 'binned' share there"
            )
    shares = axial_shares(
        partition,
        z,
        msd,
        float(atom_set.sides[2]),
        beam_spread=spread,
        edge_snap=bool(edge_snap),
        snap_dtype=snap_dtype,
    )
    prefactor = _prefactors(data, atom_scale, squared, weighting, dev)
    species = torch.tensor(species_order(atom_set), device=dev)
    z_rows = torch.from_numpy(np.array(atom_set.element_z, dtype=np.int64)).to(dev)
    elem = torch.searchsorted(species, z_rows)
    factors = (*layout_factors(req.points, req.real_dtype, dev), partition.count)
    return _slice_layouts(
        copies, shares, prefactor, elem, len(species_order(atom_set)), req, factors
    )
