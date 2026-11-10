# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.4.2",
#     "marimo==0.25.2",
#     "matplotlib>=3.10",
#     "numpy>=2.1.0",
#     "torch>=2.5.0",
# ]
# ///
"""BiP-PRISM validity regime: partitioning error of single atomic columns, as a marimo notebook."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # BiP-PRISM validity regime: where does partitioning start to fail?

    Partitioning builds the probe scattering matrix $\mathcal{S}_1$ on a few
    *parent* beams and interpolates the other beams from them. The
    interpolation is accurate where the probe stays simple. Channelling along
    an atomic column makes the probe complex, so the error should grow with
    thickness. This notebook measures that error for single atomic columns.

    For each column, the notebook computes the energy-filtered 4D-STEM cube
    twice with PRISM: once with every aperture beam (the reference), and once
    with $\mathcal{S}_1$ partitioned into 4 rings (`n_radial=4`). Only the
    probe leg is partitioned. The exit leg keeps the full diffraction
    pattern. Two numbers describe the difference:

    - the **cube error**, the relative L2 difference of the 4D cubes;
    - the **map error**, the relative L2 difference of the maps that you get
      when you sum each diffraction pattern over the full detector grid.

    **What to look at.** (a) Both errors grow with thickness inside each
    lateral cell. At 2 nm the cell changes from 10 Å to 20 Å, and the error
    drops (see section 1). (b) At a fixed thickness of 1 nm, the edge (O K
    to Cu L) changes the error by less than a factor of 3.

    **Paper:** P. Pelz, *The bi-partitioned PRISM (BiP-PRISM) algorithm for
    fast and scalable core-loss STEM-EELS simulations*, *J. Phys. Mater.*
    (2026), doi:[10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0),
    section "Validity regime and the locality diagnostic" and supplementary
    tables S1 and section S7.

    **Data:** none. The notebook builds the columns.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a GPU
       (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells. The default settings take less than a minute on an
       RTX PRO 6000 (about 2 minutes on an RTX A6000).

    ## The numbers are not quantitative without GPAW

    The paper uses GPAW transition potentials for this study. Without GPAW,
    `scatterem` uses analytic hydrogenic bound states and a free continuum.
    This notebook selects them explicitly, so you see no
    `HydrogenicFallbackWarning`. With these states, the partitioning errors
    are larger than in the paper (see "Differences from the paper" at the
    end). Install `scatterem[eels]` for GPAW and select `gpaw` below to
    compare.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On molab, select a GPU in the notebook specs "
            "and restart the notebook."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return np, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Settings

    A column has atoms 2.5 Å apart along the beam, so 4 atoms are 1 nm thick.
    The probe has 100 kV and a 40 mrad semi-angle, and the scan has 6 × 6
    positions over the cell. As in the paper, the lateral cell grows with
    thickness at a constant pixel size of 0.078 Å: a 10 Å cell on 128 × 128
    pixels up to 2 nm, and a 20 Å cell on 256 × 256 pixels above 2 nm. The
    larger cell also has more aperture beams (1473 instead of 365) for the
    same 61 parents. So the error step at 2 nm is not an effect of thickness
    alone.

    **Aperture edge.** The default is the soft-edged probe of
    `PrismEelsSimulator` in v0.4. The paper uses a hard aperture: a beam is
    in or out. With the soft edge, about twice as many beams have a nonzero
    weight (749 instead of 365 in the 10 Å cell, 2145 instead of 1473 in the
    20 Å cell). Select "hard (as in the paper)" to compare with the paper's
    aperture.
    """)
    return


@app.cell
def _(mo):
    THICKNESS_OPTIONS = ["1", "2", "4", "8", "12", "16", "20", "24"]
    thickness_atoms = mo.ui.multiselect(
        options=THICKNESS_OPTIONS,
        value=["1", "2", "4", "8", "12", "16"],
        label="atoms in the column (2.5 Å each)",
    )
    thickness_edge = mo.ui.dropdown(
        options=["O K", "Ti L", "Fe L", "Cu L"],
        value="O K",
        label="edge of the thickness sweep",
    )
    n_radial = mo.ui.slider(2, 8, value=4, step=1, label="S1 rings (n_radial)")
    aperture = mo.ui.dropdown(
        options=["hard (as in the paper)", "soft (v0.4 default)"],
        value="soft (v0.4 default)",
        label="aperture edge",
    )
    defocus = mo.ui.slider(0.0, 200.0, value=0.0, step=25.0, label="defocus (Å)")
    wavefunctions = mo.ui.dropdown(
        options=["hydrogenic", "gpaw"], value="hydrogenic", label="bound states"
    )
    mo.vstack(
        [
            thickness_atoms,
            thickness_edge,
            mo.hstack([n_radial, aperture]),
            mo.hstack([defocus, wavefunctions]),
        ]
    )
    return aperture, defocus, n_radial, thickness_atoms, thickness_edge, wavefunctions


@app.cell
def _(np, torch):
    from scatterem.physics.atom_set import AtomSet
    from scatterem.simulation.eels import (
        PrismEelsSimulator,
        prism_transition_potential,
    )

    EDGES = {
        "O K": (8, 1, 0),
        "Ti L": (22, 2, 1),
        "Fe L": (26, 2, 1),
        "Cu L": (29, 2, 1),
    }
    SPACING = 2.5  # Angstrom between the atoms of a column

    def column_simulator(Z, n_atoms, defocus):
        """One column of n_atoms atoms of element Z on the paper's adaptive grid."""
        side, pixels = (10.0, 128) if n_atoms <= 8 else (20.0, 256)
        depth = n_atoms * SPACING
        column = AtomSet(
            np.array([side, side, depth]),
            np.array([[0.5, 0.5, (i + 0.5) / n_atoms] for i in range(n_atoms)]),
            np.full(n_atoms, Z),
            msd=None,
        )
        return PrismEelsSimulator(
            column,
            eV=100e3,
            semiconvergence_angle=40.0,  # mrad
            pixels=(pixels, pixels),
            scan=(6, 6),
            num_slices=max(4, round(depth / 0.5)),  # about 0.5 A slices
            defocus=defocus,
            device="cuda",
        )

    def hard_aperture(sim):
        """A hard-edged probe aperture with defocus and unit total intensity."""
        qy = torch.fft.fftfreq(sim.pixels[0], d=sim.sampling[0], dtype=torch.float64)
        qx = torch.fft.fftfreq(sim.pixels[1], d=sim.sampling[1], dtype=torch.float64)
        q2 = qy[:, None] ** 2 + qx[None, :] ** 2
        inside = q2 <= (sim.semiconvergence_angle * 1e-3 / sim.wavelength) ** 2
        chi = torch.pi * sim.wavelength * sim.defocus * q2
        probe = inside * torch.exp(-1j * chi)
        probe = probe / (probe.abs().pow(2).sum() / probe.numel()).sqrt()
        return probe.to(sim.dtype).to(sim.device)

    def errors(edge, n_atoms, n_rings, hard, defocus, wavefunctions):
        """Cube and map errors of the partitioned against the exact PRISM cube."""
        sim = column_simulator(edge[0], n_atoms, defocus)
        probe = hard_aperture(sim) if hard else sim.probe_q()
        args = (
            probe,
            sim.transmissions(),
            sim.transition_potentials(edge, 1.0, wavefunctions=wavefunctions),
            sim.sites(edge[0]),
            sim.scan_pixels(),
        )
        common = dict(
            wavelength=sim.wavelength,
            gridsize=sim.gridsize,
            slice_distance=sim.slice_distance,
        )
        exact = prism_transition_potential(*args, **common)
        part = prism_transition_potential(
            *args,
            **common,
            partition=dict(n_radial=n_rings, focal_backprop="centroid"),
        )
        cube = float((part - exact).norm() / exact.norm())
        map_exact, map_part = exact.sum((-2, -1)), part.sum((-2, -1))
        emap = float((map_part - map_exact).norm() / map_exact.norm())
        beams = int((probe.abs() > 0).sum())
        del exact, part
        torch.cuda.empty_cache()
        return cube, emap, beams

    return EDGES, SPACING, errors


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The two sweeps

    The thickness sweep runs one column per thickness. The edge sweep runs
    the four edges at 4 atoms (1 nm). Each point computes the exact and the
    partitioned cube. The 256 × 256 points above 2 nm take most of the time.
    """)
    return


@app.cell
def _(
    EDGES,
    SPACING,
    aperture,
    defocus,
    errors,
    n_radial,
    np,
    thickness_atoms,
    thickness_edge,
    wavefunctions,
):
    _hard = aperture.value.startswith("hard")
    _atoms = sorted(int(_v) for _v in thickness_atoms.value)
    _rows = []
    for _n in _atoms:
        _c, _m, _b = errors(
            EDGES[thickness_edge.value],
            _n,
            n_radial.value,
            _hard,
            defocus.value,
            wavefunctions.value,
        )
        _rows.append((_n * SPACING / 10.0, _c, _m, _b))
        print(
            f"{_n:3d} atoms ({_n * SPACING / 10:.2f} nm, {_b} beams): "
            f"cube {_c * 100:.3f} %, map {_m * 100:.3f} %"
        )
    thickness_sweep = np.array(_rows).reshape(-1, 4)

    _rows = []
    for _name, _edge in EDGES.items():
        _c, _m, _b = errors(
            _edge, 4, n_radial.value, _hard, defocus.value, wavefunctions.value
        )
        _rows.append((_c, _m))
        print(f"{_name:5s} at 1 nm: cube {_c * 100:.3f} %, map {_m * 100:.3f} %")
    edge_sweep = np.array(_rows)
    return edge_sweep, thickness_sweep


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The validity regime

    (a) Cube and map error against thickness. The grey symbols are the
    values of the paper (O K, GPAW states, older code, hard aperture). The
    dotted line marks the change from the 10 Å to the 20 Å cell. With the
    default soft aperture, the 20 Å cell (3 and 4 nm) agrees with the paper,
    and the 10 Å cell (up to 2 nm) has larger errors than the paper.
    (b) Errors of the four edges at 1 nm.
    """)
    return


@app.cell
def _(EDGES, edge_sweep, np, plt, thickness_edge, thickness_sweep):
    # Values of the paper (figure "Validity regime", O K, GPAW states).
    PAPER_NM = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    PAPER_CUBE = np.array(
        [4.86e-4, 1.063e-3, 3.504e-3, 1.1268e-2, 4.171e-2, 7.446e-2, 0.11335, 0.15453]
    )
    PAPER_MAP = np.array(
        [2.35e-5, 5.12e-4, 2.753e-3, 1.0789e-2, 3.378e-2, 5.846e-2, 8.577e-2, 0.11316]
    )
    PAPER_EDGE_CUBE = np.array([3.504e-3, 3.295e-3, 3.593e-3, 3.870e-3])
    PAPER_EDGE_MAP = np.array([2.753e-3, 2.666e-3, 3.150e-3, 3.558e-3])

    fig_regime, _axs = plt.subplots(1, 2, figsize=(12, 4.5))
    _ax = _axs[0]
    _nm = thickness_sweep[:, 0]
    _ax.semilogy(
        _nm, 100 * thickness_sweep[:, 1], "o-", color="C0", label="cube, this run"
    )
    _ax.semilogy(
        _nm, 100 * thickness_sweep[:, 2], "s-", color="C1", label="map, this run"
    )
    if thickness_edge.value == "O K":
        _ax.semilogy(
            PAPER_NM,
            100 * PAPER_CUBE,
            "o--",
            color="0.6",
            mfc="none",
            label="cube, paper",
        )
        _ax.semilogy(
            PAPER_NM,
            100 * PAPER_MAP,
            "s--",
            color="0.6",
            mfc="none",
            label="map, paper",
        )
    _ax.axvline(2.0, color="0.3", lw=0.8, ls=":")
    _ax.set_xlabel("column thickness (nm)")
    _ax.set_ylabel("relative L2 error (%)")
    _ax.set_title(f"(a) thickness sweep, {thickness_edge.value}")
    _ax.grid(alpha=0.3, which="both")
    _ax.legend(fontsize=8)

    _ax = _axs[1]
    _x = np.arange(len(EDGES))
    _w = 0.2
    _ax.bar(
        _x - 1.5 * _w, 100 * edge_sweep[:, 0], _w, color="C0", label="cube, this run"
    )
    _ax.bar(
        _x - 0.5 * _w, 100 * edge_sweep[:, 1], _w, color="C1", label="map, this run"
    )
    _ax.bar(_x + 0.5 * _w, 100 * PAPER_EDGE_CUBE, _w, color="0.55", label="cube, paper")
    _ax.bar(_x + 1.5 * _w, 100 * PAPER_EDGE_MAP, _w, color="0.8", label="map, paper")
    _ax.set_xticks(_x, list(EDGES))
    _ax.set_ylabel("relative L2 error (%)")
    _ax.set_title("(b) edge sweep at 1 nm")
    _ax.grid(alpha=0.3, axis="y")
    _ax.legend(fontsize=8)
    fig_regime.tight_layout()
    fig_regime
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Differences from the paper

    - **Software version.** The paper used older code. Since v0.4.0, cube
      values are probabilities per incident electron and per eV, and v0.4
      builds the transition potentials with a new first-Born model and the
      specimen with a new physics layer (projected potentials and
      transmissions). The relative errors do not depend on the absolute
      scale, but they depend on the shape of the transition potentials and
      of the probe.
    - **Soft aperture (the default) against the paper.** The O K cube and
      map errors, here and in the paper (GPAW states, hard aperture):

      | thickness | cell | cube here | cube paper | map here | map paper |
      |---|---|---|---|---|---|
      | 0.25 nm | 10 Å | 0.11 % | 0.049 % | 0.062 % | 0.0024 % |
      | 0.5 nm | 10 Å | 0.39 % | 0.11 % | 0.36 % | 0.051 % |
      | 1 nm | 10 Å | 2.1 % | 0.35 % | 2.2 % | 0.28 % |
      | 2 nm | 10 Å | 8.4 % | 1.1 % | 8.8 % | 1.1 % |
      | 3 nm | 20 Å | 4.3 % | 4.2 % | 1.9 % | 3.4 % |
      | 4 nm | 20 Å | 7.0 % | 7.4 % | 4.0 % | 5.8 % |

      In the 20 Å cell (3 and 4 nm), the cube error agrees with the paper
      to 7 %, and the map error is lower than in the paper. In the 10 Å
      cell (up to 2 nm), the errors are 2 to 8 times larger than in the
      paper (cube) and larger still for the map at the smallest thickness.
      The edge sweep at 1 nm gives 0.8-2.1 % (cube) here and 0.33-0.39 % in
      the paper.
    - **Hard aperture (option).** With the hard aperture of the paper, the
      errors are about 3 to 4 times larger than in the paper at every
      thickness, with the same trend: for O K, 1.17 % at 1 nm (paper
      0.35 %) and 24 % at 4 nm (paper 7.4 %). The edge sweep at 1 nm gives
      0.5-1.2 % here.
    - **Why the aperture matters.** The soft edge gives more beams a
      nonzero weight. In the 10 Å cell, the same 61 parents must then
      interpolate 749 beams instead of 365, and the error grows. In the
      20 Å cell, the soft edge gives the outer beams small weights, and the
      error falls below that of the hard edge. So the partitioning error
      depends on the beam weights at the aperture edge, not only on the
      parent count. The hydrogenic states (instead of GPAW) and the v0.4
      transition potentials change the errors too; this notebook cannot
      separate the two effects without GPAW.
    - **Reduced sweep.** The paper sweeps up to 24 atoms (6 nm). The default
      here stops at 16 atoms (4 nm), to keep the run short. Select 20 and 24
      to extend it (about four more minutes).
    - **No on-atom diagnostic.** The paper also plots the on-atom probe
      error. This notebook shows only the cube and map errors.
    - **Single run.** The paper reran this study on an H100. Your GPU gives
      the same errors to rounding.
    """)
    return


@app.cell
def _(thickness_sweep, torch):
    import math

    assert all(math.isfinite(float(_v)) for _v in thickness_sweep.ravel())
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
