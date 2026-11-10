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
"""BiP-PRISM five-edge maps of a stepped, intermixed LaAlO3/SrTiO3 interface, as a marimo notebook."""

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
    # BiP-PRISM: five core-loss maps of a LaAlO₃/SrTiO₃ interface

    This notebook simulates elemental maps of five edges (Sr L, La M, Ti L,
    Al K and O K) across a LaAlO₃/SrTiO₃ interface with a one-cell step and
    cation intermixing. For each edge, it computes the map twice: with
    BiP-PRISM (both scattering matrices partitioned into 4 rings) and with
    unpartitioned PRISM as the reference. Both use the same grid, PRISM
    factor, crop window and detector.

    **What to look at.** The maps show the step and the intermixed layers:
    the A site (Sr/La) and the B site (Ti/Al) change across the interface,
    and the step moves the interface by one cell between the left and the
    right half. The difference row shows the partitioning error. The table
    gives the relative L2 error and the ratio of the integrated signals, as
    the paper does.

    **Paper:** P. Pelz, *The bi-partitioned PRISM (BiP-PRISM) algorithm for
    fast and scalable core-loss STEM-EELS simulations*, *J. Phys. Mater.*
    (2026), doi:[10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0),
    section "Multimodal showcase: LaAlO₃/SrTiO₃ interface", and the
    specimen construction in supplementary section S1.

    **Data:** none. The notebook builds the specimen from the published
    recipe and its random seed.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a GPU
       (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells. The default settings take about 1.5 minutes on an
       RTX PRO 6000 (4 minutes on an RTX A6000) and about 14 GB of GPU
       memory.

    ## The numbers are not quantitative without GPAW

    The paper uses GPAW transition potentials. Without GPAW, `scatterem`
    uses analytic hydrogenic bound states and a free continuum. This
    notebook selects them explicitly, so you see no
    `HydrogenicFallbackWarning`. The maps show the right sites, but their
    relative strengths are not quantitative. For GPAW, install
    `scatterem[eels]` (see the page "Simulate STEM-EELS" of the
    documentation) and select `gpaw` below.
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
    ## 1. The specimen

    The recipe follows supplementary section S1 of the paper. The
    pseudocubic cell has 16 × 16 × 6 unit cells with $a = 3.905$ Å. The axes
    are growth [001], lateral [010] and beam [100]. SrTiO₃ fills the first 8
    growth cells on the left terrace and 9 on the right terrace, so the step
    is at lateral cell 8. LaAlO₃ fills the rest.

    Intermixing exchanges paired A/B cations (La/Al with Sr/Ti) in the two
    layers next to the interface on each side: 24 of the 96 formula units in
    the adjacent layer and 10 of 96 in the next layer. The two sides use
    independent random selections with equal counts, from a NumPy PCG64
    generator with seed 20260905. Oxygen does not move. The model is
    unrelaxed. The FFT cell is periodic, so a second step and a second
    interface appear at the cell boundaries.

    Clear the check box to get the flat, sharp control specimen of the
    paper.
    """)
    return


@app.cell
def _(mo):
    intermixed = mo.ui.checkbox(value=True, label="one-cell step and intermixing")
    intermixed
    return (intermixed,)


@app.cell
def _(intermixed, np):
    from scatterem.physics.atom_set import AtomSet

    A_STO = 3.905  # Angstrom
    N_CELLS = (16, 16, 6)  # growth, lateral, beam

    def build_interface(step, fractions, seed=20260905):
        """Stepped, intermixed LaAlO3/SrTiO3 as in supplementary section S1."""
        n_growth, n_lat, n_beam = N_CELLS
        # SrTiO3 growth cells in each lateral column: 8, then 9 after the step.
        boundary = n_growth // 2 + int(step) * (np.arange(n_lat) >= n_lat // 2)
        basis = np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        )  # A, B, O, O, O (fractional within one cell)
        i, j, k = np.meshgrid(
            np.arange(n_growth), np.arange(n_lat), np.arange(n_beam), indexing="ij"
        )
        position = (np.stack([i, j, k], -1)[..., None, :] + basis) * A_STO
        sto = (i < boundary[j])[..., None]
        Z = np.where(sto, [38, 22, 8, 8, 8], [57, 13, 8, 8, 8]).astype(int)
        rng = np.random.Generator(np.random.PCG64(seed))
        population = n_lat * n_beam  # formula units in one growth layer
        for depth, fraction in enumerate(fractions):
            count = int(np.floor(fraction * population + 0.5))
            for side in (-1, 1):  # SrTiO3 side, then LaAlO3 side
                for index in rng.choice(population, count, replace=False):
                    jj, kk = divmod(int(index), n_beam)
                    if side == -1:
                        Z[boundary[jj] - 1 - depth, jj, kk, :2] = [57, 13]
                    else:
                        Z[boundary[jj] + depth, jj, kk, :2] = [38, 22]
        cell = np.array(N_CELLS) * A_STO
        return AtomSet(cell, position.reshape(-1, 3) / cell, Z.reshape(-1), msd=None), Z

    if intermixed.value:
        specimen, _Z = build_interface(step=True, fractions=(0.25, 0.10))
    else:
        specimen, _Z = build_interface(step=False, fractions=())
    # Fraction of La on the A sites of each (growth, lateral) column.
    la_fraction = (_Z[..., 0] == 57).mean(axis=2)
    print(f"box sides: {specimen.sides} Angstrom, {len(specimen.element_z)} atoms")
    return A_STO, la_fraction, specimen


@app.cell
def _(A_STO, la_fraction, plt):
    import scatterem.vis as vis
    from scatterem.data import Sampling

    fig_specimen, _ax = plt.subplots(figsize=(5, 4.5))
    _n_axes = len(fig_specimen.axes)
    vis.show_2d(
        [la_fraction],
        figax=(fig_specimen, _ax),
        sampling=Sampling(pixel_size=(A_STO, A_STO), units=("Å", "Å")),
        role="intensity",
        title=["La fraction on the A sites (growth vertical)"],
    )
    fig_specimen.axes[_n_axes].set_ylabel("La fraction in the projected column")
    fig_specimen
    return Sampling, vis


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Settings

    The paper setup is 300 kV, a 20 mrad probe and a 20 mrad EELS detector,
    a 640 × 640 grid, 12 slices (2 per cell), PRISM factor $f = 4$, crop
    divisor 22 and `epsilon` 5 eV above each threshold. The paper scan has
    179 × 179 positions over the cell. A smaller scan is faster; it does not
    change the scattering matrices.
    """)
    return


@app.cell
def _(mo):
    EDGE_NAMES = ["Sr L", "La M", "Ti L", "Al K", "O K"]
    edges = mo.ui.multiselect(options=EDGE_NAMES, value=EDGE_NAMES, label="edges")
    scan_side = mo.ui.dropdown(
        options=["60", "90", "179"], value="179", label="scan positions per side"
    )
    n_radial = mo.ui.slider(
        2, 8, value=4, step=1, label="rings of S1 and S2 (n_radial)"
    )
    wavefunctions = mo.ui.dropdown(
        options=["hydrogenic", "gpaw"], value="hydrogenic", label="bound states"
    )
    mo.vstack([edges, mo.hstack([scan_side, n_radial]), wavefunctions])
    return EDGE_NAMES, edges, n_radial, scan_side, wavefunctions


@app.cell
def _(scan_side, specimen):
    from scatterem.simulation.eels import PrismEelsSimulator

    SCAN = (int(scan_side.value), int(scan_side.value))
    sim = PrismEelsSimulator(
        specimen,
        eV=300e3,
        semiconvergence_angle=20.0,  # mrad
        pixels=(640, 640),
        scan=SCAN,
        num_slices=12,
        interpolation_factor=4,  # PRISM factor f
        device="cuda",
    )
    return SCAN, sim


@app.cell
def _(EDGE_NAMES, SCAN, edges, n_radial, sim, torch, wavefunctions):
    import time

    from scatterem.simulation.eels import prism_eels_image

    EDGES = {
        "Sr L": (38, 2, 1),
        "La M": (57, 3, 2),
        "Ti L": (22, 2, 1),
        "Al K": (13, 1, 0),
        "O K": (8, 1, 0),
    }
    # The transmissions, the probe and the scan are the same for every edge.
    _T, _probe, _scan = sim.transmissions(), sim.probe_q(), sim.scan_pixels()
    _common = dict(
        wavelength=sim.wavelength,
        gridsize=sim.gridsize,
        slice_distance=sim.slice_distance,
        detector_mrad=20.0,
        interpolation_factor=4,
        inelastic_crop=22,
        prism_mask=True,
        scan_shape=SCAN,
    )
    _part = dict(n_radial=n_radial.value)
    results = {}
    for _name in [_n for _n in EDGE_NAMES if _n in edges.value]:
        _edge = EDGES[_name]
        _tp = sim.transition_potentials(_edge, 5.0, wavefunctions=wavefunctions.value)
        _args = (_probe, _T, _tp, sim.sites(_edge[0]), _scan)
        _t0 = time.perf_counter()
        _reference = prism_eels_image(*_args, **_common)
        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        _partitioned = prism_eels_image(
            *_args, partition=_part, partition_s2=_part, **_common
        )
        torch.cuda.synchronize()
        _t2 = time.perf_counter()
        results[_name] = dict(
            reference=_reference.double().cpu().numpy(),
            partitioned=_partitioned.double().cpu().numpy(),
            error=float((_partitioned - _reference).norm() / _reference.norm()),
            ratio=float(_partitioned.sum() / _reference.sum()),
            seconds=(_t1 - _t0, _t2 - _t1),
        )
        print(
            f"{_name}: error {100 * results[_name]['error']:.3f} %, "
            f"ratio {results[_name]['ratio']:.5f}, "
            f"reference {_t1 - _t0:.1f} s, partitioned {_t2 - _t1:.1f} s"
        )
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The maps

    Top row: the BiP-PRISM maps. Bottom row: the difference to the
    unpartitioned reference, in percent of the largest reference value.
    Growth runs from top to bottom, and SrTiO₃ is at the top.
    """)
    return


@app.cell
def _(SCAN, Sampling, np, plt, results, sim, vis):
    UNIT = "probability per electron per eV"
    _names = list(results)
    _sampling = Sampling(
        pixel_size=(sim.gridsize[0] / SCAN[0], sim.gridsize[1] / SCAN[1]),
        units=("Å", "Å"),
    )
    fig_maps, _axs = plt.subplots(
        2, len(_names), figsize=(3.4 * len(_names), 6.6), squeeze=False
    )
    for _k, _name in enumerate(_names):
        _r = results[_name]
        _n_axes = len(fig_maps.axes)
        vis.show_2d(
            [_r["partitioned"]],
            figax=(fig_maps, _axs[0, _k]),
            sampling=_sampling,
            role="intensity",
            title=[f"{_name}, BiP-PRISM"],
            tight_layout=False,
        )
        fig_maps.axes[_n_axes].set_ylabel(UNIT, fontsize=7)
        _diff = (
            100 * (_r["partitioned"] - _r["reference"]) / np.abs(_r["reference"]).max()
        )
        _n_axes = len(fig_maps.axes)
        vis.show_2d(
            [_diff],
            figax=(fig_maps, _axs[1, _k]),
            sampling=_sampling,
            role="signed",
            title=[f"{_name}, minus reference"],
            tight_layout=False,
        )
        fig_maps.axes[_n_axes].set_ylabel("% of the reference maximum", fontsize=7)
    fig_maps.tight_layout()
    fig_maps
    return (UNIT,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Growth profiles, left and right of the step

    Each curve averages a map over one terrace (lateral cells 0-7 or 8-15),
    removes the atomic modulation along the growth direction with a low-pass
    filter at the lattice period, and scales the result to its maximum. Solid lines are the left terrace, dashed lines the right
    terrace. The A-site (Sr, La) and B-site (Ti, Al) profiles cross at the
    interface. On the right terrace, the crossing is one cell lower,
    because of the step. The intermixed layers make the crossing gradual.
    A coarse scan (60 or 90 positions) undersamples the atomic peaks, and
    the profiles then show ripples.
    """)
    return


@app.cell
def _(A_STO, SCAN, np, plt, results, sim):
    fig_profiles, _ax = plt.subplots(figsize=(7, 4))
    _growth = (np.arange(SCAN[0]) + 0.5) * sim.gridsize[0] / SCAN[0] / 10  # nm
    _cols = (np.arange(SCAN[1]) + 0.5) * sim.gridsize[1] / SCAN[1]
    _left = _cols < sim.gridsize[1] / 2  # lateral cells 0-7
    # Low-pass along growth: keep the Fourier harmonics below the lattice
    # period (16 cells per box), with a cosine taper. This removes the
    # atomic modulation and keeps the variation from cell to cell.
    _n_cells = int(round(sim.gridsize[0] / A_STO))
    _harmonic = np.arange(SCAN[0] // 2 + 1)
    _taper = np.where(
        _harmonic < _n_cells, np.cos(np.pi * _harmonic / (2 * _n_cells)) ** 2, 0.0
    )
    for _k, (_name, _r) in enumerate(results.items()):
        for _half, _style in ((_left, "-"), (~_left, "--")):
            _row = _r["partitioned"][:, _half].mean(axis=1)
            _smooth = np.fft.irfft(np.fft.rfft(_row) * _taper, n=SCAN[0])
            _ax.plot(
                _growth,
                _smooth / _smooth.max(),
                _style,
                color=f"C{_k}",
                label=_name if _style == "-" else None,
            )
    _ax.set_xlabel("growth position (nm), SrTiO₃ at 0")
    _ax.set_ylabel("low-pass profile / maximum")
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=8, title="solid: left, dashed: right", title_fontsize=8)
    fig_profiles.tight_layout()
    fig_profiles
    return


@app.cell
def _(intermixed, mo, results):
    # Values of the paper (GPAW states, older code, full 179 x 179 scan).
    _paper = {
        True: {
            "Sr L": (0.461, 0.99755),
            "La M": (0.777, 0.99742),
            "Ti L": (0.162, 0.99946),
            "Al K": (0.151, 0.99886),
            "O K": (0.099, 0.99971),
        },
        False: {
            "Sr L": (0.434, 0.99770),
            "La M": (0.795, 0.99730),
            "Ti L": (0.165, 0.99944),
            "Al K": (0.145, 0.99891),
            "O K": (0.099, 0.99971),
        },
    }[intermixed.value]
    _rows = [
        "| edge | error (%) | paper error (%) | integrated ratio | paper ratio |",
        "|---|---|---|---|---|",
    ]
    for _name, _r in results.items():
        _pe, _pr = _paper[_name]
        _rows.append(
            f"| {_name} | {100 * _r['error']:.3f} | {_pe:.3f} | "
            f"{_r['ratio']:.5f} | {_pr:.5f} |"
        )
    mo.md(
        "## 5. Partitioning error per edge\n\n"
        "The error is the relative L2 difference of the BiP-PRISM map and the "
        "unpartitioned map. The ratio is the integrated BiP-PRISM signal over "
        "the integrated reference signal.\n\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Differences from the paper

    - **Software version.** The paper used older code. Since v0.4.0, a map
      value is a probability per incident electron and per eV. v0.4 builds
      the transition potentials with a new first-Born model, and it builds
      the specimen with a new physics layer (projected potentials,
      transmissions and a soft-edged probe aperture). So the absolute map
      values differ from the paper. Compare the relative quantities: the
      error and the integrated ratio.
    - **Hydrogenic instead of GPAW states.** With the hydrogenic states and
      the v0.4 setup, the errors are about 1-2 % for Sr L, La M, Al K and
      O K, and about 0.24 % for Ti L. The paper gives 0.10-0.78 % with GPAW.
      The integrated ratios stay within 2 % of 1 (paper: within 0.3 %).
      Install GPAW to compare at the paper's physics.
    - **Same specimen.** The notebook builds the specimen from the recipe of
      supplementary section S1 with the same seed. The paper archives the
      same atom arrays with a checksum; this construction reproduces them
      exactly.
    - **Same size.** The default settings use the paper's grid, slices,
      PRISM factor, crop and 179 × 179 scan. The paper reports the timings
      of this run on an A100; this notebook prints its own times. The
      partitioned call uses about 2-3 GB of GPU memory and the reference
      about 13 GB.
    """)
    return


@app.cell
def _(np, results, torch):
    assert all(np.isfinite(_r["partitioned"]).all() for _r in results.values())
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
