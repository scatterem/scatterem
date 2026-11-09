# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.4.0",
#     "marimo==0.25.2",
#     "matplotlib>=3.10",
#     "numpy>=2.1.0",
#     "torch>=2.5.0",
# ]
# ///
"""BiP-PRISM core-loss Ti L map of SrTiO3, as a marimo notebook."""

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
    # BiP-PRISM: a core-loss Ti L map of SrTiO3

    This notebook simulates a STEM-EELS elemental map of the Ti L edge in
    SrTiO3 with BiP-PRISM. It follows the page "Simulate STEM-EELS"
    (`docs/eels.md`) of the `scatterem` documentation. The simulation takes a
    few seconds on a GPU.

    **Paper:** P. Pelz, *J. Phys. Mater.* (2026),
    doi:[10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0).

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a GPU
       (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells.

    ## The numbers are not quantitative without GPAW

    The transition potentials need atomic radial wavefunctions. Without GPAW,
    `scatterem` uses analytic hydrogenic bound states and a free continuum,
    and it emits a `HydrogenicFallbackWarning`. The selection rules and the
    shape of the pipeline are correct. The cross-sections are not. Thus this
    map is qualitative.

    For quantitative numbers, install the optional `eels` extra, which builds
    GPAW from source and needs a C++ compiler:

    ```bash
    CC=g++ pip install "scatterem[eels]==0.4.0" gpaw-data
    ```

    This notebook does not install GPAW.
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
    ## 1. The structure

    `AtomSet` holds the side lengths of a rectangular box in Ångström, the
    fractional coordinates of the atoms and their atomic numbers. The
    coordinates are (y, x, depth). The beam goes along the depth axis. The
    cell below replicates the SrTiO3 unit cell 4 x 4 x 8 times
    (15.6 x 15.6 x 31.2 Å, 640 atoms).
    """)
    return


@app.cell
def _(np):
    from scatterem.physics.atom_set import AtomSet

    a = 3.905
    sto = AtomSet(
        np.array([a, a, a]),  # box sides (y, x, depth), Angstrom
        np.array(
            [  # fractional coordinates (y, x, depth)
                [0.0, 0.0, 0.0],  # Sr
                [0.5, 0.5, 0.5],  # Ti
                [0.5, 0.5, 0.0],  # O
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        ),
        np.array([38, 22, 8, 8, 8]),  # atomic numbers
        msd=None,  # <u^2> per atom; None gives every atom 0.01 A^2
    ).replicated((4, 4, 8))
    print(f"box sides: {sto.sides} Angstrom")
    return (sto,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The simulator

    The grid covers the whole cell, and the scan is a regular raster over the
    cell. Thus the scan step is the cell side divided by the number of scan
    points. `semiconvergence_angle` and `detector_mrad` are in milliradians.
    """)
    return


@app.cell
def _(sto):
    from scatterem.simulation.eels import PrismEelsSimulator

    SCAN = (32, 32)
    sim = PrismEelsSimulator(
        sto,
        eV=100e3,  # probe energy
        semiconvergence_angle=25.0,  # mrad
        pixels=(256, 256),  # real-space grid across the cell
        scan=SCAN,  # scan positions, a raster over the cell
        num_slices=16,
        interpolation_factor=2,  # PRISM factor f
        device="cuda",
    )
    # For quantitative numbers with GPAW installed, pass
    # transitions=dict(wavefunctions="gpaw") to eels_map below.
    return SCAN, sim


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The Ti L map

    `edge=(22, 2, 1)` is the ionized Ti 2p subshell (Z, n, l). `epsilon` is the
    energy of the ejected electron above the threshold. `partition` and
    `partition_s2` partition both PRISM matrices; this is the "Bi" in BiP.
    """)
    return


@app.cell
def _(sim):
    ti_map = sim.eels_map(
        (22, 2, 1),  # Ti L: (Z, n, l) of the ionized subshell
        epsilon=10.0,  # eV above the ionization threshold
        detector_mrad=25.0,  # EELS collection semi-angle
        inelastic_crop=4,
        partition=dict(n_radial=4),  # partition S1
        partition_s2=dict(n_radial=4),  # ... and S2
    )
    print(f"map shape: {tuple(ti_map.data.shape)}")
    print(f"energy losses: {ti_map.energy_losses} eV")
    return (ti_map,)


@app.cell
def _(SCAN, plt, sto, ti_map):
    import scatterem.vis as vis
    from scatterem.data import Sampling

    # The scan is a raster over the whole cell: step = cell side / scan points.
    scan_sampling = Sampling(
        pixel_size=(
            float(sto.sides[0]) / SCAN[0],
            float(sto.sides[1]) / SCAN[1],
        ),
        units=("Å", "Å"),
    )
    _fig, _ax = plt.subplots(figsize=(6, 6))
    vis.show_2d(
        [ti_map.data],
        cbar=True,
        title=["Ti L map (hydrogenic states, not quantitative)"],
        figax=(_fig, _ax),
        sampling=scan_sampling,
    )
    _fig
    return


@app.cell
def _(ti_map, torch):
    assert torch.isfinite(ti_map.data).all()
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
