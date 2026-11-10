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
"""BiP-PRISM output modes: map, qEELS, PACBED and energy-momentum, as a marimo notebook."""

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
    # BiP-PRISM output modes: one calculation, four views

    The two-matrix BiP-PRISM path computes one complex amplitude for each
    detector beam, scan position and ionized atom. The output mode only
    decides how the code adds these intensities:

    1. **Elemental map.** Sum over all detector beams inside the collection
       aperture.
    2. **Spectrum image (qEELS).** Keep one momentum axis, and sum over the
       other axis.
    3. **Field-summed momentum spectrum.** Sum the spectrum image over the
       scan.
    4. **Energy-momentum stack.** Repeat the momentum spectrum for a series
       of energies above the edge threshold.

    Because all modes add the same intensities, the sum over momentum must
    give the map. This notebook checks that invariant. It is the
    oxygen-row test of the supplementary material (section S8) of the
    BiP-PRISM paper.

    **What to look at.** The printed residual of the invariant must be at the
    level of float64 rounding (about 1e-16). The three oxygen columns are
    equivalent, so their map peaks and their energy-momentum diagrams must
    agree.

    **Paper:** P. Pelz, *The bi-partitioned PRISM (BiP-PRISM) algorithm for
    fast and scalable core-loss STEM-EELS simulations*, *J. Phys. Mater.*
    (2026), doi:[10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0).

    **Data:** none. The notebook builds the specimen.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a GPU
       (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells. The default settings take less than a minute.

    ## Hydrogenic states, as in the paper

    This test uses analytic hydrogenic bound states and a free continuum, as
    the paper does for it. The cell below selects them explicitly, so you
    see no `HydrogenicFallbackWarning`. The invariant does not depend on the
    transition potentials. The absolute values are not quantitative.
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
    ## 1. The specimen: a row of three oxygen columns

    The cubic cell has 12 Å sides. Three O columns sit on one row, 4 Å apart,
    with two atoms each along the beam. The grid has 96 × 96 pixels and
    4 slices. The probe has 100 kV and a 20 mrad semi-angle. The EELS
    collection semi-angle is 40 mrad.
    """)
    return


@app.cell
def _(np, torch):
    from scatterem.physics.atom_set import AtomSet
    from scatterem.simulation.eels import PrismEelsSimulator

    CELL = 12.0  # Angstrom
    COLUMNS_X = (2.0, 6.0, 10.0)  # x of the three O columns, Angstrom
    o_row = AtomSet(
        np.array([CELL, CELL, CELL]),  # box sides (y, x, depth), Angstrom
        np.array([[0.5, x / CELL, (z + 0.5) / 2] for x in COLUMNS_X for z in range(2)]),
        np.full(6, 8),  # oxygen
        msd=None,
    )
    SCAN = (24, 24)
    sim = PrismEelsSimulator(
        o_row,
        eV=100e3,
        semiconvergence_angle=20.0,  # mrad
        pixels=(96, 96),
        scan=SCAN,
        num_slices=4,
        device="cuda",
        dtype=torch.complex128,  # float64 shows the invariant at round-off
    )
    EDGE = (8, 1, 0)  # O K
    DETECTOR_MRAD = 40.0
    TRANSITIONS = dict(wavefunctions="hydrogenic")
    return CELL, DETECTOR_MRAD, EDGE, SCAN, TRANSITIONS, sim


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Settings

    `S1 rings` and `S2 rings` set the partition of the probe matrix and of
    the detector matrix (`n_radial`). The detector aperture is twice the
    probe aperture, so it needs more rings: with 4 detector rings the
    equivalent columns differ by a few percent. `epsilon` is the energy of
    the ejected electron above the threshold. The energy sweep sets the
    energy-momentum stack.
    """)
    return


@app.cell
def _(mo):
    n_s1 = mo.ui.dropdown(
        options=["2", "4", "8"], value="4", label="S1 rings (probe matrix)"
    )
    n_s2 = mo.ui.dropdown(
        options=["4", "8", "16"], value="16", label="S2 rings (detector matrix)"
    )
    epsilon = mo.ui.slider(
        1.0, 50.0, value=1.0, step=1.0, label="epsilon above threshold (eV)"
    )
    resolved_axis = mo.ui.dropdown(
        options={"x (axis 1)": 1, "y (axis 0)": 0},
        value="x (axis 1)",
        label="resolved momentum axis",
    )
    n_energies = mo.ui.slider(
        5, 40, value=20, step=5, label="energies in the energy-momentum stack"
    )
    mo.vstack([mo.hstack([n_s1, n_s2]), epsilon, resolved_axis, n_energies])
    return epsilon, n_energies, n_s1, n_s2, resolved_axis


@app.cell
def _(n_s1, n_s2):
    OPTIONS = dict(
        inelastic_crop=None,  # full grid: exact contraction
        prism_mask=False,
        partition=dict(n_radial=int(n_s1.value), focal_backprop="centroid"),
        partition_s2=dict(n_radial=int(n_s2.value)),
    )
    return (OPTIONS,)


@app.cell
def _(DETECTOR_MRAD, EDGE, OPTIONS, TRANSITIONS, epsilon, resolved_axis, sim):
    _common = dict(
        detector_mrad=DETECTOR_MRAD,
        epsilon=float(epsilon.value),
        transitions=TRANSITIONS,
        **OPTIONS,
    )
    emap = sim.eels_map(EDGE, **_common)
    spectrum = sim.qeels(EDGE, axis=resolved_axis.value, **_common)
    beams = sim.qeels(EDGE, axis="both", **_common)
    return beams, emap, spectrum


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The invariant: sum over momentum equals the map

    The cell sums each momentum-resolved output over its momentum axis and
    compares the sum with the map. The residual is the largest absolute
    difference, divided by the largest map value.
    """)
    return


@app.cell
def _(beams, emap, spectrum):
    _peak = emap.data.abs().max()
    residual_axis = float((spectrum.data.sum(-1) - emap.data).abs().max() / _peak)
    residual_beams = float((beams.data.sum(-1) - emap.data).abs().max() / _peak)
    print(f"one resolved axis: max |sum_q - map| / max(map) = {residual_axis:.2e}")
    print(f"every beam kept:   max |sum_q - map| / max(map) = {residual_beams:.2e}")
    print("paper (older code): 2.69e-16")
    assert max(residual_axis, residual_beams) < 1e-12
    return (residual_axis,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Map, spectrum image, momentum spectrum and core-loss PACBED

    (a) The elemental map. The dashed line marks the scan row of panel (b).
    (b) The spectrum image along that row: scan position on the horizontal
    axis, momentum on the vertical axis. The horizontal axis is scan
    position, not energy loss. (c) The momentum spectrum, summed over the
    whole scan. (d) Core-loss PACBED: every detector beam kept, averaged over
    the scan.
    """)
    return


@app.cell
def _(CELL, SCAN, beams, emap, np, plt, sim, spectrum, torch):
    import scatterem.vis as vis
    from scatterem.data import Sampling

    UNIT = "probability per electron per eV"
    _cmap = vis.colormap_for_role("intensity").cmap
    _scan_step = CELL / SCAN[1]
    _map = emap.data.double().cpu().numpy()
    row = int(np.argmax(_map.sum(axis=1)))
    # theta = lambda * q: grid pixels -> 1/Angstrom -> mrad
    _q_mrad = (
        spectrum.q_pixels.double().cpu().numpy()
        / sim.gridsize[1]
        * sim.wavelength
        * 1e3
    )
    _spec = spectrum.data.double().cpu().numpy()

    def scaled(values):
        """Divide by a power of ten, so that no axis needs an offset label."""
        _e = int(np.floor(np.log10(np.nanmax(np.abs(values)))))
        return values / 10.0**_e, f"{UNIT} (×10$^{{{_e}}}$)"

    _map, _map_unit = scaled(_map)
    _row_spec, _row_unit = scaled(_spec[row].T)
    _sum_spec, _sum_unit = scaled(_spec.sum(axis=(0, 1)))

    fig_modes, _axs = plt.subplots(1, 4, figsize=(17, 4.2))
    _n_axes = len(fig_modes.axes)
    vis.show_2d(
        [_map],
        figax=(fig_modes, _axs[0]),
        sampling=Sampling(pixel_size=(CELL / SCAN[0], _scan_step), units=("Å", "Å")),
        role="intensity",
        title=["(a) O K map"],
        tight_layout=False,
    )
    fig_modes.axes[_n_axes].set_ylabel(_map_unit)  # the colour bar of (a)
    _axs[0].axhline(row, color="w", lw=1.0, ls="--")

    _im = _axs[1].imshow(
        _row_spec,
        origin="lower",
        aspect="auto",
        cmap=_cmap,
        extent=[0, CELL, _q_mrad.min(), _q_mrad.max()],
    )
    fig_modes.colorbar(_im, ax=_axs[1], label=_row_unit)
    _axs[1].set_xlabel("scan position x (Å)")
    _axs[1].set_ylabel("momentum q (mrad)")
    _axs[1].set_title("(b) spectrum image, marked row")

    _axs[2].plot(_q_mrad, _sum_spec, color="k")
    _axs[2].set_xlabel("momentum q (mrad)")
    _axs[2].set_ylabel(f"sum over scan, {_sum_unit}")
    _axs[2].set_title("(c) field-summed momentum spectrum")
    _axs[2].grid(alpha=0.3)

    # Put each kept detector beam on its (qy, qx) grid pixel.
    _qp = beams.q_pixels.long().cpu().numpy()
    _half = int(np.abs(_qp).max())
    pacbed = np.full((2 * _half + 1, 2 * _half + 1), np.nan)
    pacbed[_qp[:, 0] + _half, _qp[:, 1] + _half] = (
        beams.data.mean(dim=(0, 1)).double().cpu().numpy()
    )
    pacbed, _pacbed_unit = scaled(pacbed)
    _dq = sim.wavelength / sim.gridsize[0] * 1e3  # mrad per detector pixel
    _n_axes = len(fig_modes.axes)
    vis.show_2d(
        [pacbed],
        figax=(fig_modes, _axs[3]),
        sampling=Sampling(pixel_size=(_dq, _dq), units=("mrad", "mrad")),
        role="diffraction",
        title=["(d) core-loss PACBED"],
        tight_layout=False,
    )
    fig_modes.axes[_n_axes].set_ylabel(_pacbed_unit)  # the colour bar of (d)
    fig_modes.tight_layout()
    torch.cuda.synchronize()
    fig_modes
    return Sampling, UNIT, fig_modes, row, vis


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Energy-momentum stack at three probe positions

    The energy loss enters through `epsilon`, the energy of the ejected
    electron above the threshold. The cell repeats the momentum spectrum for
    a series of `epsilon` values from 1 eV to 100 eV at three probe
    positions: on the first column, on the third column, and midway between
    two columns. Energy loss is on the horizontal axis (O K onset 532 eV
    plus `epsilon`), and momentum on the vertical axis. The two column
    positions are equivalent, so their diagrams must agree. Each panel has
    its own colour scale, because the signal between the columns is much
    weaker.

    With the hydrogenic continuum, the signal grows with energy over this
    range. A real O K edge falls after its onset. Do not read the energy
    dependence quantitatively.
    """)
    return


@app.cell
def _(
    CELL,
    DETECTOR_MRAD,
    EDGE,
    OPTIONS,
    TRANSITIONS,
    n_energies,
    np,
    resolved_axis,
    sim,
    torch,
):
    from scatterem.simulation.eels import prism_eels_image

    O_K_ONSET = 532.0  # eV; sets only the energy-loss axis
    POSITIONS_X = (2.0, 10.0, 4.0)  # Angstrom: column 1, column 3, between
    _px = sim.pixels[1] / CELL
    _scan = torch.tensor(
        [[6.0 * _px, _x * _px] for _x in POSITIONS_X],
        dtype=torch.float64,
        device=sim.device,
    )
    _T, _probe, _sites = sim.transmissions(), sim.probe_q(), sim.sites(EDGE[0])
    eps_values = np.linspace(1.0, 100.0, int(n_energies.value))
    _stack = []
    for _e in eps_values:
        _tp = sim.transition_potentials(EDGE, float(_e), **TRANSITIONS)
        _spec, _qpix = prism_eels_image(
            _probe,
            _T,
            _tp,
            _sites,
            _scan,
            wavelength=sim.wavelength,
            gridsize=sim.gridsize,
            slice_distance=sim.slice_distance,
            detector_mrad=DETECTOR_MRAD,
            scan_shape=(1, 3),
            qeels_axis=resolved_axis.value,
            **OPTIONS,
        )
        _stack.append(_spec.reshape(3, -1).double().cpu().numpy())
    omega_q = np.stack(_stack)  # (energy, position, q)
    omega_q_mrad = _qpix.double().cpu().numpy() / sim.gridsize[1] * sim.wavelength * 1e3
    column_difference = float(
        np.abs(omega_q[:, 0] - omega_q[:, 1]).max() / np.abs(omega_q[:, 0]).max()
    )
    print(f"equivalent columns: max |difference| / max = {column_difference:.2e}")
    return O_K_ONSET, POSITIONS_X, eps_values, omega_q, omega_q_mrad


@app.cell
def _(
    O_K_ONSET,
    POSITIONS_X,
    UNIT,
    eps_values,
    np,
    omega_q,
    omega_q_mrad,
    plt,
    vis,
):
    _cmap = vis.colormap_for_role("intensity").cmap
    # One power of ten for all three panels, so that no colour bar needs an
    # offset label.
    _e = int(np.floor(np.log10(float(np.nanmax(np.abs(omega_q))))))
    _omega = omega_q / 10.0**_e
    fig_omega, _axs = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    _names = ("on column 1", "on column 3", "between columns")
    for _k, _ax in enumerate(_axs):
        _im = _ax.imshow(
            _omega[:, _k].T,
            origin="lower",
            aspect="auto",
            cmap=_cmap,
            vmin=0.0,
            extent=[
                O_K_ONSET + eps_values[0],
                O_K_ONSET + eps_values[-1],
                omega_q_mrad.min(),
                omega_q_mrad.max(),
            ],
        )
        fig_omega.colorbar(_im, ax=_ax, label=f"{UNIT} (×10$^{{{_e}}}$)")
        _ax.set_xlabel("energy loss (eV)")
        _ax.set_title(f"x = {POSITIONS_X[_k]:.0f} Å, {_names[_k]}")
    _axs[0].set_ylabel("momentum q (mrad)")
    fig_omega.tight_layout()
    fig_omega
    return


@app.cell
def _(mo, residual_axis):
    mo.md(rf"""
    ## Differences from the paper

    - **Software version.** The paper used older code. Since v0.4.0, a map
      value is a probability per incident electron and per eV. v0.4 builds
      the transition potentials with a new first-Born model, and it builds
      the specimen with a new physics layer (projected potentials,
      transmissions and a soft-edged probe aperture). So the absolute values
      differ from the paper. The page "Simulate STEM-EELS" of the
      documentation compares v0.4 with the paper code: the map changes shape
      by 43-60 % with the paper's specimen setup, mostly through the new
      hydrogenic continuum states.
    - **The invariant does not change.** This run gives a residual of
      {residual_axis:.1e}. The paper gives 2.69e-16. Both are float64
      rounding.
    - **Energy sweep.** The paper stacks 40 energies. The default here is 20.
    - **Hardware.** The paper ran this test on a CPU. This notebook runs on
      the GPU, in complex128.
    """)
    return


@app.cell
def _(residual_axis, torch):
    assert residual_axis < 1e-12
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
