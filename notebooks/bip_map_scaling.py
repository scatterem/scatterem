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
"""BiP-PRISM map scaling: time and GPU memory against thickness, grid and scan, as a marimo notebook."""

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
    # BiP-PRISM map scaling: time and memory on your GPU

    This notebook measures what an elemental map costs on the GPU that runs
    it. It compares two calculations of the same Ti L map of SrTiO₃:

    - **exact PRISM**: the dual scattering matrix with every beam;
    - **BiP-PRISM**: both scattering matrices partitioned into 4 rings.

    It sweeps three quantities, one at a time, as in figure 2 of the paper:
    (a) time against specimen thickness, (b) peak GPU memory against grid
    size, and (c) time against the number of scan positions. CUDA events
    time each call, and `torch.cuda.max_memory_allocated` gives the peak
    memory. One warm-up call runs first.

    **What to look at.** (a) Both PRISM maps scale linearly with thickness.
    (b) The exact map needs much more memory than BiP-PRISM, and the gap
    grows with the grid. (c) The time grows with the number of scan
    positions, because each position needs its own contraction. On the
    small 128² grid, this contraction dominates, so exact PRISM and
    BiP-PRISM take almost the same time and their curves overlap. The
    printed relative error shows what the memory saving costs in accuracy.

    **Paper:** P. Pelz, *The bi-partitioned PRISM (BiP-PRISM) algorithm for
    fast and scalable core-loss STEM-EELS simulations*, *J. Phys. Mater.*
    (2026), doi:[10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0),
    figure 2 and supplementary table S2.

    **Data:** none. The notebook builds the specimen.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a GPU
       (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells. The default ranges take about 1.5 minutes on an
       RTX PRO 6000 (about 4 minutes on an RTX A6000) and about 28 GB of GPU
       memory.
    3. Do not run other GPU work at the same time: it changes the timings.

    ## Hydrogenic states, as in the paper

    This study uses analytic hydrogenic bound states, as the paper does for
    it. The timings and the memory do not depend on the transition-potential
    values. The notebook selects the hydrogenic states explicitly, so you
    see no `HydrogenicFallbackWarning`.
    """)
    return


@app.cell
def _():
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On molab, select a GPU in the notebook specs "
            "and restart the notebook."
        )
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"GPU: {GPU_NAME}")
    return GPU_NAME, np, plt, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Settings

    The specimen is SrTiO₃ with 8 × 8 unit cells laterally and $N_Z$ cells
    along the beam, with 2 slices per cell. The probe has 100 kV and a
    30 mrad semi-angle. The EELS detector has 50 mrad. PRISM factor $f = 1$,
    crop divisor 16 and `contract="probe"`, as in supplementary table S2.
    The scan has $P = S \times S$ positions over the central unit cell.

    - The thickness sweep uses a 384 × 384 grid and an 8 × 8 scan.
    - The grid sweep and the scan sweep use 26 cells (10 nm).
    - The grid sweep uses an 8 × 8 scan, the scan sweep a 128 × 128 grid.

    Exact PRISM needs about 30 GB at 384 × 384 and about 50 GB at
    512 × 512. The default runs exact PRISM up to 384 × 384.

    `float32 matmul precision`: v0.4 keeps the torch default `"highest"`.
    `"medium"` uses TF32 and gives the speed of v0.3 (see "The cost of a
    call" in the documentation).
    """)
    return


@app.cell
def _(mo):
    thickness_cells = mo.ui.multiselect(
        options=["2", "5", "10", "20", "40", "77"],
        value=["2", "5", "10", "20"],
        label="thickness sweep: cells along the beam",
    )
    grids = mo.ui.multiselect(
        options=["128", "192", "256", "384", "512", "768", "1024"],
        value=["128", "256", "384", "512"],
        label="grid sweep: pixels per side",
    )
    scans = mo.ui.multiselect(
        options=["8", "16", "32", "64"],
        value=["8", "16", "32"],
        label="scan sweep: positions per side",
    )
    exact_max_grid = mo.ui.dropdown(
        options=["256", "384", "512"],
        value="384",
        label="largest grid for exact PRISM",
    )
    precision = mo.ui.dropdown(
        options=["highest", "medium"],
        value="highest",
        label="float32 matmul precision",
    )
    mo.vstack([thickness_cells, grids, scans, mo.hstack([exact_max_grid, precision])])
    return exact_max_grid, grids, precision, scans, thickness_cells


@app.cell
def _(np, precision, time, torch):
    import scatterem
    from scatterem.physics.atom_set import AtomSet
    from scatterem.simulation.eels import PrismEelsSimulator, prism_eels_image

    scatterem.configure(matmul_precision=precision.value)
    A_STO = 3.905  # Angstrom
    N_LAT = 8
    TI_L = (22, 2, 1)
    _unit = AtomSet(
        np.array([A_STO, A_STO, A_STO]),
        np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        ),
        np.array([38, 22, 8, 8, 8]),  # Sr, Ti, O, O, O
        msd=None,
    )

    def inputs(pixels, n_z, scan_side):
        """Kernel inputs of one operating point (built before the timer)."""
        sim = PrismEelsSimulator(
            _unit.replicated((N_LAT, N_LAT, n_z)),
            eV=100e3,
            semiconvergence_angle=30.0,  # mrad
            pixels=(pixels, pixels),
            scan=(scan_side, scan_side),
            num_slices=2 * n_z,
            device="cuda",
        )
        # Scan positions over the central unit cell, in grid pixels.
        _f = ((N_LAT // 2) + (np.arange(scan_side) + 0.5) / scan_side) / N_LAT
        _y, _x = np.meshgrid(_f * pixels, _f * pixels, indexing="ij")
        scan = torch.as_tensor(
            np.column_stack([_y.ravel(), _x.ravel()]),
            dtype=torch.float64,
            device="cuda",
        )
        tp = sim.transition_potentials(TI_L, 1.0, wavefunctions="hydrogenic")
        args = (sim.probe_q(), sim.transmissions(), tp, sim.sites(TI_L[0]), scan)
        common = dict(
            wavelength=sim.wavelength,
            gridsize=sim.gridsize,
            slice_distance=sim.slice_distance,
            detector_mrad=50.0,
            interpolation_factor=1,
            inelastic_crop=16,
            prism_mask=False,
            scan_shape=(scan_side, scan_side),
            contract="probe",
        )
        return args, common

    PARTITION = dict(partition=dict(n_radial=4), partition_s2=dict(n_radial=4))

    PEAKS = []  # peak bytes of every call, for the last cell

    def measure(pixels, n_z, scan_side, partitioned):
        """Time (s), peak memory (GB) and map of one call; None on out-of-memory."""
        args = None
        try:
            args, common = inputs(pixels, n_z, scan_side)
            options = PARTITION if partitioned else {}
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            emap = prism_eels_image(*args, **common, **options)
            stop.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(stop) / 1e3
            PEAKS.append(torch.cuda.max_memory_allocated())
            peak = PEAKS[-1] / 1e9  # decimal GB
            result = (seconds, peak, emap.detach().cpu())
        except torch.cuda.OutOfMemoryError:
            result = None
        del args
        torch.cuda.empty_cache()
        return result

    # One warm-up call (CUDA start-up, FFT plans, kernel compilation).
    _t0 = time.perf_counter()
    measure(128, 2, 8, True)
    measure(128, 2, 8, False)
    print(f"warm-up: {time.perf_counter() - _t0:.1f} s")
    return A_STO, PEAKS, measure


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The three sweeps

    Each point builds its inputs, then times one call of exact PRISM and one
    of BiP-PRISM. The relative error compares the two maps.
    """)
    return


@app.cell
def _(A_STO, exact_max_grid, grids, measure, scans, thickness_cells):
    def sweep(points, exact_limit=None):
        rows = []
        for pixels, n_z, scan_side in points:
            part = measure(pixels, n_z, scan_side, True)
            exact = None
            if exact_limit is None or pixels <= exact_limit:
                exact = measure(pixels, n_z, scan_side, False)
            error = float("nan")
            if part is not None and exact is not None:
                error = float((part[2] - exact[2]).norm() / exact[2].norm())
            rows.append(
                dict(
                    pixels=pixels,
                    n_z=n_z,
                    scan=scan_side,
                    part=None if part is None else part[:2],
                    exact=None if exact is None else exact[:2],
                    error=error,
                )
            )
            _p = "OOM" if part is None else f"{part[0]:6.2f} s {part[1]:6.2f} GB"
            _e = (
                "skipped"
                if exact_limit is not None and pixels > exact_limit
                else (
                    "OOM" if exact is None else f"{exact[0]:6.2f} s {exact[1]:6.2f} GB"
                )
            )
            print(
                f"  grid {pixels:4d}, {n_z:3d} cells ({n_z * A_STO / 10:5.1f} nm), "
                f"scan {scan_side:2d}^2: BiP-PRISM {_p} | exact {_e} | "
                f"error {100 * error:.2f} %"
            )
        return rows

    print("(a) thickness")
    thickness_rows = sweep(
        [(384, int(_n), 8) for _n in sorted(thickness_cells.value, key=int)]
    )
    print("(b) grid")
    grid_rows = sweep(
        [(int(_g), 26, 8) for _g in sorted(grids.value, key=int)],
        exact_limit=int(exact_max_grid.value),
    )
    print("(c) scan")
    scan_rows = sweep([(128, 26, int(_s)) for _s in sorted(scans.value, key=int)])
    return grid_rows, scan_rows, thickness_rows


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Scaling

    Coloured symbols: this run. Grey symbols: the paper, measured on an
    NVIDIA A100 (40 GB) with older code. Compare the trends, not the
    absolute times: they depend on the GPU and on the matmul precision.
    """)
    return


@app.cell
def _(A_STO, GPU_NAME, grid_rows, np, plt, scan_rows, thickness_rows):
    # Paper values (A100 40 GB; figure 2 and its data).
    PAPER_THICKNESS = dict(
        n_z=[2, 3, 5, 8, 13, 21, 34, 51, 64, 77],
        part=[0.832, 1.245, 2.088, 3.309, 5.351, 8.607, 13.993, 21.427, 26.51, 31.791],
        exact=[
            2.244,
            2.929,
            4.302,
            6.376,
            9.837,
            15.308,
            24.335,
            36.039,
            45.071,
            53.954,
        ],
    )
    PAPER_GRID = dict(
        pixels=[128, 192, 256, 384, 512, 768, 1024],
        part=[0.084, 0.155, 0.258, 0.551, 0.955, 2.126, 3.757],
        exact=[3.242, 7.27, 12.909, 29.02, np.nan, np.nan, np.nan],
    )
    PAPER_SCAN = dict(
        positions=[64, 256, 1024, 4096],
        part=[9.98, 10.168, 18.238, 57.152],
        exact=[8.839, 8.812, 16.926, 55.885],
    )

    def column(rows, key, index):
        return np.array([np.nan if _r[key] is None else _r[key][index] for _r in rows])

    fig_scaling, _axs = plt.subplots(1, 3, figsize=(16, 4.6))
    _ax = _axs[0]
    _nm = np.array([_r["n_z"] for _r in thickness_rows]) * A_STO / 10
    _ax.plot(
        _nm, column(thickness_rows, "exact", 0), "o-", color="C0", label="exact PRISM"
    )
    _ax.plot(
        _nm, column(thickness_rows, "part", 0), "s-", color="C2", label="BiP-PRISM"
    )
    _pnm = np.array(PAPER_THICKNESS["n_z"]) * A_STO / 10
    _ax.plot(
        _pnm,
        PAPER_THICKNESS["exact"],
        "o--",
        color="0.6",
        mfc="none",
        label="exact, paper",
    )
    _ax.plot(
        _pnm,
        PAPER_THICKNESS["part"],
        "s--",
        color="0.6",
        mfc="none",
        label="BiP, paper",
    )
    _ax.set_xlabel("thickness (nm)")
    _ax.set_ylabel("time per map (s)")
    _ax.set_title("(a) time against thickness, 384² grid")

    _ax = _axs[1]
    _g = np.array([_r["pixels"] for _r in grid_rows])
    _ax.loglog(_g, column(grid_rows, "exact", 1), "o-", color="C0", label="exact PRISM")
    _ax.loglog(_g, column(grid_rows, "part", 1), "s-", color="C2", label="BiP-PRISM")
    _ax.loglog(
        PAPER_GRID["pixels"],
        PAPER_GRID["exact"],
        "o--",
        color="0.6",
        mfc="none",
        label="exact, paper",
    )
    _ax.loglog(
        PAPER_GRID["pixels"],
        PAPER_GRID["part"],
        "s--",
        color="0.6",
        mfc="none",
        label="BiP, paper",
    )
    _ax.set_xlabel("grid (pixels per side)")
    _ax.set_ylabel("peak GPU memory (GB)")
    _ax.set_title("(b) memory against grid, 10 nm")

    _ax = _axs[2]
    _p = np.array([_r["scan"] ** 2 for _r in scan_rows])
    _ax.loglog(_p, column(scan_rows, "exact", 0), "o-", color="C0", label="exact PRISM")
    _ax.loglog(_p, column(scan_rows, "part", 0), "s-", color="C2", label="BiP-PRISM")
    _ax.loglog(
        PAPER_SCAN["positions"],
        PAPER_SCAN["exact"],
        "o--",
        color="0.6",
        mfc="none",
        label="exact, paper",
    )
    _ax.loglog(
        PAPER_SCAN["positions"],
        PAPER_SCAN["part"],
        "s--",
        color="0.6",
        mfc="none",
        label="BiP, paper",
    )
    _ax.set_xlabel("scan positions P")
    _ax.set_ylabel("time per map (s)")
    _ax.set_title("(c) time against scan, 128² grid")
    for _a in _axs:
        _a.grid(alpha=0.3, which="both")
        _a.legend(fontsize=8)
    fig_scaling.suptitle(f"this run: {GPU_NAME}", fontsize=10)
    fig_scaling.tight_layout()
    fig_scaling
    return (column,)


@app.cell
def _(A_STO, column, grid_rows, mo, np, thickness_rows):
    def exponent(x, y):
        _ok = np.isfinite(y) & (y > 0)
        if _ok.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(x[_ok]), np.log(y[_ok]), 1)[0])

    _nm = np.array([_r["n_z"] for _r in thickness_rows]) * A_STO / 10
    _k_exact = exponent(_nm, column(thickness_rows, "exact", 0))
    _k_part = exponent(_nm, column(thickness_rows, "part", 0))
    _ratios = [
        (_r["pixels"], _r["exact"][1] / _r["part"][1])
        for _r in grid_rows
        if _r["exact"] is not None and _r["part"] is not None
    ]
    _ratio_text = ", ".join(f"{_g}²: {_q:.0f}x" for _g, _q in _ratios)
    _errors = ", ".join(
        f"{_r['n_z'] * A_STO / 10:.1f} nm: {100 * _r['error']:.1f} %"
        for _r in thickness_rows
    )
    mo.md(
        f"""
    ## 4. Summary

    - Time exponent in thickness (fit of log time against log thickness):
      exact PRISM {_k_exact:.2f}, BiP-PRISM {_k_part:.2f}. The paper code
      and v0.4 give 1.00 for both.
    - Memory of exact PRISM over BiP-PRISM: {_ratio_text}. The paper gives
      53x at 384² (29.0 GB against 0.551 GB).
    - Partitioning error of the map against exact PRISM: {_errors}. The
      paper gives 0.01 % at 0.8 nm and 15.3 % at 30 nm; the error grows
      with thickness.
    """
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
      transmissions and a soft-edged probe aperture). The timings and the
      memory do not depend on these values. The page "Simulate STEM-EELS"
      of the documentation compares v0.4 with the paper code in detail.
    - **Hardware.** The paper measured on one A100 (40 GB), with three
      repetitions after a warm-up. This notebook measures one call per point
      on your GPU. On molab (RTX PRO 6000, 96 GB), exact PRISM also fits at
      512 × 512, where the A100 ran out of memory.
    - **Reduced ranges.** The paper sweeps thickness to 77 cells (30 nm),
      grids to 1024² and scans to 64². The defaults stop at 20 cells, 512²
      and 32², to keep the run short. Select the larger values to extend
      the sweeps.
    - **Scan sweep.** In the paper, the time stays flat up to 16² positions
      and then grows. v0.4 is faster per call at small scans, so the time
      grows from the first point. The page "Simulate STEM-EELS" of the
      documentation reports a linear fit over 8², 32² and 64² scans on one
      GPU: 10.1 ms per scan position for the paper code and 11.4 ms for
      v0.4.
    - **No conventional multislice.** The paper also times conventional
      transition-potential multislice where it is tractable. This notebook
      compares only the two PRISM maps.
    """)
    return


@app.cell
def _(PEAKS, thickness_rows, torch):
    assert all(_r["part"] is not None for _r in thickness_rows)
    # Each call resets the peak counter, so take the largest peak of all calls.
    _peak = max(PEAKS + [torch.cuda.max_memory_allocated()])
    print(f"peak GPU memory: {_peak / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
