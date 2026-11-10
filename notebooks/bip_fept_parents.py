# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.4.1",
#     "marimo==0.25.2",
#     "matplotlib>=3.10",
#     "numpy>=2.1.0",
#     "torch>=2.5.0",
# ]
# ///
"""BiP-PRISM Fe L map of a FePt nanoparticle and its parent-ring sweep."""

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
    # BiP-PRISM: Fe L map of a FePt nanoparticle, and the parent sweep

    This notebook repeats the FePt parent sweep of the BiP-PRISM paper at a
    reduced size. The specimen is a real FePt nanoparticle with 23 196 atoms
    (6 569 Fe). The notebook calculates the Fe L elemental map with
    BiP-PRISM, which partitions both scattering matrices on *parent* beams.
    Then it calculates the same map without partitioning, at the same PRISM
    factor *f*, and compares the two.

    **What to look at:**

    1. The map: the Fe signal of the whole particle, at atomic resolution.
    2. The error curve: the relative L2 error against the unpartitioned map
       falls quickly when the number of parent rings increases.
    3. The memory column: the partitioned calls need much less GPU memory
       than the unpartitioned call. This is the purpose of BiP-PRISM.

    **Paper:** P. Pelz, *The bi-partitioned PRISM (BiP-PRISM) algorithm for
    fast and scalable core-loss STEM-EELS simulations*, *J. Phys. Mater.*
    (2026), doi:[10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0).

    **Data:** the atomic model is the atomic-electron-tomography
    reconstruction of Y. Yang *et al.*, *Nature* **542**, 75-79 (2017),
    doi:[10.1038/nature21042](https://doi.org/10.1038/nature21042).
    Brown, Ciston and Ophus used the same model for their PRISM-EELS
    benchmark: *Phys. Rev. Research* **1**, 033186 (2019),
    doi:[10.1103/PhysRevResearch.1.033186](https://doi.org/10.1103/PhysRevResearch.1.033186).
    The model file is third-party data. `scatterem` does not redistribute
    it. The notebook downloads it from the data page of the authors,
    [physics.ucla.edu/research/imaging/FePt](https://www.physics.ucla.edu/research/imaging/FePt/).
    That page makes the data freely available and asks users to cite the
    Nature paper. It gives no explicit licence. If you use the model, cite
    Yang *et al.* (2017).

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a GPU
       (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells.

    The model (0.8 MB) downloads on first use, and the notebook checks its
    SHA-256. Set `FEPT_DATA_ROOT` to keep it in a different directory (the
    default is `data`).
    """)
    return


@app.cell
def _():
    import hashlib
    import os
    import time
    import urllib.request
    import warnings
    import zipfile
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On molab, select a GPU in the notebook specs "
            "and restart the notebook."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return Path, hashlib, np, os, plt, time, torch, urllib, warnings, zipfile


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Get the atomic model

    The cell downloads `Final_Model.zip` from the data page of Yang *et al.*
    and checks the SHA-256 of the archive and of the model file in it. A
    different file stops the notebook: the numbers below are only valid for
    this model.
    """)
    return


@app.cell
def _(Path, hashlib, os, urllib, zipfile):
    MODEL_URL = (
        "https://www.physics.ucla.edu/research/imaging/FePt/Data/Final_Model.zip"
    )
    ZIP_SHA256 = "0726ce9393b16fab0e111cbbeb07795a41edde375753fea2f0bbdfcced442a66"
    XYZ_NAME = "FePt_FullParticleModel_FinalModel.xyz"
    XYZ_SHA256 = "5c7a6e462f15cfdb66db88059b6202a9f2b55fc93d0dc4129fc33d8945578dd1"

    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    data_root = Path(os.environ.get("FEPT_DATA_ROOT", "data")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    xyz_path = data_root / XYZ_NAME
    if not (xyz_path.is_file() and _sha256(xyz_path) == XYZ_SHA256):
        _zip_path = data_root / "Final_Model.zip"
        if not (_zip_path.is_file() and _sha256(_zip_path) == ZIP_SHA256):
            print(f"download: {MODEL_URL}")
            _request = urllib.request.Request(
                MODEL_URL, headers={"User-Agent": "scatterem-notebook"}
            )
            with urllib.request.urlopen(_request, timeout=120) as _response:
                _zip_path.write_bytes(_response.read())
        if _sha256(_zip_path) != ZIP_SHA256:
            raise RuntimeError(
                f"{_zip_path} does not have the expected SHA-256. The file on "
                "the data page changed. Stop here: do not use another model."
            )
        with zipfile.ZipFile(_zip_path) as _archive:
            xyz_path.write_bytes(_archive.read(XYZ_NAME))
    if _sha256(xyz_path) != XYZ_SHA256:
        raise RuntimeError(f"{xyz_path} does not have the expected SHA-256.")
    print(f"model: {xyz_path} (SHA-256 verified)")
    return (xyz_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Build the structure

    The file has one line per atom: the element and the Cartesian
    coordinates (x, y, z) in Å. The beam goes along z, as in the paper. The
    box is the largest coordinate along each axis
    (93.9 × 93.7 × 97.8 Å), and the particle sits at the box centre, so the
    vacuum is the same on all sides. The simulator draws no thermal
    displacement (one static configuration), as in the paper.
    """)
    return


@app.cell
def _(np, xyz_path):
    from scatterem.physics.atom_set import AtomSet

    _rows = [
        _line.split()
        for _line in xyz_path.read_text().splitlines()[2:]
        if _line.strip()
    ]
    _species = np.array([_row[0] for _row in _rows])
    _xyz = np.array([[float(_v) for _v in _row[1:4]] for _row in _rows])
    Z = np.where(_species == "Fe", 26, 78)
    box = _xyz.max(axis=0)
    _xyz = _xyz + (0.5 * box - 0.5 * (_xyz.min(axis=0) + _xyz.max(axis=0)))
    fept = AtomSet(box, _xyz / box, Z, msd=None)  # no thermal draw below
    print(
        f"{len(Z)} atoms ({int((Z == 26).sum())} Fe, {int((Z == 78).sum())} Pt), "
        f"box {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} Angstrom"
    )
    return box, fept


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Choose the settings

    The paper uses a 936² grid (0.1 Å pixels), a 94 × 94 scan (1 Å steps),
    49 slices of 2 Å, 300 kV, a 20 mrad probe and a 20 mrad EELS collection
    angle, crop divisor 22, and *f* = 4 and *f* = 8.

    **The default is the paper's grid and scan at *f* = 8 only.** The sweep
    then takes about 1.5 minutes and 17 GB on an RTX PRO 6000 (about
    4 minutes on an RTX A6000). The paper's
    *f* = 4 sweep is not the default: at 936², its unpartitioned reference
    needs 63 GB, and the hydrogenic transition potentials of this notebook
    have 27 channels instead of the 4 channels of the paper, so it takes
    much longer. To see *f* = 4, select the 480² grid (0.2 Å pixels) and
    the 47 × 47 scan (2 Å steps): that sweep takes about 4 to 6 minutes and
    17 GB.

    `n_radial` is the number of parent rings on each matrix. The nominal
    number of parent beams is 1 + 3 n (n + 1): 19, 61, 217 and 469 for 2,
    4, 8 and 12 rings. The code snaps them to the aperture beams.

    **TF32** is off by default: the float32 matrix products run at full
    float32 precision, the default of `scatterem` 0.4. Select it to use
    TF32, as the code of the paper did. TF32 moves a map by about 1e-4
    (relative L2) and makes the sweep about 1.5 times faster.
    """)
    return


@app.cell
def _(mo):
    grid_ui = mo.ui.dropdown(
        options={"936² (paper, 0.1 Å)": 936, "480² (0.2 Å)": 480},
        value="936² (paper, 0.1 Å)",
        label="grid",
    )
    scan_ui = mo.ui.dropdown(
        options={"94 × 94 (paper, 1 Å)": 94, "47 × 47 (2 Å)": 47},
        value="94 × 94 (paper, 1 Å)",
        label="scan",
    )
    f_ui = mo.ui.dropdown(options={"8": 8, "4": 4}, value="8", label="PRISM factor f")
    rings_ui = mo.ui.multiselect(
        options={"2": 2, "4": 4, "8": 8, "12": 12},
        value=["2", "4", "8", "12"],
        label="parent rings n_radial (both matrices)",
    )
    tf32_ui = mo.ui.checkbox(value=False, label="TF32 matrix products")
    mo.vstack([mo.hstack([grid_ui, scan_ui, f_ui, tf32_ui]), rings_ui])
    return f_ui, grid_ui, rings_ui, scan_ui, tf32_ui


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build the inputs once

    The transmission functions, the probe, the scan and the Fe L transition
    potentials do not depend on the partition. The cell builds them once,
    and every call below uses the same inputs. So the differences between
    the maps come only from the partitioning.

    **Transition potentials.** The paper used transition potentials from
    muSTEM. This notebook calculates them with `scatterem`
    (`wavefunctions="auto"`). With GPAW installed, `scatterem` uses GPAW
    (all-electron states). Without GPAW, it uses analytic hydrogenic bound
    states and a free continuum, and it emits the
    `HydrogenicFallbackWarning`, which the cell prints. This notebook does
    not install GPAW (its build needs a C++ compiler and is slow), so on
    molab the map uses the hydrogenic states and is **not quantitative**.
    The error *relative* to the unpartitioned map measures the partitioning,
    so the comparison stays valid.
    """)
    return


@app.cell
def _(box, f_ui, fept, grid_ui, scan_ui, tf32_ui, time, torch, warnings):
    import scatterem
    from scatterem.simulation.eels import PrismEelsSimulator, gpaw_available

    scatterem.configure(matmul_precision="medium" if tf32_ui.value else "highest")
    NPIX, NSCAN, F = grid_ui.value, scan_ui.value, f_ui.value
    WAVEFUNCTIONS = "gpaw" if gpaw_available() else "hydrogenic"
    sim = PrismEelsSimulator(
        fept,
        eV=300e3,
        semiconvergence_angle=20.0,  # mrad
        pixels=(NPIX, NPIX),
        scan=(NSCAN, NSCAN),
        num_slices=int(round(box[2] / 2.0)),  # 2 A slices: 49
        interpolation_factor=F,
        device="cuda",
    )
    _t0 = time.perf_counter()
    T = sim.transmissions()
    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        tp = sim.transition_potentials((26, 2, 1), epsilon=10.0, wavefunctions="auto")
    for _w in _caught:
        print(f"{_w.category.__name__}: {_w.message}")
    probe_q, scan_pixels, fe_sites = sim.probe_q(), sim.scan_pixels(), sim.sites(26)
    torch.cuda.synchronize()
    print(
        f"wavefunctions: {WAVEFUNCTIONS}; {sim.num_slices} slices of "
        f"{sim.slice_thickness:.2f} A; pixel {sim.sampling[0]:.3f} A; "
        f"{len(fe_sites)} Fe sites; {tp.array.shape[0]} transition channels; "
        f"inputs built in {time.perf_counter() - _t0:.1f} s"
    )
    return F, NPIX, NSCAN, WAVEFUNCTIONS, T, fe_sites, probe_q, scan_pixels, sim, tp


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Run the sweep

    The cell runs the unpartitioned reference first and then one BiP-PRISM
    call for each ring count. It records the time of each call (CUDA
    synchronised) and the peak GPU memory that the call allocates. The
    options are those of the paper: `inelastic_crop=22`,
    `mag_preserve=True`, `focal_backprop=None`. The paper forced
    `contract="columns"`; this notebook uses `"auto"`, which gives the same
    map to rounding and picks the faster order.
    """)
    return


@app.cell
def _(F, T, fe_sites, probe_q, rings_ui, scan_pixels, sim, time, torch, tp):
    from scatterem.simulation.eels import prism_eels_image

    _common = dict(
        wavelength=sim.wavelength,
        gridsize=sim.gridsize,
        slice_distance=sim.slice_distance,
        detector_mrad=20.0,
        interpolation_factor=F,
        inelastic_crop=22,
        scan_shape=sim.scan,
        contract="auto",  # same map as "columns" of the paper, to rounding
        mag_preserve=True,
        focal_backprop=None,
    )

    def _run(n_radial):
        _partition = None if n_radial == 0 else dict(n_radial=n_radial)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        _map = prism_eels_image(
            probe_q,
            T,
            tp,
            fe_sites,
            scan_pixels,
            partition=_partition,
            partition_s2=_partition,
            **_common,
        )
        torch.cuda.synchronize()
        return dict(
            map=_map.double().cpu(),
            seconds=time.perf_counter() - _t0,
            peak_bytes=torch.cuda.max_memory_allocated(),
        )

    rings = sorted(rings_ui.value)
    runs = {0: _run(0)}
    print(
        f"unpartitioned: {runs[0]['seconds']:.1f} s, {runs[0]['peak_bytes'] / 1e9:.2f} GB"
    )
    for _n in rings:
        runs[_n] = _run(_n)
        print(
            f"n_radial={_n}: {runs[_n]['seconds']:.1f} s, "
            f"{runs[_n]['peak_bytes'] / 1e9:.2f} GB"
        )
    return rings, runs


@app.cell
def _(F, mo, rings, runs, torch):
    # Paper values (corrected parent sweep, 936^2 grid, 94 x 94 scan,
    # muSTEM transition potentials), relative L2 error in % per ring count.
    PAPER_ERR = {
        4: {2: 8.746, 4: 1.670, 8: 0.505, 12: 0.247},
        8: {2: 7.878, 4: 1.596, 8: 0.449, 12: 0.078},
    }
    _ref = runs[0]["map"]
    errors = {}
    _lines = [
        "| parent rings | relative L2 error | Pearson r | intensity ratio "
        "| time | peak GPU memory | paper error (936², muSTEM) |",
        "|---|---|---|---|---|---|---|",
        f"| none (reference) | 0 | 1 | 1 | {runs[0]['seconds']:.1f} s "
        f"| {runs[0]['peak_bytes'] / 1e9:.2f} GB | 0 |",
    ]
    for _n in rings:
        _m = runs[_n]["map"]
        _err = float((_m - _ref).norm() / _ref.norm())
        _r = float(torch.corrcoef(torch.stack([_m.ravel(), _ref.ravel()]))[0, 1])
        errors[_n] = _err
        _paper = PAPER_ERR.get(F, {}).get(_n)
        _lines.append(
            f"| {_n} | {100 * _err:.3f} % | {_r:.5f} | "
            f"{float(_m.sum() / _ref.sum()):.4f} | {runs[_n]['seconds']:.1f} s | "
            f"{runs[_n]['peak_bytes'] / 1e9:.2f} GB | "
            + (f"{_paper:.3f} %" if _paper is not None else "-")
            + " |"
        )
    print("\n".join(_lines))
    mo.md("\n".join(_lines))
    return PAPER_ERR, errors


@app.cell
def _(F, NPIX, NSCAN, PAPER_ERR, WAVEFUNCTIONS, errors, np, plt, rings, runs, sim):
    import scatterem.vis as vis
    from scatterem.data import Sampling

    _ref = runs[0]["map"].numpy()
    _show = 4 if 4 in rings else rings[0]
    _part = runs[_show]["map"].numpy()
    _scan_sampling = Sampling(
        pixel_size=(sim.gridsize[0] / NSCAN, sim.gridsize[1] / NSCAN),
        units=("Å", "Å"),
    )
    fig, _axes = plt.subplots(1, 4, figsize=(17, 4.3))
    vis.show_2d(
        [_ref],
        figax=(fig, _axes[0]),
        role="intensity",
        cbar=True,
        sampling=_scan_sampling,
        title=[f"a  unpartitioned, f = {F}"],
    )
    vis.show_2d(
        [_part],
        figax=(fig, _axes[1]),
        role="intensity",
        cbar=True,
        sampling=_scan_sampling,
        title=[f"b  BiP-PRISM, n_radial = {_show}"],
    )
    _diff = (_part - _ref) / _ref.max()
    _lim = float(np.abs(_diff).max())
    vis.show_2d(
        [_diff],
        figax=(fig, _axes[2]),
        role="signed",
        cbar=True,
        sampling=_scan_sampling,
        norm=vis.NormalizationConfig(interval_type="manual", vmin=-_lim, vmax=_lim),
        title=["c  (b - a) / max(a)"],
    )
    _cbars = [_ax for _ax in fig.axes if _ax not in list(_axes)]
    for _cb in _cbars[:2]:
        _cb.set_ylabel("probability per electron per eV")
    if len(_cbars) > 2:
        _cbars[2].set_ylabel("difference / reference maximum")
    _ax = _axes[3]
    _ax.semilogy(
        rings,
        [100 * errors[_n] for _n in rings],
        "o-",
        color="#0072B2",
        label=f"this notebook ({NPIX}², {NSCAN}² scan, {WAVEFUNCTIONS})",
    )
    _paper = PAPER_ERR.get(F, {})
    if _paper:
        _ax.semilogy(
            list(_paper),
            list(_paper.values()),
            "s--",
            color="#D55E00",
            label="paper (936², 94² scan, muSTEM)",
        )
    _ax.set_xlabel("parent rings n_radial (both matrices)")
    _ax.set_ylabel("relative L2 error vs unpartitioned (%)")
    _ax.set_xticks([2, 4, 8, 12])
    _ax.set_title(f"d  partitioning error, f = {F}")
    _ax.grid(True, which="both", alpha=0.3)
    _ax.legend(fontsize=8)
    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Why the numbers differ from the paper

    The trend is the same as in the paper: the error falls by two orders of
    magnitude from 2 to 12 rings, and the partitioned calls need a fraction
    of the memory of the reference. With the default settings (936² grid,
    94 × 94 scan, *f* = 8, hydrogenic states), the errors are about 1.5
    times those of the paper: about 11.7, 2.4, 0.63 and 0.11 % for 2, 4, 8
    and 12 rings, against 7.88, 1.60, 0.45 and 0.08 % in the paper. The
    values are not the same, for these reasons:

    1. **Transition potentials.** The paper used 4 Fe L transition
       channels from muSTEM. `scatterem` 0.4 calculates its own channels
       with a rewritten first-Born model: here 27 channels (the monopole and
       both dipole terms). Without GPAW, the states are hydrogenic, which
       changes the shape of each potential. The partitioning error depends
       on how far the potentials reach, so it changes too.
    2. **Absolute scale.** Since version 0.4, a map value is a probability
       per incident electron per eV. The paper's maps had no absolute
       scale. A relative error does not depend on this scale.
    3. **Specimen setup.** Version 0.4 builds the projected potentials,
       the transmission functions and the probe with its own physics layer.
       The paper used the code of its time.
    4. **Grid and scan.** If you select the 480² grid or the 47 × 47 scan,
       the pixels are 0.2 Å and the scan steps are 2 Å. The aperture then
       holds the same beams, but the crop window has fewer pixels.
    5. **Hardware and precision.** The paper measured time and memory on an
       H100, with TF32 matrix products. This notebook uses full float32
       precision unless you select TF32. The 27 channels cost more time than the 4 channels of the
       paper. Do not compare the absolute times. The memory falls with the
       ring count, as in the paper (paper, *f* = 8: 15.8 GB unpartitioned,
       2.65 GB at 4 rings).

    The paper reports errors against the *unpartitioned map at the same f*.
    This notebook does the same. Neither is a total error against *f* = 1.
    """)
    return


@app.cell
def _(runs, torch):
    assert all(torch.isfinite(_r["map"]).all() for _r in runs.values())
    # Each call resets the peak counter, so the peak of the notebook is the
    # largest peak of the calls.
    _peak = max(
        [torch.cuda.max_memory_allocated()] + [_r["peak_bytes"] for _r in runs.values()]
    )
    print(f"peak GPU memory: {_peak / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
