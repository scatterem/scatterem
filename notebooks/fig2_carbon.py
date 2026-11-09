# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.4.0",
#     "marimo==0.25.2",
#     "matplotlib>=3.10",
#     "torch>=2.5.0",
# ]
# ///
"""Paper reproduction: Figure 2: amorphous carbon (runs experiments/ff_stem/Fig2_carbon.py)."""

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
    # Figure 2: amorphous carbon

    This notebook runs the script `experiments/ff_stem/Fig2_carbon.py` of the
    `scatterem` repository. The script reconstructs amorphous carbon (300 kV, 19.68 mrad, 0.25 Å scan step) with
    direct (SSB) ptychography, tilt-corrected dark field (tcDF) and their
    fusion (FFF). Then the notebook shows the images that the script saves.

    The notebook does not copy the physics. It downloads the script of the
    release tag `v0.4.0` from GitHub and runs it unchanged. The output
    differs from the published figure; `docs/figures.md` in the repository
    gives the differences and their causes.

    **Paper:** S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao,
    M. Wu, D. Stroppa, D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci,
    D. Krishnan, X. Ye, F. K. M. Schur, E. Spiecker, P. Pelz, *"Gap-Free
    Information Transfer in 4D-STEM via Fusion of Complementary Scattering
    Channels"*, **Advanced Science** 13, e76620 (2026),
    doi:[10.1002/advs.76620](https://doi.org/10.1002/advs.76620).

    **Data:** `You2026Carbon` from Zenodo record
    [18008901](https://zenodo.org/records/18008901)
    (DOI 10.5281/zenodo.18008901), licence **CC-BY-4.0**. If you use the
    data, cite the record and the paper. The run needs about **0.8 GiB** peak GPU memory.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a
       GPU (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells.

    The dataset downloads on first use, and its md5 checksums are verified.
    Set `FFSTEM_DATA_ROOT` to keep the data in a different directory (the
    default is `data`).
    """)
    return


@app.cell
def _():
    import contextlib
    import io
    import os
    import runpy
    import sys
    import tempfile
    import urllib.request
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On molab, select a GPU in the notebook specs "
            "and restart the notebook."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return Path, contextlib, io, os, runpy, sys, tempfile, torch, urllib


@app.cell
def _(Path, os, tempfile, urllib):
    SCRIPT = "Fig2_carbon.py"
    TAG = "v0.4.0"
    URL = (
        "https://raw.githubusercontent.com/scatterem/scatterem/"
        f"{TAG}/experiments/ff_stem/{SCRIPT}"
    )

    # The script writes its images to outputs/ next to itself, so it runs
    # from a new temporary directory. SCATTEREM_SOURCE selects a local
    # source tree instead of GitHub (for tests).
    WORK_DIR = Path(tempfile.mkdtemp(prefix="scatterem-fig2_carbon-"))
    script_path = WORK_DIR / SCRIPT
    _source = os.environ.get("SCATTEREM_SOURCE")
    if _source:
        _local = Path(_source) / "experiments" / "ff_stem" / SCRIPT
        script_path.write_text(_local.read_text())
        print(f"script: {_local}")
    else:
        with urllib.request.urlopen(URL, timeout=60) as _response:
            script_path.write_bytes(_response.read())
        print(f"script: {URL}")
    return (script_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Run the script

    The script prints its progress below. It ends with an `OK:` line that
    gives the image shapes and the calibrated `dk`.
    """)
    return


@app.cell
def _(Path, contextlib, io, os, runpy, script_path, sys, torch):
    class _Tee(io.TextIOBase):
        """Write to the live output and keep a copy."""

        def __init__(self, stream):
            self.stream = stream
            self.copy = io.StringIO()

        def write(self, text):
            self.stream.write(text)
            self.copy.write(text)
            return len(text)

        def flush(self):
            self.stream.flush()

    os.environ["FFSTEM_DATA_ROOT"] = str(
        Path(os.environ.get("FFSTEM_DATA_ROOT", "data")).resolve()
    )
    os.environ["FFSTEM_HEADLESS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    torch.cuda.reset_peak_memory_stats()
    _tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(_tee):
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        except SystemExit as _exit:
            # The script exits with code 0 when it cannot get the data.
            # The check of the OK line below finds that case.
            if _exit.code not in (None, 0):
                raise
    _ok = [_line for _line in _tee.copy.getvalue().splitlines() if " OK: " in _line]
    if not _ok:
        raise RuntimeError(
            "The script did not print an OK line. Read its output above."
        )
    OK_LINE = _ok[-1]
    print(OK_LINE)
    return (OK_LINE,)


@app.cell
def _(OK_LINE, mo, script_path):
    _pngs = sorted((script_path.parent / "outputs").glob("*.png"))
    mo.vstack(
        [mo.md(f"`{OK_LINE}`")]
        + [
            mo.vstack([mo.md(f"**{_png.name}**"), mo.image(src=_png, width=600)])
            for _png in _pngs
        ]
    )
    return


@app.cell
def _(OK_LINE, torch):
    assert OK_LINE
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
