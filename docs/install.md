# Install

```bash
pip install scatterem
```

Python 3.10 or newer. Twelve runtime dependencies, all from PyPI.

## About the GPU

A CUDA GPU is not optional in practice. The reconstructions are FFT- and
kernel-bound, and the NVIDIA Warp kernels that do the direct-ptychography work
need one for anything past a small test. Everything runs on CPU, just slowly
enough that you would not want to.

If the default `torch` wheel does not match your driver, install `torch` first
from the index that does, then install this package — pip will leave your
existing `torch` alone:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install scatterem
```

`import scatterem` initialises Warp, which compiles a few kernels the first time
and prints a short banner. That is expected, and it is cached afterwards.

:::{admonition} Matmul precision
:class: note

`import scatterem` does not change torch's float32 matmul precision, so it stays
at torch's default, `"highest"` (true fp32). The published figures were
computed at `"medium"` (TF32 on Ampere and later GPUs); to run exactly as they
did, call `torch.set_float32_matmul_precision("medium")` after importing.
:::

## How much GPU memory

Peak memory is set by the *output* grid, so it grows with scan size and
upsampling rather than with detector size. Measured with
`torch.cuda.max_memory_allocated`:

| Reconstruction | Scan | Output | Peak |
|---|---|---|---|
| Amorphous carbon | 96² | 96² | 0.8 GiB |
| Co₃O₄ | — | 185² | 1.2 GiB |
| Au, low dose, reduced scan | 128² | 512² | 1.4 GiB |
| Au, low dose, paper configuration | 256² | 1024² | 5.5 GiB |
| Gd₂O₃ | 512² | 1024² | 21.7 GiB |

These are the figure scripts of `experiments/ff_stem/` with v0.4 on an NVIDIA
RTX A6000, over the whole script.

:::{admonition} Raising `n_batches` does not reduce memory much
:class: warning

Measured on the Au reconstruction: `n_batches` of 15, 60 and 150 gave 5.7, 5.4
and 5.3 GiB.

**Shrinking the scan is the effective lever.** Pass `scan_edge_crop` to the
named datasets, or crop your own array before constructing.
:::

(molab)=
## molab

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py)

molab is the free hosted notebook service of marimo. The
[FF-STEM demo notebook](https://github.com/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py)
runs the Au low-dose reconstruction end to end, with the paper's configuration
(1024² output, 5.5 GiB peak). On a GPU with less than 8 GiB it uses a reduced
scan (512² output, 1.4 GiB). Switch on the GPU in the notebook specs; nothing
else is needed. The code and the data are both public, the notebook installs
`scatterem` itself, and the dataset downloads itself, md5-verified.

The folder `notebooks/` also holds an interactive notebook for each FF-STEM
figure and for the BiP-PRISM figures; [Notebooks](notebooks.md) lists them.

<span id="from-source"></span>

## Development version from GitHub

The PyPI package is the latest release. To install the current state of the
`main` branch instead:

```bash
pip install git+https://github.com/scatterem/scatterem
```

To work on the code, clone the repository and install it in editable mode with
the test dependencies:

```bash
git clone https://github.com/scatterem/scatterem
cd scatterem
pip install -e ".[dev]"
pytest
```

## For STEM-EELS simulation

The simulation code needs nothing beyond the above to run. To be
*quantitative*, it needs `gpaw`: GPAW supplies the all-electron bound and
continuum radial wavefunctions that the ionization transition potentials are
built from. The `eels` extra installs `gpaw` and `ase`; `ase` also reads `.cif`
and extended `.xyz` structure files.

```bash
CC=g++ pip install "scatterem[eels]" gpaw-data
```

PyPI has no GPAW wheels, so pip builds GPAW from its source package. GPAW
26.7.0 has C++ sources. The build uses `mpic++` when it is on the `PATH`, and
otherwise the C compiler of your Python, which cannot compile them. So the
build needs a C++ compiler (`g++`, set with `CC=g++` as above, or an MPI
toolchain that provides `mpic++`) and the Python development headers
(`python3-dev` on Debian and Ubuntu). Without `CC=g++` and without `mpic++`, the
build stops with `fatal error: algorithm: No such file or directory`. With
`CC=g++`, the build did not need libxc, MPI or a Fortran compiler on Ubuntu
24.04 (GPAW 26.7.0, Python 3.12). If you cannot compile, install GPAW from
conda-forge, which has prebuilt packages. The default, `wavefunctions="auto"`, uses GPAW when it is
installed, as v0.3 did. Without GPAW, the code falls back to hydrogenic orbitals
and a free continuum, and emits a `HydrogenicFallbackWarning`. These states run
the whole pipeline but do not give accurate cross-sections. In both cases v0.4
gives a new absolute scale (first-Born, Å²/eV). After a best scalar fit, the map
shapes differ from v0.3 by 18-61 % with the hydrogenic states. With GPAW, they
differ by 0.03-0.6 % for the K, L23 and M45 edges we tested, and by 1.4 % for
Fe L1. See
[The BiP-PRISM paper and v0.4](#bip-prism-paper-v04).

The code caches atomic wavefunctions and transition potentials on disk. Set
`SCATTEREM_EELS_CACHE` to put the cache somewhere with room.

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `eels` | `gpaw`, `ase` | quantitative transition potentials; `.cif` and extended `.xyz` files |
| `triton` | `triton` | the optional Triton kernels; without it the code uses torch |
| `notebook` | `ipython`, `marimo` | inline display of `scatterem.vis.show` figures |
| `dev` | `pytest` | running the test suite |
