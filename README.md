> ### This branch no longer reproduces the published `dk` exactly
>
> `master` is the code published with S. You et al., *Gap-Free Information
> Transfer in 4D-STEM via Fusion of Complementary Scattering Channels*, Advanced
> Science (2026) e76620 ([doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620)).
>
> The bright-field disk fit that sets the reciprocal-space sampling `dk` has been
> replaced with an independent implementation, so that this branch carries no
> third-party GPL-3.0 code and its Apache-2.0 licence is accurate. The two fits
> locate the disk edge slightly differently, and the difference grows with noise:
>
> | dataset | `dk` change | fused image vs. published |
> |---|---|---|
> | carbon (300 kV) | 0.15% | correlation 0.9999999 |
> | Co3O4 (200 kV) | 0.07% | correlation 0.9999932 |
> | **Gd2O3 (60 kV, low dose — Figure 1)** | **1.37%** | **correlation 0.925** |
>
> For carbon and Co3O4 the difference is invisible. **For Gd2O3 it is not**: the
> reconstruction that this code now produces is measurably different from the one
> in Figure 1, because a 1.37% error in `dk` propagates through the aberration fit
> into the phase image. Which of the two disk fits is closer to the true radius has
> not been established — that needs an independent calibration, not a comparison
> of the two candidates against each other.
>
> If you need the exact published reconstruction, use the tagged commit
> `master-as-published`, which is the code as it was when the paper appeared. If
> you need code you can redistribute under Apache-2.0, use this branch. If you are
> starting new work, use [`main`](../../tree/main).

# scatterem2

## Example:

- Example implemation of the datasets used in the paper

## Getting started:

- [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- `git clone` the repo and `cd` into the directory
- run `uv sync` to install all the dependencies in an editable environment
- run `uv sync --dev` to install all the dev dependencies in an editable environment

## Dependency management:

- use `uv add package_name` to add dependencies
- use `uv remove package_name` to remove dependencies
- use `uv add dev_package_name --dev` to add a dev dependency, i.e. that devs need (e.g. pytest) but you don't want shipped to users
- use `uv pip install testing_package_name` to install a package you think you might need, but don't want to add to dependencies just yet

## Running python/scripts in environment:

- use `uv run python`, `uv run jupyterlab` etc. to automatically activate the environment and run your command
- alternatively use `source .venv/bin/activate` to explicitly activate environment and use `python`, `jupyterlab` etc. as usual
  - note that if you're using an IDE like VS Code, it probably activates the environment automatically


## pre-commit installation
`pip install pre-commit`
`pre-commit install`
`pre-commit run --all-files`
