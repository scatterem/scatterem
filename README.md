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
