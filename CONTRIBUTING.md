# Contributing to scatterem

Thanks for your interest in `scatterem`.

This repository is public before the main code release so the project can establish good open-source habits early. That means contributions are welcome, but the scope of acceptable contributions is narrower than it will be after publication.

## Current contribution policy

Before the source release, the most useful contributions are:

- issue reports about repository structure, documentation clarity, and packaging metadata
- feature requests and research use cases
- questions about planned workflows, interoperability, and datasets
- suggestions on tutorials, benchmarks, and reproducibility expectations

Please do not open pull requests that add or request private source files from the unpublished implementation. Those changes cannot be reviewed in the open yet.

## Before opening an issue

Please check:

- [README.md](README.md) for the current project status
- [ROADMAP.md](ROADMAP.md) for planned release phases
- [SUPPORT.md](SUPPORT.md) for where different kinds of questions belong

When opening an issue, concrete context is especially helpful:

- the workflow you care about
- the data scale you expect
- relevant hardware constraints
- whether you need CPU support, CUDA, or multi-GPU execution
- any reference methods or papers you want to compare against

## Pull requests

During the pre-release phase, pull requests are most likely to be accepted when they improve:

- documentation and wording
- issue templates or contributor experience
- repository metadata and release scaffolding
- CI checks that validate the public scaffold

If you plan to spend meaningful time on a larger change, open an issue first so we can confirm it fits the current public scope.

## Development notes

The repository uses:

- `ruff` for linting
- `black` for formatting
- `pytest` for public scaffold checks
- `pre-commit` for local validation

Typical setup:

```bash
git clone https://github.com/<your-username>/scatterem.git
cd scatterem
uv sync --group dev
pre-commit install
```

Typical checks:

```bash
uv run pytest -v
uv run ruff check .
uv run black --check .
```

## Review expectations

We aim to keep review comments actionable and respectful. Response times may vary while the project is still in its pre-publication phase, but issues and pull requests that improve public readiness are welcome.

## License

By contributing to this repository, you agree that your contributions will be licensed under the Apache License 2.0.
