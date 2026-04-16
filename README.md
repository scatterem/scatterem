# scatterem

[![CI](https://github.com/ECLIPSE-Lab/scatterem/actions/workflows/test.yml/badge.svg)](https://github.com/ECLIPSE-Lab/scatterem/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

`scatterem` is a forthcoming open-source Python toolkit for scattering simulations and electron microscopy reconstruction, built on PyTorch.

## Project status

This repository is intentionally in a pre-release state while the associated research publication is in progress.

- The public repository currently contains project scaffolding, contribution guidelines, issue templates, and release planning material.
- The core implementation, tutorials, and benchmarks will be opened once publication constraints are lifted.
- Until then, this repository acts as the public home for project planning, community feedback, and open-source infrastructure.

The goal is to make the eventual code release feel mature on day one rather than opening a repository only after the paper appears.

## Planned scope

The first public release is expected to include:

- Ptychographic reconstruction workflows for single-slice and multi-slice settings
- Wave propagation and scattering models
- PyTorch-based neural network components for learned reconstruction
- Reproducible examples and benchmark configurations
- Installation instructions, API documentation, and tutorials

## What is useful today

Even before the code is public, this repository is the right place to:

- Follow project progress
- Open feature requests and share use cases
- Report documentation or packaging issues in the public scaffold
- Discuss interoperability, datasets, and reproducibility expectations
- Review the roadmap and release criteria

## Release readiness principles

The public release is being prepared around a few priorities:

- Reproducibility first: examples, environment details, and benchmark settings should be explicit
- Honest scope: supported workflows and known limitations will be documented clearly
- Contributor friendliness: issue templates, contribution policy, and review expectations are defined in advance
- Research credibility: citation guidance, changelog discipline, and method references are part of the initial release

## Repository layout

```text
scatterem/
├── .github/                 # CI, issue templates, pull request template
├── docs/                    # Documentation skeleton and release planning notes
├── example/                 # Placeholder for future public examples
├── scatterem/               # Placeholder package for future public source release
├── tests/                   # Public test scaffolding
├── CHANGELOG.md             # Human-readable release history
├── CODE_OF_CONDUCT.md       # Community expectations
├── CONTRIBUTING.md          # Contribution policy for the pre-release phase
├── ROADMAP.md               # Release phases and milestones
├── SECURITY.md              # Security disclosure guidance
└── SUPPORT.md               # Where to ask for help and what feedback is useful
```

## Contributing before release

Code contributions are intentionally limited until the publication is out, but contributions are still welcome in a few areas:

- roadmap feedback
- issue reports on packaging and repository infrastructure
- documentation suggestions
- user stories, requirements, and integration constraints

See [CONTRIBUTING.md](CONTRIBUTING.md) for the current policy.

## Maintainer and launch planning

The repository already includes launch-oriented operational docs so the eventual code release is less chaotic:

- [MAINTAINERS.md](MAINTAINERS.md)
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)
- [PUBLICATION_DAY.md](PUBLICATION_DAY.md)
- [TRIAGE.md](TRIAGE.md)
- [DISCUSSIONS.md](DISCUSSIONS.md)

## Citation

If you are tracking this project for future research use, please see [CITATION.cff](CITATION.cff). Citation metadata will be updated alongside the first public code release.

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
