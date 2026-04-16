# Maintainer Guide

This document is a lightweight operating guide for maintainers of the public `scatterem` repository.

## Current role of the repository

Before the first public code release, this repository is meant to:

- communicate project status honestly
- collect user needs and feature requests
- prepare contributor workflows and governance
- reduce launch-day friction once publication constraints are lifted

It is not yet the full public source distribution.

## Maintainer priorities by phase

### Pre-publication

Prioritize:

- keeping public messaging accurate
- answering scope and roadmap questions clearly
- collecting signals about desired workflows, datasets, and hardware support
- improving docs, templates, and release readiness

Avoid:

- implying that unpublished code is already available
- promising timelines you are not confident about
- accepting PRs that depend on non-public implementation details

### Publication week

Prioritize:

- landing the real code, docs, and examples cleanly
- validating installation and examples from a fresh checkout
- publishing a release with notes and citation metadata updates
- triaging the first wave of user questions quickly and calmly

### First month after release

Prioritize:

- fixing installation and onboarding problems quickly
- clarifying unsupported workflows and limitations
- turning repeated user questions into docs improvements
- creating a visible cadence of small, reliable maintenance

## Recommended maintainer rhythm

During active periods, a simple rhythm works well:

- check new issues and pull requests at least a few times per week
- label issues quickly even if a full response must wait
- close unclear loops by linking to README, ROADMAP, SUPPORT, or SECURITY
- convert recurring confusion into concrete documentation updates

## Issue handling guidelines

Use labels to separate:

- bugs in the public scaffold
- feature requests
- questions
- release blockers
- documentation improvements

If GitHub Discussions is enabled later, move usage questions and open-ended exploration there while keeping GitHub Issues for actionable work.

## Communication principles

- be explicit about what is available now versus later
- prefer honest uncertainty over overconfident promises
- thank people for concrete use cases, especially before release
- treat early external questions as product research, not interruption

## Publication-day references

Use these companion documents:

- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)
- [PUBLICATION_DAY.md](PUBLICATION_DAY.md)
- [TRIAGE.md](TRIAGE.md)
- [SUPPORT.md](SUPPORT.md)
