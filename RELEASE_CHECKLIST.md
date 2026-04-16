# Release Checklist

Use this checklist before cutting the first real public release.

## Repository contents

- [ ] Public source code is present and importable
- [ ] Private or embargoed material has been removed
- [ ] Public examples are included and runnable
- [ ] Test data is either included appropriately or fetched reproducibly
- [ ] Placeholder text in README, docs, and examples has been replaced

## Documentation

- [ ] README reflects the released state rather than the pre-release scaffold
- [ ] Installation instructions were tested from a clean environment
- [ ] Quickstart example runs as written
- [ ] Known limitations are documented
- [ ] Supported Python, PyTorch, CUDA, and Warp versions are listed
- [ ] Citation metadata matches the publication and release

## Packaging and CI

- [ ] `pyproject.toml` metadata is up to date
- [ ] Wheels and source distribution build successfully
- [ ] CI passes on the intended Python versions
- [ ] Tests cover at least one user-facing workflow
- [ ] Publish workflow is pointed at the intended release trigger

## Community readiness

- [ ] Issue templates still match the released project scope
- [ ] `CONTRIBUTING.md` reflects the post-publication contribution model
- [ ] `SUPPORT.md` matches how you want users to ask for help
- [ ] `ROADMAP.md` reflects the next realistic milestones
- [ ] Maintainers know who will monitor issues during launch week

## Release artifacts

- [ ] Release notes summarize what is included and what is not
- [ ] Version number is consistent across package metadata and changelog
- [ ] `CHANGELOG.md` has a release entry
- [ ] Citation text and links are correct
- [ ] Optional announcement copy is prepared for paper, lab site, or social posts
