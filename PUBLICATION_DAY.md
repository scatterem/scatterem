# Publication-Day Runbook

This runbook is intended to reduce stress on the day the embargo lifts.

## Before making the repository fully public

1. Confirm the paper or publication event is actually live.
2. Re-read the README and remove any remaining pre-release language that is no longer true.
3. Verify that no private code, credentials, paths, or datasets are present in the repo history you are about to publish.
4. Run the install and quickstart steps from a fresh checkout on at least one clean machine or environment.
5. Make sure a maintainer is available to watch incoming issues for the first 24 to 72 hours.

## Publication-day sequence

1. Merge the public source release branch.
2. Update `README.md`, `ROADMAP.md`, `CHANGELOG.md`, and `CITATION.cff` if any final publication details changed.
3. Trigger or enable the intended release workflow.
4. Create the GitHub release with concise notes:
   - what is included
   - what remains experimental
   - how to install
   - where to report issues
5. Verify the published package can actually be installed by a fresh user.
6. Pin or highlight one onboarding issue, discussion, or roadmap item if you want early community input.

## First 24 hours

Watch closely for:

- installation failures
- missing dependencies
- broken example paths
- unclear documentation
- confusion about supported hardware or workflows

Respond quickly to onboarding blockers even if the full fix takes longer. Early responsiveness matters more than perfection.

## First week

- turn repeated questions into README or docs improvements
- label issues consistently
- close or clarify stale confusion quickly
- avoid promising broad support until the first real user feedback is in

## Things not to do

- do not silently leave pre-release wording in place after code is public
- do not publish to PyPI without verifying the install path from scratch
- do not let the first wave of issues sit unlabeled for days if it can be avoided
- do not overcommit to timelines for complex features during launch week
