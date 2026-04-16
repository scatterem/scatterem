# Issue Triage Guide

This guide helps keep incoming public feedback manageable.

## Fast first pass

For each new issue:

1. decide whether it is a bug, feature request, question, documentation issue, or security concern
2. add at least one label
3. confirm whether it belongs in Issues or would fit better in Discussions later
4. identify whether it blocks adoption, release readiness, or long-term roadmap planning

## Suggested label set

Consider creating labels such as:

- `bug`
- `documentation`
- `enhancement`
- `question`
- `good first issue`
- `release-blocker`
- `research-workflow`
- `installation`
- `reproducibility`

## Priority guidance

Highest priority:

- install failures
- broken quickstart or example steps
- misleading documentation
- packaging or release workflow failures

Medium priority:

- feature requests tied to realistic user workflows
- documentation gaps that create friction but not hard blockers

Lower priority:

- broad exploratory ideas without concrete use cases
- requests that depend on unpublished or unsupported internal behavior

## Closing loops well

When closing or deferring an issue:

- explain why briefly
- link to the relevant doc when possible
- say whether the issue is out of scope, postponed, duplicated, or blocked on publication

## Pre-release special case

Before the code release, many issues will really be signals about user demand. Even if no code action is possible yet, it is useful to capture:

- the workflow
- the scale
- the hardware assumptions
- the expected outputs
- comparison tools or baseline methods
