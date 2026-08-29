# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs) for Open Executive and
for adaptations built on top of it.

An ADR captures **one** decision: the context that forced it, the options that
were on the table, what was chosen, and what the choice costs. It is a record of
a moment, not living documentation. Living documentation belongs in
`docs/architecture.md` and in
`packages/core/openexecutive/architecture/prebuilt/`.

## Convention

**Filename** — `NNNN-kebab-case-title.md`, where `NNNN` is a zero-padded
four-digit sequence number assigned in order of acceptance. Numbers are never
reused.

**Status** — one of:

| Status | Meaning |
| --- | --- |
| `Proposed` | Under discussion; not yet binding. |
| `Accepted` | Binding as of the recorded date. |
| `Superseded by NNNN` | Replaced by a later ADR. |
| `Deprecated` | No longer binding, with no direct replacement. |

**Immutability** — once an ADR is `Accepted`, its Context, Decision, and
Consequences sections are not rewritten. Corrections of fact may be appended in
an explicitly marked amendment block. A change of position requires a **new**
ADR that supersedes the old one; the old one's status line is then updated to
point at it. This is the only permitted edit to an accepted record.

**Scope discipline** — an ADR states what it decides *and* what it does not
decide. A decision about one component is not a decision about the system.
Where an ADR is scoped to an adaptation or a downstream product rather than to
upstream Open Executive, it must say so in its own Scope section.

**Evidence** — an ADR that rests on measurement must record enough identity to
be reproducible: exact artifact hashes, sizes, versions, runtime builds,
hardware, sampling configuration, and date. It must record evidence that cuts
against the decision as well as evidence that supports it. An ADR that reports
only the winner's strengths is not a decision record.

## Template

```markdown
# NNNN. <Title>

- **Status:** Proposed | Accepted | Superseded by NNNN | Deprecated
- **Date:** YYYY-MM-DD
- **Deciders:** <who>
- **Scope:** <what this decision governs — and what it does not>

## Context
## Decision
## What This Decision Does Not Establish
## Evidence
## Consequences
## Alternatives Considered
## Limitations of the Evidence
## Revisit Criteria
```

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-local-executive-model-qwen3.5.md) | Local Executive / Orchestrator model for the Moeen adaptation | Accepted |
