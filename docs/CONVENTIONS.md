# Research documentation conventions

## Evidence classification

Every non-trivial claim in this repository must be tagged with its evidence class:

| Tag | Class | Meaning |
|------|-------|---------|
| `[D]` | Documented fact | Official AMD / Microsoft documentation, cited. Highest authority. |
| `[S]` | Source observation | Read directly from public source code (file / commit referenced). |
| `[R]` | Reverse-engineering finding | Derived from binaries, disassembly, or runtime analysis; may be incomplete. |
| `[M]` | Measurement | Produced by an experiment in this repo; reproducible per the referenced experiment doc. |
| `[H]` | Hypothesis | Plausible claim not yet confirmed. Must never be relied on as fact. |
| `[U]` | Unknown / open question | Explicitly unresolved. |

Precedence when classes conflict: `[D]` > `[S]` > `[R]` > `[M]` > `[H]`. Conflicts are recorded, never silently resolved.

## Citation format

```text
[D] RX 7700 XT is gfx1101 (RDNA 3, 54 CUs, 12 GiB).
    Source: AMD ROCm — Accelerator and GPU hardware specifications,
    https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html (accessed 2026-10-27)
```

- One claim per bullet, tag first.
- Access date required. Include version numbers for drivers, SDKs, and tools where relevant.

## Files

- `docs/findings/YYYY-MM-DD-<topic>.md` — durable findings on a topic (hardware, ISA, driver behavior, FSR 4 internals).
- `docs/experiments/EXP-<NNN>-<slug>.md` — one experiment per file, numbered monotonically.

### Finding template

```markdown
# <Topic>

## Summary
## Findings        (tagged, cited bullets)
## Open questions  (tagged [U])
## Sources
```

### Experiment template

```markdown
# EXP-NNN: <Question>

Date: YYYY-MM-DD
Milestone: M?

## Question
## Method
## Result          (tag [M]; include raw numbers / outputs)
## Interpretation  (tag [D]/[S]/[R]/[H] as appropriate)
## Next
```

## Rules

1. Failed experiments are recorded, not deleted.
2. Never rewrite the Result section of a recorded experiment; append an addendum.
3. A milestone may only be marked complete in `MILESTONES.md` after its validation gate passed; link the experiment or result.
4. Speculation stays tagged `[H]` or gets tested.
