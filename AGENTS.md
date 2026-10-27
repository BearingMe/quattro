# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project Overview

Research project reproducing native INT8 neural-network inference on AMD RDNA 3 (gfx1101, RX 7700 XT, Windows 11), using AMD FSR 4 Super Resolution as the case study. DirectX 12 + HLSL first; native INT8 WMMA with INT32 accumulation is the research objective.

## Authoritative Documents

- [`README.md`](README.md) — mission and working rules.
- [`docs/MILESTONES.md`](docs/MILESTONES.md) — roadmap M01–M15, checkpoints, validation gates, execution loop.
- [`docs/architecture.md`](docs/architecture.md) — module boundaries, dependency rules, directory and artifact policy. **Binding.**
- [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) — evidence tagging: `[D]` documented · `[S]` source · `[R]` reverse-engineering · `[M]` measurement · `[H]` hypothesis · `[U]` unknown.
- [`.references/README.md`](.references/README.md) — pinned research sources; never modify anything under `.references/sources/`.

## Tech Stack

- C++20 (host + inference), HLSL (shaders), DirectX 12, CMake + CMake Presets.
- Python only for offline analysis scripts (`scripts/`), never as a runtime.
- Never add Rust, CUDA, ZLUDA, Vulkan, NVIDIA APIs, or ML frameworks.

## Commands

```powershell
cmake --preset debug                  # configure (VS 2022 x64)
cmake --preset release                # configure release
cmake --build --preset debug          # build
ctest --preset debug                  # run smoke tests
build/bin/rdna3-int8.exe list         # experiment CLI
build/bin/rdna3-int8.exe run <name>   # exit 0 pass · 1 experiment failure · 2 usage error
```

Experiments register explicitly in `src/main.cpp`; do not create additional executables (see `docs/architecture.md` §6).

Toolchain: VS 2022 Community (MSVC 14.44 — not on PATH; use vcvars or the CMake VS generator), Windows SDK 10.0.26100 (`dxc.exe` at `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64`), CMake/Ninja/Python 3.13/Git on PATH.

## Architecture (summary — see docs/architecture.md)

- Modular monolith: `core` → `gpu` → `inference`, plus `reference` (CPU-only) and `experiments` (orchestrates all). Dependencies flow one direction; nothing depends on `experiments`.
- `core` never touches GPU APIs; `gpu` never learns FSR 4 model semantics; model-specific code stays in `inference`.
- One experiment CLI (`rdna3-int8.exe`) with explicit registration — no executable-per-experiment, no plugin systems.
- ALL generated files go under `build/` — never the repo root.
- Directories are created only when real code needs them; no placeholder files or empty modules.

## Workflow

1. Follow the milestone loop in `docs/MILESTONES.md` §7: research → smallest experiment → implement → validate → document → report gate → **stop**. Do not auto-advance milestones; the next milestone requires explicit authorization.
2. Record results per `docs/CONVENTIONS.md`: experiments in `docs/experiments/EXP-NNN-slug.md`, durable knowledge in `docs/findings/`, engineering decisions in `.references/findings.md` where they stem from research.
3. A milestone is complete only when its validation gate PASSES on the gfx1101 hardware, with evidence recorded. Compilation alone never passes a gate.

## Memory and Context Discipline

Each category of knowledge has one authoritative home — cross-reference it, never duplicate it:

| Information | Owner |
|---|---|
| Permanent agent & engineering rules | `AGENTS.md`, `docs/architecture.md` |
| Milestone progress & gate evidence | `docs/MILESTONES.md` |
| Research-derived engineering decisions | `.references/findings.md` |
| Current tasks & next actions | OpenCode todos (ephemeral — rebuild from the active milestone if lost) |
| Reusable facts & pointers | AgentMemory (concise; supplements docs, never replaces them) |
| Detailed technical findings | `docs/findings/` |
| Experimental records & measurements | `docs/experiments/` |
| External references | `.references/` |
| Implementation & history | Source code, Git — never backup copies or memory entries |
| Generated artifacts | `build/` |

Rules:

1. Store knowledge, not conversations. Persist only what helps a future session: verified facts, decisions, constraints, reproducible failures. Tag evidence per `docs/CONVENTIONS.md` — a `[H]` repeated across sessions is still `[H]`; never invent versions, measurements, or experiment IDs.
2. Before persisting, ask: does this affect a future decision, is it reliable, and is it already recorded somewhere authoritative? Prefer one precise fact plus a pointer to the evidence file over copied excerpts. Scope findings to their hardware/FSR version — a gfx1101 measurement is not all RDNA 3; FSR 4.0.2 is not FSR 4.1.
3. Keep memory correct: when evidence changes, update the authoritative document first, then the memory entry; mark superseded findings and retire what is wrong, duplicated, or obsolete. Failed experiments are durable negative evidence — record, never delete. Claim memory writes only when they actually succeeded.
4. Session start: read `AGENTS.md`, identify the active milestone in `docs/MILESTONES.md`, read only that milestone's gate and the relevant source files, check `git status`, and retrieve only targeted AgentMemory facts. Do not read every document, dump the memory database, or reconstruct project history.
5. Session end: update milestone status and gate evidence, record findings, correct stale memory entries, close finished todos, and report incomplete work. No session-summary files; no memory writes just to demonstrate memory management.
6. Treat context as a budget: search precisely, read specific files, reference sources rather than pasting them. When context is tight, prioritize: active objective > validation gate > relevant code > verified constraints > blockers > next steps.
7. Never store credentials, tokens, keys, or sensitive environment data in memory, docs, or experiment reports.

## Never

- Never implement future milestones "while you're in there"; no speculative abstractions or placeholder modules.
- Never modify, rebuild, or commit inside `.references/sources/` (upstream code is pinned, read-only).
- Never write generated artifacts (binaries, disassembly, captures, dumps) anywhere outside `build/`.
- Never create new top-level directories, build systems, or build targets without documented justification.
- Never mark a gate PASS from plausible output, missing errors, or theoretical performance — PASS/FAIL/BLOCKED/UNVERIFIED must be evidence-backed.
- Never present hypotheses as facts; tag claims per CONVENTIONS.md.
- Never delete untracked files you did not create; never run destructive cleanup without explicit confirmation.
- Never commit secrets or local environment files.
- Never disable or skip validation checks to make a gate pass.

## Definition of Done

- Requested milestone/checkpoint implemented with its gate PASSED on hardware (or explicitly BLOCKED/UNVERIFIED with reasons).
- Results documented per CONVENTIONS.md; `docs/MILESTONES.md` status updated.
- `git status` clean or intentionally explained; conventional-commits message; no unrelated changes.
