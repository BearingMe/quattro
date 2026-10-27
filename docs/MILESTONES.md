# Milestones

Status: `☐` proposed · `◐` in progress · `☑` gate passed · `✖` closed with negative findings

Rules:

- A **milestone** is a meaningful, independently verifiable capability. A **checkpoint** is a concrete intermediate result recorded in `docs/`.
- Every milestone has an executable validation gate performed on the target hardware. Gates are never waved through.
- Milestone details are filled in when the milestone starts — no speculative design (KISS / YAGNI).
- The first engineering milestone requires explicit authorization before implementation begins.

---

## M0 — Bootstrap & environment baseline — ☑ (2026-10-27)

Establish repository structure, documentation conventions, and a verified hardware/toolchain baseline.

**Gate:** Environment verified on hardware and recorded in [`docs/findings/2026-10-27-environment-baseline.md`](findings/2026-10-27-environment-baseline.md); project structure exists.

## M1 — Toolchain smoke test (hello compute) — ☐ proposed

Smallest possible D3D12 compute pipeline on the RX 7700 XT: create a device explicitly selecting the AMD adapter, compile a trivial HLSL kernel with DXC, dispatch, read back, validate against expected bytes.

**Gate:** Executable test reads back expected values from a GPU dispatch on gfx1101.

## M2 — INT8 arithmetic baseline — ☐ proposed

Validate native INT8 integer math semantics in HLSL (packed dot-product / DP4A path: `dot4add_i8packed`) against a CPU reference over randomized inputs.

**Gate:** Bit-exact match vs CPU reference across multiple random seeds; experiment recorded.

## M3 — WMMA / cooperative-matrix investigation — ☐ proposed

Determine the real path to native INT8 WMMA from D3D12/HLSL on gfx1101: Cooperative Vector (SM 6.9) and DirectX Linear Algebra support in the current driver, required DXC / Agility SDK versions; decide whether a HIP comparison is needed (explicit decision point).

**Gate:** Either a validated INT8 matrix-multiply with INT32 accumulation vs CPU reference on gfx1101, or a documented negative result with evidence.

## M4 — Performance characterization — ☐ proposed

Measure M2/M3 kernel variants with GPU timestamp queries; establish a reproducible measurement methodology.

**Gate:** Recorded, reproducible measurements (repeated runs, variance reported).

## M5 — Convolution primitives — ☐ proposed

INT8 convolution primitives (implicit GEMM unless measurements justify otherwise) shaped like the tensor operations FSR 4 uses.

**Gate:** Correctness vs reference plus timing recorded.

## M6 — FSR 4 RDNA 3 execution path study — ☐ proposed

Document how the shipped FSR 4.1 INT8 implementation actually executes on RDNA 3 (SDK, driver, GPU captures, ISA evidence).

**Gate:** Findings document with `[D]`/`[S]`/`[R]` evidence for the execution path.

## M7+ — End-to-end mini-pipeline — ☐ undefined

Single-image INT8 inference pipeline informed by M6. Scope is deliberately defined only after M6.
