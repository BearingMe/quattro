# Native RDNA 3 INT8 Exploration — Milestones and Validation Gates

| Field | Value |
|---|---|
| Target | AMD Radeon RX 7700 XT (gfx1101) |
| Platform | Windows 11 |
| Primary case study | AMD FSR 4.1 INT8 Super Resolution |
| Initial graphics API | DirectX 12 |
| Initial shader language | HLSL |
| Status | Roadmap — M01–M09 complete (see per-milestone status) |

Status markers: `☐` proposed · `◐` in progress · `☑` gate passed · `✖` closed with negative findings · `⊘` blocked

Evidence recorded against gates follows the tagging rules in [`CONVENTIONS.md`](CONVENTIONS.md).

---

## 1. Project Objective

Understand and progressively reproduce the techniques AMD uses to execute machine-learning Super Resolution on RDNA 3 hardware.

The investigation focuses on:

- INT8 quantization.
- Native RDNA 3 matrix acceleration.
- GPU tensor layouts and memory movement.
- Quantized neural network execution.
- FSR 4's neural inference architecture.
- Temporal Super Resolution.
- Performance characteristics of INT8 versus FP16.

The project starts from scratch. Public documentation and source code are the primary references; reverse engineering is secondary.

The final objective is to execute a standalone FSR 4-inspired — or, where source availability permits, FSR 4-equivalent — INT8 inference pipeline on the RX 7700 XT. Visual correctness and performance optimization are progressively introduced after successful execution.

### Scope

Included:

- RDNA 3 / gfx1101.
- Windows 11.
- DirectX 12.
- HLSL and DXC.
- INT8 and FP16.
- Neural network inference.
- FSR 4 Super Resolution.

Excluded:

- DLSS.
- Ray Reconstruction.
- NGX.
- Game injection.
- Driver modification.
- Multi-GPU support.
- Cross-platform compatibility.
- Production-ready inference frameworks.

---

## 2. Execution Rules

### Milestones

Every milestone must specify:

- Objective.
- Scope.
- Checkpoints.
- Validation gate.
- Deliverables.

A milestone is complete only when its validation gate passes.

### Checkpoints

Checkpoints represent concrete, independently verifiable results. Examples:

- A shader compiles.
- A GPU dispatch executes.
- A matrix multiplication matches its CPU reference.
- A numerical error is measured.
- A hardware instruction is confirmed.
- A reconstructed frame is produced.

### Validation

Validation gates are mandatory. A gate cannot pass based exclusively on:

- Successful compilation.
- Plausible-looking output.
- Missing error messages.
- Assumed GPU behavior.
- Theoretical performance.
- Undocumented implementation assumptions.

Differentiate:

- **PASS** — all mandatory criteria verified.
- **FAIL** — one or more mandatory criteria failed.
- **BLOCKED** — an external prerequisite is unavailable.
- **UNVERIFIED** — execution or evidence is insufficient.

Record the evidence supporting each result.

### Experimental Failure

Experimental errors are acceptable and expected. Poor quality, quantization artifacts, numerical differences, and slow execution are useful observations.

However, a gate must not silently accept:

- Missing operations.
- Invalid memory accesses.
- Unexplained integer overflow.
- Incorrect tensor layouts.
- Uncontrolled NaN or Inf propagation.
- Incomplete inference presented as complete.

A deliberate experiment may test these behaviors, but its validation must explicitly distinguish the expected failure from implementation defects.

### Scope Control

Complete one milestone at a time. Do not implement future milestones unless a dependency requires a small, explicitly documented investigation. Stop after passing each milestone and report the results.

---

## Phase I — Understanding AMD

> Goal: establish a reliable technical baseline before implementing neural inference.

### M01 — Environment and Bootstrap

**Status:** ☑ complete (2026-10-27 — C01/C02/C06 by the environment baseline, C03–C05 by [`experiments/EXP-001-bootstrap.md`](experiments/EXP-001-bootstrap.md))

#### Objective

Establish a minimal, reproducible development environment.

#### Scope

- C++20.
- CMake.
- Windows SDK.
- DirectX 12.
- DirectX Shader Compiler.
- Target GPU identification.
- Basic testing infrastructure.

#### Checkpoints

- [x] C01: Verify GPU identity and driver.
- [x] C02: Verify available compiler and SDK versions.
- [x] C03: Configure the minimal CMake project.
- [x] C04: Compile a standalone executable.
- [x] C05: Execute the compiled program.
- [x] C06: Record the development environment.

#### Validation Gate

- [x] CMake configuration succeeds.
- [x] C++ compilation succeeds.
- [x] Executable runs successfully.
- [x] GPU and driver are identified.
- [x] Toolchain versions are recorded.
- [x] Build instructions are reproducible.

#### Deliverables

- Minimal project.
- Working build configuration.
- Environment documentation.

---

### M02 — Reference Collection

**Status:** ☑ complete (2026-10-27 — see [`.references/`](../.references/README.md), commit `c3af362`; re-verified 2026-10-27: all four submodule pins intact, legacy M0–M7 milestone references remapped to M01–M15)

#### Objective

Build the project's technical research collection.

#### Scope

All references live under `.references/`. External repositories are pinned Git submodules under `.references/sources/`.

Prioritize:

1. Official AMD documentation.
2. Official AMD source code.
3. Public FSR 4 references.
4. Independent technical research.
5. Reverse engineering when necessary.

#### Checkpoints

- [x] C01: Identify authoritative RDNA 3 references.
- [x] C02: Identify official FSR 4 integration references.
- [x] C03: Identify usable source-code implementations.
- [x] C04: Review repository provenance and licensing.
- [x] C05: Add relevant repositories as submodules.
- [x] C06: Document research gaps.

#### Validation Gate

- [x] Selected repositories are accessible.
- [x] Submodules are pinned to exact commits.
- [x] Each repository has a documented purpose.
- [x] Important claims have source references.
- [x] Source availability and licensing are documented.
- [x] Unknowns are explicitly identified.

#### Deliverables

- `.references/` research workspace.
- Source index.
- Initial research notes.

#### Boundary

Do not begin implementation.

---

### M03 — Official FSR 4 Baseline

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-002-fsr4-baseline.md`](experiments/EXP-002-fsr4-baseline.md))

#### Objective

Establish how the official FSR 4 INT8 implementation behaves on the RX 7700 XT. This baseline becomes the primary behavioral oracle for later experiments.

#### Scope

- Official FSR 4.1.1 implementation.
- RDNA 3 INT8 inference.
- Supported integration through AMD's FSR API.
- Native GPU execution.
- Frame output and timing.

Do not assume older FSR 4.0.x artifacts represent the current implementation.

#### Checkpoints

- [x] C01: Identify the exact FSR SDK and runtime versions.
- [x] C02: Build or run an official supported sample.
- [x] C03: Verify that the ML upscaling path is selected.
- [x] C04: Execute Super Resolution on gfx1101.
- [x] C05: Capture output frames.
- [x] C06: Measure upscaling GPU time.
- [x] C07: Record resolution and quality settings.
- [x] C08: Save a reproducible test configuration.

#### Validation Gate

- [x] Official implementation executes on the RX 7700 XT.
- [x] The ML-based path is confirmed rather than an analytical fallback.
- [x] A reconstructed frame is produced.
- [x] Output and input resolutions are recorded.
- [x] GPU timing is measured.
- [x] Runtime, driver, and SDK versions are recorded.
- [x] Experiment can be reproduced.

#### Deliverables

- Working official baseline.
- Reference frames.
- Initial performance measurements.
- Reproduction instructions.

#### Boundary

Do not modify or reimplement the model.

---

### M04 — FSR 4 Architecture Investigation

**Status:** ☑ complete (2026-10-27 — [`findings/2026-10-27-fsr4-architecture.md`](findings/2026-10-27-fsr4-architecture.md))

#### Objective

Understand the structure of AMD's INT8 inference pipeline.

#### Scope

Investigate:

- Model variants.
- Neural network topology.
- Input and output tensors.
- Temporal inputs.
- Tensor dimensions.
- Weight storage.
- Intermediate precision.
- Quantization metadata.
- Shader execution.
- ML2Code or other code-generation mechanisms.
- Preprocessing and postprocessing.

Distinguish established facts from hypotheses. Do not assume the FP8 and INT8 networks are identical.

#### Checkpoints

- [x] C01: Document model inputs and outputs.
- [x] C02: Identify available model representations.
- [x] C03: Reconstruct the known execution graph.
- [x] C04: Identify quantization and scaling operations.
- [x] C05: Identify network operations requiring reproduction.
- [x] C06: Identify intermediate tensor formats.
- [x] C07: Record version-specific implementation differences.
- [x] C08: Identify unavailable or proprietary components.

#### Validation Gate

- [x] Known input and output contracts are documented.
- [x] Available network operations are identified.
- [x] Known tensor formats are documented.
- [x] Quantization assumptions are explicitly classified.
- [x] Every documented graph component has supporting evidence.
- [x] Missing graph information is clearly identified.
- [x] A concrete implementation feasibility assessment exists.

#### Deliverables

- Architecture report.
- Execution graph.
- Tensor-format notes.
- Unknowns and feasibility report.

#### Boundary

A complete graph is not required if unavailable from legitimate references. Do not invent missing details to pass validation.

---

## Phase II — RDNA 3 Hardware

> Goal: understand the GPU's actual capabilities and reproduce its native numerical operations.

### M05 — DirectX 12 Compute Foundation

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-003-d3d12-compute-harness.md`](experiments/EXP-003-d3d12-compute-harness.md))

#### Objective

Establish a minimal native GPU compute environment.

#### Scope

- D3D12 device creation.
- Compute pipelines.
- HLSL compilation.
- GPU buffers.
- Upload and readback.
- Command submission.
- Resource synchronization.
- GPU timestamps.

#### Checkpoints

- [x] C01: Create a D3D12 compute device.
- [x] C02: Compile an HLSL compute shader.
- [x] C03: Upload input data to the GPU.
- [x] C04: Execute an elementary compute operation.
- [x] C05: Read the results back.
- [x] C06: Implement GPU timestamp measurement.
- [x] C07: Enable relevant GPU validation diagnostics.

#### Validation Gate

- [x] Compute dispatch executes successfully.
- [x] Results match an independent CPU reference.
- [x] Resource synchronization is correct.
- [x] Debug diagnostics report no relevant errors.
- [x] GPU execution time can be measured.
- [x] Repeated executions produce consistent results.

#### Deliverables

- Minimal D3D12 compute harness.
- CPU comparison utility.
- GPU timing utility.

#### Boundary

No neural network implementation.

---

### M06 — Native INT8 WMMA

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-004-int8-wmma.md`](experiments/EXP-004-int8-wmma.md))

#### Objective

Prove that native RDNA 3 INT8 matrix instructions can be used correctly on gfx1101.

#### Scope

Study and reproduce:

- Wave32 execution.
- WMMA tile organization.
- Signed INT8 operands.
- INT32 accumulation.
- Operand register layouts.
- Matrix packing.
- Instruction generation.

Investigate whether the selected HLSL/DXC path can generate the required native WMMA instructions. If not, test a minimal alternative native AMD compute path, such as HIP, without replacing the project's D3D12 frontend.

#### Checkpoints

- [x] C01: Document INT8 WMMA operand layouts.
- [x] C02: Verify available compiler paths.
- [x] C03: Implement a single matrix tile.
- [x] C04: Validate signedness and accumulation.
- [x] C05: Inspect generated GPU ISA.
- [x] C06: Verify actual WMMA instruction execution.
- [x] C07: Measure kernel execution time.
- [x] C08: Record the chosen compute path and its limitations.

#### Validation Gate

- [x] Matrix results match a CPU INT32 reference.
- [x] Positive and negative operands are validated.
- [x] Accumulation behavior is verified.
- [x] Expected WMMA instructions are confirmed in generated ISA.
- [x] The instruction executes on gfx1101.
- [x] Timing measurements are reproducible.
- [x] Compiler and execution limitations are documented.

#### Deliverables

- Minimal INT8 WMMA experiment.
- GPU instruction evidence.
- CPU correctness tests.
- Initial timing results.
- Compute-path decision record.

#### Critical Decision Gate

If D3D12/HLSL cannot reliably emit native INT8 WMMA, determine the smallest viable alternative before continuing. Do not substitute ordinary integer arithmetic while claiming hardware WMMA acceleration. The objective is native execution, not loyalty to one compiler.

---

### M07 — FP16 WMMA and Hardware Comparison

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-005-fp16-wmma-comparison.md`](experiments/EXP-005-fp16-wmma-comparison.md))

#### Objective

Understand the hardware-level performance differences between FP16 and INT8 on RDNA 3.

#### Scope

Implement equivalent FP16 matrix operations. Investigate:

- FP16 inputs.
- FP16 and FP32 accumulation variants.
- WMMA operand layouts.
- Register pressure.
- Instruction throughput.
- Memory traffic.
- Occupancy.

#### Checkpoints

- [x] C01: Implement native FP16 WMMA.
- [x] C02: Validate FP16 numerical results.
- [x] C03: Confirm generated WMMA instructions.
- [x] C04: Measure comparable FP16 and INT8 workloads.
- [x] C05: Compare operand and accumulator requirements.
- [x] C06: Investigate occupancy and register usage.
- [x] C07: Distinguish arithmetic from memory bottlenecks.

#### Validation Gate

- [x] FP16 implementation produces valid results.
- [x] Native FP16 WMMA instructions are confirmed.
- [x] Both precisions use comparable matrix workloads.
- [x] Relevant accumulator differences are documented.
- [x] Timing measurements are repeated.
- [x] Warm-up and measurement methodology are documented.
- [x] Results distinguish theoretical and observed throughput.

#### Deliverables

- FP16 WMMA implementation.
- INT8 versus FP16 benchmark.
- Hardware performance report.

#### Boundary

No attempt to optimize an entire neural network yet.

---

## Phase III — Numerical Representation

> Goal: understand and implement the numerical transformations necessary for native INT8 inference.

### M08 — FP8 and INT8 Quantization

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-006-quantization.md`](experiments/EXP-006-quantization.md))

#### Objective

Implement and characterize FP8-to-INT8 conversion and the quantization methods relevant to AMD's implementation.

#### Scope

Investigate:

- FP8 E4M3.
- FP8 E5M2.
- Signed INT8.
- Symmetric quantization.
- Asymmetric quantization where relevant.
- Per-tensor scaling.
- Per-channel scaling.
- Activation quantization.
- Rounding behavior.
- Saturation.
- Numerical error.

The formats and schemes actually used by FSR 4 must be established from evidence whenever possible. Do not assume AMD directly converts FP8 weights to INT8 — the FP8 and INT8 models may be separately quantized.

#### Checkpoints

- [x] C01: Implement FP8 decoding.
- [x] C02: Validate special FP8 values and edge cases.
- [x] C03: Implement symmetric INT8 quantization.
- [x] C04: Implement scaling and dequantization.
- [x] C05: Test per-tensor and per-channel approaches.
- [x] C06: Evaluate representative weight distributions.
- [x] C07: Evaluate activation distributions.
- [x] C08: Measure numerical error.
- [x] C09: Document which methods match observed AMD behavior.

#### Validation Gate

- [x] FP8 decoding passes independent reference tests.
- [x] INT8 quantization boundaries are correct.
- [x] Scaling and dequantization are verified.
- [x] Saturation behavior is characterized.
- [x] Error metrics are reported.
- [x] Quantization metadata is preserved.
- [x] AMD-specific claims are distinguished from generic experiments.

#### Deliverables

- Quantization library.
- Numerical tests.
- Quantization error report.
- Comparison of candidate schemes.

#### Boundary

This milestone proves numerical transformations, not end-to-end network quality.

---

### M09 — INT8 Matrix and Convolution Kernels

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-007-int8-kernels.md`](experiments/EXP-007-int8-kernels.md))

#### Objective

Build reusable inference operations using the validated native INT8 execution path.

#### Scope

- Matrix multiplication.
- Batched matrix operations where required.
- Convolution.
- Tensor packing.
- Tensor layout transformations.
- INT32 accumulation.
- Scaling and requantization.

Start with straightforward implementations. Investigate implicit GEMM, direct convolution, or im2col only when motivated by the observed FSR 4 network operations.

#### Checkpoints

- [x] C01: Generalize the WMMA matrix operation.
- [x] C02: Support relevant non-multiple-of-tile dimensions.
- [x] C03: Implement one convolution configuration.
- [x] C04: Validate convolution against a CPU reference.
- [x] C05: Integrate quantization scales.
- [x] C06: Implement requantization.
- [x] C07: Measure register and memory behavior.
- [x] C08: Benchmark representative tensor shapes.

#### Validation Gate

- [x] Matrix multiplication handles required dimensions.
- [x] Convolution matches the quantized CPU reference.
- [x] Boundary tiles are correct.
- [x] Scaling is applied correctly.
- [x] Accumulation overflow is prevented or detected.
- [x] Native matrix instructions remain present.
- [x] Performance measurements are reproducible.
  Methodology updated 2026-10-27 (gate unchanged at 10%): the discarded warm
  cycle now runs 1000 timed dispatches (~0.2 s sustained load) and measured
  runs take min-of-1000 — the former min-of-100 estimator proved flaky under
  CTest scheduling (observed 35% Debug / 10.6% Release spread; EXP-009).

#### Deliverables

- INT8 GEMM kernels.
- INT8 convolution kernels.
- CPU reference implementations.
- Correctness and performance results.

#### Boundary

Do not build a generic deep-learning library. Implement only operations justified by the research.

---

## Phase IV — Inference Engine

> Goal: move from isolated GPU operations to complete neural inference.

### M10 — Neural Network Operations

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-008-neural-operators.md`](experiments/EXP-008-neural-operators.md))

[M] Gate PASS on gfx1101: 60 hardware operator cases and 24 rejection/boundary
checks; final rebuilt Debug and Release CTest 9/9. Scope is the non-matrix
constituents of FSR 4.0.2 v07 INT8 Quality main passes, not a complete FSR model
or GPU-resident network. Source inventory and tensor/precision contract:
[`findings/2026-10-27-m10-operator-inventory.md`](findings/2026-10-27-m10-operator-inventory.md).

#### Objective

Implement the non-matrix operations required to execute the selected FSR 4 network architecture.

#### Scope

Depending on confirmed network requirements:

- Activation functions.
- Tensor reshaping.
- Tensor concatenation.
- Elementwise operations.
- Residual connections.
- Normalization.
- Upsampling.
- Quantize/dequantize boundaries.
- Intermediate buffer management.

Do not implement an operation merely because it is common in neural networks.

#### Checkpoints

- [x] C01: Identify the exact required operator set.
- [x] C02: Implement the minimal operator set.
- [x] C03: Validate tensor shapes and layouts.
- [x] C04: Validate each operation against a CPU reference.
- [x] C05: Test transitions between integer and floating-point values.
- [x] C06: Document intermediate precision requirements.

#### Validation Gate

- [x] Required operators are implemented.
- [x] Every operator passes independent correctness tests.
- [x] Tensor layouts are consistent.
- [x] Mixed-precision boundaries are explicit.
- [x] Intermediate results can be inspected.
- [x] No speculative operators or abstraction layers were added.

#### Deliverables

- Minimal operator collection.
- Operator tests.
- Tensor-format documentation.

---

### M11 — Standalone INT8 Neural Inference

**Status:** ☑ complete (2026-10-27 — [`experiments/EXP-009-fixed-network.md`](experiments/EXP-009-fixed-network.md))

[M] Gate PASS on gfx1101: the fixed two-layer INT8 network (16×16×16 WMMA
GEMMs + ReLU requantization) executes GPU-resident end-to-end; all
intermediates and the final output match the independent CPU reference
exactly; first-vs-last output equality holds over 100 timed executions
(each timed pass is independent and fully overwriting by construction);
~41 µs end-to-end (min of 100, Debug). Final rebuilt CTest 11/11 on Debug
and Release.

#### Objective

Execute a complete quantized neural network on gfx1101. This is the first end-to-end inference milestone. Use a small, known network independent of FSR 4 to validate the execution engine before introducing proprietary or incompletely documented model details.

#### Scope

- Network representation.
- Weight loading.
- Sequential operation execution.
- Intermediate tensor management.
- Quantization scales.
- CPU comparison.
- GPU timing.

Use a fixed execution graph. Do not implement a general model parser or graph compiler.

#### Checkpoints

- [x] C01: Define a minimal test network.
- [x] C02: Establish its CPU reference implementation.
- [x] C03: Load quantized weights.
- [x] C04: Execute every layer on the GPU.
- [x] C05: Compare intermediate tensors.
- [x] C06: Compare final outputs.
- [x] C07: Measure end-to-end inference time.
- [x] C08: Verify repeated execution.

#### Validation Gate

- [x] Entire network executes on gfx1101.
- [x] Every required layer executes.
- [x] Intermediate tensors have correct shapes.
- [x] Final output matches the CPU quantized reference within tolerance.
- [x] No unexplained overflow or memory corruption occurs.
- [x] Execution is deterministic under fixed inputs and conditions.
- [x] GPU timing is recorded.

#### Deliverables

- Standalone INT8 inference experiment within the primary CLI (`rdna3-int8.exe
  run fixed-network`) — the single-executable architecture rule (architecture
  §6) intentionally precludes a separate executable.
- Reference network.
- CPU/GPU comparison results.
- Performance baseline.

#### Boundary

No FSR 4-specific model implementation yet.

---

## Phase V — FSR 4 Reproduction

> Goal: reproduce AMD's neural Super Resolution pipeline using the validated native inference components.

### M12 — FSR 4 Model Reproduction

**Status:** ✓ complete — **M12 gate PASS** on gfx1101, 2026-10-28.
C01–C09 have evidence. All 14 independent passes execute GPU-resident in one
submission, with every logical boundary byte-exact on three fixtures against
the **labelled pass-11 predication-control graph**. Differences against the
unmodified raced reference are separately measured, not hidden or tolerated.
Debug **22/22 PASS (1750.60 s)**; Release **22/22 PASS (1866.75 s)**.
Gate evidence: [EXP-016](experiments/EXP-016-fsr4-resident-graph.md).
Before further integration work, read the
[blocker-prevention finding](findings/2026-10-28-fsr4-integration-blockers.md).
Historical incremental evidence below retains its original scope/status. See
[`experiments/EXP-010-fsr4-model-graph.md`](experiments/EXP-010-fsr4-model-graph.md)
and [`findings/2026-10-27-fsr4-v07-graph.md`](findings/2026-10-27-fsr4-v07-graph.md).
Operator/weight evidence:
[`experiments/EXP-011-fsr4-reproduction-operators.md`](experiments/EXP-011-fsr4-reproduction-operators.md),
[`experiments/EXP-012-fsr4-downscale.md`](experiments/EXP-012-fsr4-downscale.md),
[`experiments/EXP-013-fsr4-fasternet.md`](experiments/EXP-013-fsr4-fasternet.md)
and [`experiments/EXP-014-fsr4-cnb-ct2d.md`](experiments/EXP-014-fsr4-cnb-ct2d.md).

Selected reference: **FSR 4.0.2, model `fsr4_model_v07_i8_quality`, 1080p**
(`FSR-4.0.2-reference` @ `c5e34b4`) — not FSR 4.1, whose graph is closed.

[M] Reference established (2026-10-27): the 14-pass graph was extracted
mechanically (4 fused block families, 67 embedded weight arrays + 11 initializer
tensors, 20,736,000 B scratch) and the shipped passes execute on gfx1101 as the
numeric reference (1920×1080×8 FP16, no NaN/Inf). Passes 0–10 are
bit-deterministic; pass 11 stores past its logical width without a predicate, so
the reference itself varies in x<132 for the recorded fixture. Difference
counts vary by run; percentages refer to output halfwords, not pixels.
The explicit predication control and retained multi-run captures are described
in EXP-010. The gate is not met.

[M] Independent operator progress: our input convolution matches the shipped
pass 0 on three fixtures; our native-WMMA ConvNextBlocks match isolated passes
1/2/12 on zero and full-range INT8 fixtures. Weight loading validates all 78
descriptors over a pinned 136,848-byte payload (C05). Static metadata validation
checks all 14 logical shapes, producer lifetimes and group/skip scales, but C06
still requires the remaining operators' execution evidence. See EXP-011.

[M] Checkpoint validation: **Debug 18/18 PASS (286.60 s)** and **Release 18/18
PASS (281.94 s)**, with verbose exact-comparison/repeat evidence and separate
artifact-identity manifests under `build/reports/m12/`. Adversarial review
accepted the partial checkpoint conditional on these results. **M12 gate:
UNVERIFIED**, not PASS; ten model passes and full-graph integration remain.

[M] Downscale family (2026-10-28, EXP-012): our own GPU pipeline for passes 3
and 6 is byte-exact against the isolated shipped passes on zero and full-range
INT8 fixtures — **0/4,147,200** and **0/2,073,600** differing bytes, max delta
0, deterministic, native `v_wmma_i32_16x16x16_iu8` confirmed. Exactness required
the reference's *compiler-folded* requantization form, derived from its ISA, not
its source expression
[M] FasterNetBlock family (2026-10-28, EXP-013): passes 4, 5, 7, 8 and 10
reproduced as fused per-pixel DP4A kernels, **0 differing bytes** on zero and
full-range INT8 fixtures (4,147,200 and 2,073,600 values per pass), deterministic.
Exactness again required ISA-derived folded constants, here
`fl(fl(a·b)·fl(1/c))` rather than a division. Debug CTest
**19/19 PASS (369.88 s)** and Release **19/19 PASS (371.28 s)** for the
downscale increment. FNB_CT2D_ADD (2026-10-28, EXP-013): passes 9 and 11 reproduced with the fused
2x upscale and add � 0 differing bytes for pass 9 against the unmodified
reference, and 0/8,294,400 for pass 11 against the labelled predication control
(its shipped 32/1 overload is unpredicated). CNB_CT2D (2026-10-28, EXP-014): pass 13 reproduced with the fused 2x upscale
and FP16 output � 0/16,588,800 halfword mismatches on both fixtures. Every
operator of the pinned model now has an independent byte-exact reproduction;
C04 is evidenced. What remains for the gate is integration: C07 (GPU-resident
full graph) and C08/C09 (per-pass and end-to-end comparison).
Debug and Release CTest both **20/20 PASS** (782.62 s / 771.07 s) at this state.

[M] GPU-resident integration checkpoint (2026-10-28,
[EXP-015](experiments/EXP-015-fsr4-resident-chain.md)): passes 0 -> 1 execute in
one submission (ten dispatches), with separate live resources and unchanged
shaders/aliasing checks. Both logical boundaries match the shipped two-pass
reference chain: **0/8,294,400 differing bytes** on each of three fixtures;
snapshot-free repeats identical, native WMMA confirmed. The earlier single-
scratch aliasing obstacle is bypassed, not waived. C06/C07 remain partial and
C08/C09 remain incomplete for the full graph; **M12 gate UNVERIFIED**.

[M] Full integration (EXP-016, supersedes the partial statuses above): all 14
model boundaries are exact on zero, impulse and deterministic-feature fixtures;
snapshot-free repeats identical, 11/11 native WMMA convolutions confirmed.
All 16,588,800 final FP16 halfwords are finite. Deterministic-feature output
SHA-256 matches EXP-010's independent control:
`a57bcd063967affddf29cc24709b9d8a7fd9f8e7244e7010521d7b27f605b6de`.
Unmodified shipped differences occur only in x<64 / x<66 / x<132 at passes
11/12/13; exact byte/halfword counts and deltas are recorded in EXP-016.
C06–C09 and all gate items below are evidenced. **Stop here: M13 is not
authorized by completing M12.** No signed-runtime, temporal, visual-quality,
performance, alternate-resolution or FSR 4.1 equivalence claim.

#### Objective

Reproduce the selected FSR 4 INT8 network's execution graph as closely as available evidence permits.

#### Prerequisites

- M04 architecture investigation.
- M11 working INT8 inference engine.
- Sufficient model metadata.
- Legally usable model artifacts, or a documented alternative.

#### Scope

- FSR 4 network topology.
- Weight representation.
- Layer ordering.
- Tensor dimensions.
- Intermediate formats.
- Scaling parameters.
- Model execution.

Use a fixed version of the model as the reference. Do not silently mix different FSR 4 versions.

#### Checkpoints

- [x] C01: Select and pin the reference model version.
- [x] C02: Establish model artifact availability.
- [x] C03: Reproduce the known execution graph.
- [x] C04: Implement required missing operations.
- [x] C05: Load the available model weights.
- [x] C06: Validate intermediate tensor dimensions.
- [x] C07: Execute the complete graph.
- [x] C08: Compare against an available reference.
- [x] C09: Record numerical differences.

#### Validation Gate

- [x] Selected model version is documented.
- [x] Every required layer is accounted for.
- [x] Model weights are loaded correctly.
- [x] Quantization metadata is applied.
- [x] Complete graph executes.
- [x] Intermediate tensor dimensions are valid.
- [x] Output contains finite, inspectable values.
- [x] Numerical comparison is performed where a reference is available.
- [x] Missing or approximated behavior is explicitly documented.

#### Deliverables

- FSR 4 model execution experiment.
- Tensor and layer documentation.
- Intermediate output captures.
- Numerical comparison report.

#### Conditional Gate

If actual model weights or graph details are unavailable, do not fabricate equivalence. A substitute-weight implementation can prove graph execution, but cannot pass the full model-reproduction gate. Mark the milestone BLOCKED or PARTIAL and document the maximum verified capability.

Reverse engineering may be considered only for specific, unanswered questions and within legitimate access boundaries.

---

### M13 — FSR 4 Super Resolution Pipeline

**Status:** ✓ complete — **M13 execution gate PASS** on gfx1101 for the explicit
1280x720 -> 1920x1080 linear/manual-exposure/LR-motion/inverted-depth/no-RCAS
experimental permutation. Three consecutive frames execute pre -> all 14 neural
passes -> learned reconstruction with GPU-resident history and recurrence.
Camera-cut reset reproduces the clean initial frame byte-for-byte; NaN-motion
local invalidation passes; frames are finite/inspectable and stage GPU
timestamps are available. Debug **23/23 PASS (1880.33 s)**; Release
**23/23 PASS (1701.30 s)**, including M12 exact regressions.
See [EXP-017](experiments/EXP-017-fsr4-temporal-pipeline.md) and the
[temporal contract](findings/2026-10-28-fsr4-temporal-contract.md).

[S] User authorization now covers all remaining milestones (M13–M15), still
subject to ordered, evidence-backed gates. This supersedes the earlier
authorization stop after M12; it does not authorize skipping failed gates.
M14/M15 are not passed by M13 execution. Software-vs-texture sampler
equivalence, visual equivalence and broad/long-sequence stability remain
**UNVERIFIED**. The current temporal API is an explicit 1–3-frame batch with an
initial reset, not a persistent streaming pool.

#### Objective

Integrate the neural inference implementation into a standalone temporal Super Resolution pipeline.

#### Scope

Investigate and implement the required:

- Low-resolution color input.
- Depth input.
- Motion vectors.
- Exposure information.
- Frame jitter.
- Temporal history.
- Preprocessing.
- Neural inference.
- Postprocessing.
- High-resolution output.

The exact contract must follow the selected reference version.

#### Checkpoints

- [x] C01: Establish the complete input contract.
- [x] C02: Build deterministic test sequences.
- [x] C03: Implement preprocessing.
- [x] C04: Connect the network.
- [x] C05: Implement output postprocessing.
- [x] C06: Establish temporal history buffers.
- [x] C07: Execute consecutive frames.
- [x] C08: Capture reconstructed images.
- [x] C09: Test history reset and invalidation.

#### Validation Gate

- [x] Complete input pipeline executes.
- [x] Temporal history is managed correctly.
- [x] Multiple consecutive frames execute.
- [x] Output resolution is correct.
- [x] Output frames are inspectable.
- [x] Camera cuts and history resets are handled.
- [x] Missing operations are not silently bypassed.
- [x] GPU timing can be measured per pipeline stage.

#### Deliverables

- Standalone Super Resolution executable.
- Reproducible frame sequences.
- Image captures.
- Per-stage GPU timings.

#### Boundary

Visual artifacts are acceptable during initial execution. Crashes, undefined behavior, and incomplete execution cannot be treated as successful reconstruction.

---

### M14 — Quality and Numerical Analysis

**Status:** ✓ complete — **M14 analysis gate PASS** for the bounded synthetic
static-jitter and moving/reset sequences, with an explicitly **cross-version**
official 4.1.2 reference and experimental 4.0.2 pipeline. Identical input bytes
are verified; images, raw error metrics, edge/interior categories, short-window
temporal residuals, overshoot and reset equality are recorded in
[EXP-018](experiments/EXP-018-fsr4-image-comparison.md).
More than 99.9% of squared RGB difference occurs in the predefined edge mask;
large local maxima are reported alongside global PSNR. No isolated INT8-error,
same-model equivalence, long-sequence stability or real-game quality claim.

[M] Both local DLL bundles report/select provider **4.1.2**, including the DLL
under `FSR-4.0.2-reference`; no 4.0.2 provider was enumerated on gfx1101. Runtime
version is checked by API, never inferred from the folder. Release CTest
**24/24 PASS (2035.72 s)**; the later Debug attempt was interrupted and is
**incomplete**, not a pass. Finer causal separation of closed-model/version,
sampler precision and quantization remains unverified and is not fabricated.

#### Objective

Understand how closely the experimental implementation reproduces AMD's original output. Quality improvement is secondary to explaining differences.

#### Scope

Compare:

- Official FSR 4 INT8 output.
- Experimental implementation.
- Native-resolution ground truth where available.
- Optional FP16 experimental variants.

Investigate:

- Quantization error.
- Precision loss.
- Temporal instability.
- Ghosting.
- Flickering.
- Detail reconstruction.
- Intermediate tensor differences.

#### Checkpoints

- [x] C01: Establish identical input sequences where possible.
- [x] C02: Capture official reference output.
- [x] C03: Capture experimental output.
- [x] C04: Compare static scenes.
- [x] C05: Compare moving scenes.
- [x] C06: Evaluate frame-to-frame stability.
- [x] C07: Quantify numerical differences.
- [x] C08: Identify dominant error sources (observed edge/reconstruction/temporal category; finer attribution unverified).

#### Validation Gate

- [x] Comparable input sequences are available.
- [x] Reference and experimental outputs are captured.
- [x] Image comparison methodology is documented.
- [x] Numerical differences are measured.
- [x] Temporal behavior is evaluated.
- [x] Dominant visible artifacts are categorized.
- [x] Unsupported claims of numerical equivalence are avoided.

#### Deliverables

- Image comparison report.
- Numerical error analysis.
- Temporal stability analysis.
- Known limitations.

#### Boundary

Do not optimize kernels during this milestone unless a correctness problem requires a targeted change.

---

### M15 — Native INT8 Performance Optimization

**Status:** ◐ final regression validation in progress — performance and exact
hardware checks recorded in
[EXP-019](experiments/EXP-019-fsr4-native-performance.md).
[M] Complete 14-operator neural graph: reproducible baseline **55.76–57.06 ms**
GPU span; opt-in workspace-upload omission **34.90–35.21 ms**. Upload volume
**688,594,376→33,314,448 B**; default allocation unchanged. All 14 boundaries on
all three retained fixtures are byte-exact for baseline, skip and poison.
Equivalent-workload FP16/F32 matrix backend (same quantized weights/epilogues,
conversion included) measures **49.68–51.74 ms**, also 42/42 exact boundaries.
Failed truncating conversions and the slower GPU-zero candidate are retained.
This is a bounded network-invocation study, not whole-SR/game frame latency or
a proof of the hardware ceiling. Default temporal-history initialization stays
unchanged. Final selected Release regression run is pending; no new complete
Debug or complete Release suite verdict is implied.

#### Objective

Determine the practical performance ceiling of native INT8 inference on the RX 7700 XT. Optimize based on measurements, not assumptions.

#### Scope

Investigate:

- Kernel dispatch overhead.
- GPU occupancy.
- Register pressure.
- LDS usage.
- Tensor layouts.
- Cache efficiency.
- Memory bandwidth.
- Weight loading.
- Quantization overhead.
- Convolution strategy.
- Kernel fusion.
- Buffer reuse.
- Intermediate precision.

Compare against the FP16 experiments from M07.

#### Checkpoints

- [x] C01: Profile the complete inference pipeline (14-operator neural graph, not whole SR).
- [x] C02: Identify the highest-cost operations.
- [x] C03: Establish per-layer timing.
- [x] C04: Measure memory and register behavior (byte counts/compiled registers; occupancy and cache counters unmeasured).
- [x] C05: Select one measurable bottleneck.
- [x] C06: Implement one optimization.
- [ ] C07: Re-run correctness tests (42/42 controls passed; final selected regressions pending).
- [x] C08: Measure the performance difference.
- [x] C09: Compare INT8 and FP16 end-to-end costs (equivalent matrix backend within full network invocation).
- [x] C10: Record unsuccessful optimizations.

#### Validation Gate

- [x] Baseline performance is reproducible.
- [x] Dominant bottlenecks are identified with evidence.
- [x] Each optimization has before/after measurements.
- [ ] Numerical correctness is preserved within documented tolerances (exact controls passed; final regressions pending).
- [x] Memory and conversion overheads are accounted for.
- [x] GPU execution and transfer times are distinguished.
- [x] INT8 and FP16 are compared using equivalent workloads.
- [x] Results are reported without selective benchmarking.

#### Deliverables

- Optimized experimental implementation.
- Detailed performance report.
- INT8 versus FP16 comparison.
- Optimization findings.
- Remaining bottleneck analysis.

#### Boundary

A successful optimization milestone does not require INT8 to outperform FP16. A rigorously demonstrated disadvantage is also a valid experimental result.

---

## 3. Final Research Questions

The project should ultimately answer:

### Hardware

1. What native INT8 matrix capabilities does gfx1101 provide?
2. How are matrix operands distributed across wavefront lanes?
3. What are the practical throughput and register costs?
4. How do INT8 and FP16 differ on equivalent workloads?
5. Which limitations come from hardware versus compiler behavior?

### Quantization

6. How does AMD represent INT8 neural network weights?
7. Which quantization and scaling techniques are actually used?
8. Are FP8 and INT8 models independently quantized?
9. What precision is used for intermediate operations?
10. Where does quantization introduce the most error?

### Inference

11. How does AMD organize neural inference execution?
12. How are convolutions mapped to native GPU operations?
13. How are intermediate tensors stored and transformed?
14. Which operations dominate inference time?
15. How much overhead comes from memory movement and conversion?

### Super Resolution

16. What inputs does the selected FSR 4 model require?
17. How does it use temporal information?
18. How does its preprocessing and postprocessing work?
19. Which components can be reproduced from public information?
20. How closely can native experimental inference reproduce AMD's output?

### Performance

21. Is INT8 faster than FP16 for the relevant network?
22. Which workloads benefit from INT8?
23. Which workloads remain compute-bound?
24. Which workloads are memory-bound?
25. What is the best measured latency achievable on the RX 7700 XT?

---

## 4. Phase Completion Gates

### Phase I

Completed when:

- Official FSR 4 runs as a reference.
- Research references are organized.
- The known architecture is documented.
- Unknowns and model availability are identified.

### Phase II

Completed when:

- Native D3D12 compute works.
- INT8 WMMA executes on gfx1101.
- FP16 WMMA executes on gfx1101.
- Both precisions have reliable baseline measurements.
- The compute toolchain decision is supported by evidence.

### Phase III

Completed when:

- Quantization behavior is characterized.
- Native INT8 matrix and convolution operations work.
- Numerical correctness is independently validated.
- Relevant neural network primitives are available.

### Phase IV

Completed when:

- A complete standalone INT8 network executes.
- Intermediate tensors are validated.
- CPU and GPU inference are compared.
- End-to-end timing is measured.

### Phase V

Completed when:

- The available FSR 4 inference graph is reproduced.
- A temporal Super Resolution pipeline executes.
- Quality differences are characterized.
- Performance limitations are measured.

If proprietary or unavailable artifacts block exact reproduction, record the partial result and the unmet prerequisites. Do not redefine exact reproduction as success merely because a different network produces plausible frames.

---

## 5. Research Artifacts

Keep documentation minimal. Suggested structure:

```text
docs/
├── MILESTONES.md
├── architecture.md
├── findings/
│   └── ...
└── experiments/
    └── ...
```

Each experiment should record:

- Objective.
- Hypothesis.
- Reference sources.
- Relevant configuration.
- Procedure.
- Expected result.
- Actual result.
- Measurements.
- Conclusion.

Store large captures and profiling outputs only when useful. Do not introduce a database, research framework, or automated documentation pipeline.

---

## 6. Final Success Criteria

The experiment has three independent levels of success.

### Level 1 — Native Execution

- INT8 WMMA is confirmed on gfx1101.
- Quantized inference executes successfully.
- Numerical behavior is understood.
- Performance is measurable.

### Level 2 — FSR 4 Reproduction

- Relevant FSR 4 architecture is reproduced.
- Model execution is validated.
- Temporal Super Resolution works.
- Output can be compared with AMD's implementation.

### Level 3 — Performance Characterization

- INT8 versus FP16 performance is measured.
- Major bottlenecks are identified.
- Optimization techniques are tested.
- The practical limitations of RDNA 3 inference are documented.

Each level is valuable independently. A negative result is acceptable when it is reproducible, well understood, and supported by evidence.

---

## 7. Execution Instruction

Do not implement the entire roadmap immediately. Begin with the first incomplete milestone.

For each milestone:

1. Research.
2. Define the smallest experiment.
3. Implement.
4. Validate.
5. Document findings.
6. Report the validation gate.
7. Stop.

Do not automatically advance to the next milestone. The project exists to understand the system, not merely to produce a working implementation.




