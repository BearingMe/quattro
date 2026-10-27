# Native RDNA 3 INT8 Exploration — Milestones and Validation Gates

| Field | Value |
|---|---|
| Target | AMD Radeon RX 7700 XT (gfx1101) |
| Platform | Windows 11 |
| Primary case study | AMD FSR 4.1 INT8 Super Resolution |
| Initial graphics API | DirectX 12 |
| Initial shader language | HLSL |
| Status | Roadmap — M01–M02 complete (see per-milestone status) |

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

**Status:** ☑ complete (2026-10-27 — see [`.references/`](../.references/README.md), commit `c3af362`)

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

**Status:** ☐ proposed

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

- [ ] C01: Identify the exact FSR SDK and runtime versions.
- [ ] C02: Build or run an official supported sample.
- [ ] C03: Verify that the ML upscaling path is selected.
- [ ] C04: Execute Super Resolution on gfx1101.
- [ ] C05: Capture output frames.
- [ ] C06: Measure upscaling GPU time.
- [ ] C07: Record resolution and quality settings.
- [ ] C08: Save a reproducible test configuration.

#### Validation Gate

- [ ] Official implementation executes on the RX 7700 XT.
- [ ] The ML-based path is confirmed rather than an analytical fallback.
- [ ] A reconstructed frame is produced.
- [ ] Output and input resolutions are recorded.
- [ ] GPU timing is measured.
- [ ] Runtime, driver, and SDK versions are recorded.
- [ ] Experiment can be reproduced.

#### Deliverables

- Working official baseline.
- Reference frames.
- Initial performance measurements.
- Reproduction instructions.

#### Boundary

Do not modify or reimplement the model.

---

### M04 — FSR 4 Architecture Investigation

**Status:** ☐ proposed

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

- [ ] C01: Document model inputs and outputs.
- [ ] C02: Identify available model representations.
- [ ] C03: Reconstruct the known execution graph.
- [ ] C04: Identify quantization and scaling operations.
- [ ] C05: Identify network operations requiring reproduction.
- [ ] C06: Identify intermediate tensor formats.
- [ ] C07: Record version-specific implementation differences.
- [ ] C08: Identify unavailable or proprietary components.

#### Validation Gate

- [ ] Known input and output contracts are documented.
- [ ] Available network operations are identified.
- [ ] Known tensor formats are documented.
- [ ] Quantization assumptions are explicitly classified.
- [ ] Every documented graph component has supporting evidence.
- [ ] Missing graph information is clearly identified.
- [ ] A concrete implementation feasibility assessment exists.

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

**Status:** ☐ proposed

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

- [ ] C01: Create a D3D12 compute device.
- [ ] C02: Compile an HLSL compute shader.
- [ ] C03: Upload input data to the GPU.
- [ ] C04: Execute an elementary compute operation.
- [ ] C05: Read the results back.
- [ ] C06: Implement GPU timestamp measurement.
- [ ] C07: Enable relevant GPU validation diagnostics.

#### Validation Gate

- [ ] Compute dispatch executes successfully.
- [ ] Results match an independent CPU reference.
- [ ] Resource synchronization is correct.
- [ ] Debug diagnostics report no relevant errors.
- [ ] GPU execution time can be measured.
- [ ] Repeated executions produce consistent results.

#### Deliverables

- Minimal D3D12 compute harness.
- CPU comparison utility.
- GPU timing utility.

#### Boundary

No neural network implementation.

---

### M06 — Native INT8 WMMA

**Status:** ☐ proposed

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

- [ ] C01: Document INT8 WMMA operand layouts.
- [ ] C02: Verify available compiler paths.
- [ ] C03: Implement a single matrix tile.
- [ ] C04: Validate signedness and accumulation.
- [ ] C05: Inspect generated GPU ISA.
- [ ] C06: Verify actual WMMA instruction execution.
- [ ] C07: Measure kernel execution time.
- [ ] C08: Record the chosen compute path and its limitations.

#### Validation Gate

- [ ] Matrix results match a CPU INT32 reference.
- [ ] Positive and negative operands are validated.
- [ ] Accumulation behavior is verified.
- [ ] Expected WMMA instructions are confirmed in generated ISA.
- [ ] The instruction executes on gfx1101.
- [ ] Timing measurements are reproducible.
- [ ] Compiler and execution limitations are documented.

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

**Status:** ☐ proposed

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

- [ ] C01: Implement native FP16 WMMA.
- [ ] C02: Validate FP16 numerical results.
- [ ] C03: Confirm generated WMMA instructions.
- [ ] C04: Measure comparable FP16 and INT8 workloads.
- [ ] C05: Compare operand and accumulator requirements.
- [ ] C06: Investigate occupancy and register usage.
- [ ] C07: Distinguish arithmetic from memory bottlenecks.

#### Validation Gate

- [ ] FP16 implementation produces valid results.
- [ ] Native FP16 WMMA instructions are confirmed.
- [ ] Both precisions use comparable matrix workloads.
- [ ] Relevant accumulator differences are documented.
- [ ] Timing measurements are repeated.
- [ ] Warm-up and measurement methodology are documented.
- [ ] Results distinguish theoretical and observed throughput.

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

**Status:** ☐ proposed

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

- [ ] C01: Implement FP8 decoding.
- [ ] C02: Validate special FP8 values and edge cases.
- [ ] C03: Implement symmetric INT8 quantization.
- [ ] C04: Implement scaling and dequantization.
- [ ] C05: Test per-tensor and per-channel approaches.
- [ ] C06: Evaluate representative weight distributions.
- [ ] C07: Evaluate activation distributions.
- [ ] C08: Measure numerical error.
- [ ] C09: Document which methods match observed AMD behavior.

#### Validation Gate

- [ ] FP8 decoding passes independent reference tests.
- [ ] INT8 quantization boundaries are correct.
- [ ] Scaling and dequantization are verified.
- [ ] Saturation behavior is characterized.
- [ ] Error metrics are reported.
- [ ] Quantization metadata is preserved.
- [ ] AMD-specific claims are distinguished from generic experiments.

#### Deliverables

- Quantization library.
- Numerical tests.
- Quantization error report.
- Comparison of candidate schemes.

#### Boundary

This milestone proves numerical transformations, not end-to-end network quality.

---

### M09 — INT8 Matrix and Convolution Kernels

**Status:** ☐ proposed

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

- [ ] C01: Generalize the WMMA matrix operation.
- [ ] C02: Support relevant non-multiple-of-tile dimensions.
- [ ] C03: Implement one convolution configuration.
- [ ] C04: Validate convolution against a CPU reference.
- [ ] C05: Integrate quantization scales.
- [ ] C06: Implement requantization.
- [ ] C07: Measure register and memory behavior.
- [ ] C08: Benchmark representative tensor shapes.

#### Validation Gate

- [ ] Matrix multiplication handles required dimensions.
- [ ] Convolution matches the quantized CPU reference.
- [ ] Boundary tiles are correct.
- [ ] Scaling is applied correctly.
- [ ] Accumulation overflow is prevented or detected.
- [ ] Native matrix instructions remain present.
- [ ] Performance measurements are reproducible.

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

**Status:** ☐ proposed

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

- [ ] C01: Identify the exact required operator set.
- [ ] C02: Implement the minimal operator set.
- [ ] C03: Validate tensor shapes and layouts.
- [ ] C04: Validate each operation against a CPU reference.
- [ ] C05: Test transitions between integer and floating-point values.
- [ ] C06: Document intermediate precision requirements.

#### Validation Gate

- [ ] Required operators are implemented.
- [ ] Every operator passes independent correctness tests.
- [ ] Tensor layouts are consistent.
- [ ] Mixed-precision boundaries are explicit.
- [ ] Intermediate results can be inspected.
- [ ] No speculative operators or abstraction layers were added.

#### Deliverables

- Minimal operator collection.
- Operator tests.
- Tensor-format documentation.

---

### M11 — Standalone INT8 Neural Inference

**Status:** ☐ proposed

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

- [ ] C01: Define a minimal test network.
- [ ] C02: Establish its CPU reference implementation.
- [ ] C03: Load quantized weights.
- [ ] C04: Execute every layer on the GPU.
- [ ] C05: Compare intermediate tensors.
- [ ] C06: Compare final outputs.
- [ ] C07: Measure end-to-end inference time.
- [ ] C08: Verify repeated execution.

#### Validation Gate

- [ ] Entire network executes on gfx1101.
- [ ] Every required layer executes.
- [ ] Intermediate tensors have correct shapes.
- [ ] Final output matches the CPU quantized reference within tolerance.
- [ ] No unexplained overflow or memory corruption occurs.
- [ ] Execution is deterministic under fixed inputs and conditions.
- [ ] GPU timing is recorded.

#### Deliverables

- Standalone INT8 inference executable.
- Reference network.
- CPU/GPU comparison results.
- Performance baseline.

#### Boundary

No FSR 4-specific model implementation yet.

---

## Phase V — FSR 4 Reproduction

> Goal: reproduce AMD's neural Super Resolution pipeline using the validated native inference components.

### M12 — FSR 4 Model Reproduction

**Status:** ☐ proposed

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

- [ ] C01: Select and pin the reference model version.
- [ ] C02: Establish model artifact availability.
- [ ] C03: Reproduce the known execution graph.
- [ ] C04: Implement required missing operations.
- [ ] C05: Load the available model weights.
- [ ] C06: Validate intermediate tensor dimensions.
- [ ] C07: Execute the complete graph.
- [ ] C08: Compare against an available reference.
- [ ] C09: Record numerical differences.

#### Validation Gate

- [ ] Selected model version is documented.
- [ ] Every required layer is accounted for.
- [ ] Model weights are loaded correctly.
- [ ] Quantization metadata is applied.
- [ ] Complete graph executes.
- [ ] Intermediate tensor dimensions are valid.
- [ ] Output contains finite, inspectable values.
- [ ] Numerical comparison is performed where a reference is available.
- [ ] Missing or approximated behavior is explicitly documented.

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

**Status:** ☐ proposed

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

- [ ] C01: Establish the complete input contract.
- [ ] C02: Build deterministic test sequences.
- [ ] C03: Implement preprocessing.
- [ ] C04: Connect the network.
- [ ] C05: Implement output postprocessing.
- [ ] C06: Establish temporal history buffers.
- [ ] C07: Execute consecutive frames.
- [ ] C08: Capture reconstructed images.
- [ ] C09: Test history reset and invalidation.

#### Validation Gate

- [ ] Complete input pipeline executes.
- [ ] Temporal history is managed correctly.
- [ ] Multiple consecutive frames execute.
- [ ] Output resolution is correct.
- [ ] Output frames are inspectable.
- [ ] Camera cuts and history resets are handled.
- [ ] Missing operations are not silently bypassed.
- [ ] GPU timing can be measured per pipeline stage.

#### Deliverables

- Standalone Super Resolution executable.
- Reproducible frame sequences.
- Image captures.
- Per-stage GPU timings.

#### Boundary

Visual artifacts are acceptable during initial execution. Crashes, undefined behavior, and incomplete execution cannot be treated as successful reconstruction.

---

### M14 — Quality and Numerical Analysis

**Status:** ☐ proposed

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

- [ ] C01: Establish identical input sequences where possible.
- [ ] C02: Capture official reference output.
- [ ] C03: Capture experimental output.
- [ ] C04: Compare static scenes.
- [ ] C05: Compare moving scenes.
- [ ] C06: Evaluate frame-to-frame stability.
- [ ] C07: Quantify numerical differences.
- [ ] C08: Identify dominant error sources.

#### Validation Gate

- [ ] Comparable input sequences are available.
- [ ] Reference and experimental outputs are captured.
- [ ] Image comparison methodology is documented.
- [ ] Numerical differences are measured.
- [ ] Temporal behavior is evaluated.
- [ ] Dominant visible artifacts are categorized.
- [ ] Unsupported claims of numerical equivalence are avoided.

#### Deliverables

- Image comparison report.
- Numerical error analysis.
- Temporal stability analysis.
- Known limitations.

#### Boundary

Do not optimize kernels during this milestone unless a correctness problem requires a targeted change.

---

### M15 — Native INT8 Performance Optimization

**Status:** ☐ proposed

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

- [ ] C01: Profile the complete inference pipeline.
- [ ] C02: Identify the highest-cost operations.
- [ ] C03: Establish per-layer timing.
- [ ] C04: Measure memory and register behavior.
- [ ] C05: Select one measurable bottleneck.
- [ ] C06: Implement one optimization.
- [ ] C07: Re-run correctness tests.
- [ ] C08: Measure the performance difference.
- [ ] C09: Compare INT8 and FP16 end-to-end costs.
- [ ] C10: Record unsuccessful optimizations.

#### Validation Gate

- [ ] Baseline performance is reproducible.
- [ ] Dominant bottlenecks are identified with evidence.
- [ ] Each optimization has before/after measurements.
- [ ] Numerical correctness is preserved within documented tolerances.
- [ ] Memory and conversion overheads are accounted for.
- [ ] GPU execution and transfer times are distinguished.
- [ ] INT8 and FP16 are compared using equivalent workloads.
- [ ] Results are reported without selective benchmarking.

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
