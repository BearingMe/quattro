# Architecture

How code in this repository is organized. Companion documents: [`MILESTONES.md`](MILESTONES.md) (roadmap and gates), [`CONVENTIONS.md`](CONVENTIONS.md) (evidence tagging), [`../.references/README.md`](../.references/README.md) (external sources). Last revised: 2026-10-27.

## 1. Architectural style

**Modular monolith** — one build system, one primary executable, conceptual module boundaries enforced by dependency direction, not by ceremony.

- KISS, YAGNI, composition over inheritance, early returns, explicit dependencies.
- Incremental architecture: structure grows only when a milestone requires it.
- A module exists because it owns a concrete responsibility.

Never introduce: Clean Architecture layering, plugin systems, dependency-injection containers, abstract factories, interfaces without multiple concrete consumers, microservices, generic ML frameworks, or premature backend abstractions.

## 2. Language and build policy

| Concern | Choice |
|---|---|
| Host language | **C++20** — all orchestration, numerical references, inference logic |
| GPU language | **HLSL** — DirectX 12 first; native INT8 WMMA is a research objective, not an assumption |
| HIP | Only if a milestone proves the HLSL/DXC path cannot emit the required RDNA 3 instructions (see M06 decision gate); minimal compute path, never a second frontend |
| Python | Offline only: analysis, quantization studies, model conversion, visualization; lives in `scripts/` (created when first needed); never a runtime or orchestration framework |
| Excluded | Rust, CUDA, ZLUDA, Vulkan, NVIDIA APIs |

**CMake is the single authoritative build system.** No independent build systems, per-experiment projects, custom build wrappers, or duplicate configurations. Use CMake Presets for distinct configurations.

Current status: the CMake project (root `CMakeLists.txt`, `CMakePresets.json`) exists since M01 (commit `cf56bca`); M05 added the first module code (`src/gpu/`, `src/reference/`, `src/experiments/`) and the offline DXC shader pipeline (`DXC_EXECUTABLE` cache variable → `build/bin/shaders/*.cso`).

## 3. Module responsibilities

Conceptual boundaries — **not** separate CMake libraries. Create `src/<module>/` directories only when real code lands; start cohesive, split targets only with concrete justification.

| Module | Owns | May depend on | Must never |
|---|---|---|---|
| **core** | Tensor shapes/formats, quantization parameters, FP8 decode, INT8 quantize/dequantize, numerical conversions | nothing project-local | touch GPU APIs |
| **reference** | CPU implementations for validation: matmul, convolution, quantized ops, numerical comparison | core | execute on GPU |
| **gpu** | D3D12 device, buffers, shader compilation, dispatch, sync, GPU timestamps, hardware kernels | core | know FSR 4 model semantics |
| **inference** | Network representation, layer execution, operator sequencing, weights, intermediate tensors, FSR 4 model execution | core, gpu | appear below gpu/core in the stack |
| **experiments** | Orchestration of independently runnable experiments and benchmarks | all modules | be depended on by any production module |

Model-specific behavior stays in `inference` (model code under `src/inference/models/` when it exists); general numerical primitives stay in `core`/`gpu`.

## 4. Dependency direction

```text
experiments ──► inference ──► gpu ──► core
      │                            
      └──────► reference ──────────┘
```

- `core` depends on no project module.
- `reference` → `core`; `gpu` → `core`; `inference` → `core, gpu`; `experiments` → everything.
- Nothing depends on `experiments`; tests depend only on what they validate; no cycles.
- Enforce with plain headers and direct includes — no dependency-inversion infrastructure. If a second GPU backend ever becomes real, revisit boundaries with actual requirements, not speculation.

## 5. Directory conventions

Target structure — directories are created **only** when concrete implementation needs them:

```text
.references/          external knowledge (pinned submodules; never modify upstream)
docs/                 knowledge produced here (milestones, architecture, findings, experiments)
src/
  main.cpp            single CLI entry point
  core/  reference/   modules (created when first needed)
  gpu/shaders/        HLSL sources
  inference/models/   model-specific code
  experiments/        experiment registrations/implementations
tests/                test executable sources
scripts/              offline Python/shell utilities (created when first needed)
assets/               test media and models (created when first needed)
build/                ALL generated output (git-ignored)
CMakeLists.txt / CMakePresets.json   build (introduced at M01)
AGENTS.md  README.md
```

Generated artifacts live under `build/`:

```text
build/
├── bin/        executables, compiled shaders
├── captures/   GPU captures
├── profiles/   profiling output
├── reports/    benchmark / analysis dumps
└── temp/       transient conversions and scratch data
```

These subdirectories are logical destinations — create them via build rules when first needed.

## 6. Experiment execution model

Experiments must not proliferate executables. One primary CLI:

```powershell
rdna3-int8.exe list
rdna3-int8.exe run wmma-int8
rdna3-int8.exe run wmma-fp16
rdna3-int8.exe run quantization
rdna3-int8.exe run inference
```

- Experiments are independently executable and must not call each other's implementation.
- Failures print a meaningful reason and exit nonzero; shared infrastructure is reused via modules.
- Registration is an explicit table in `main.cpp` — no dynamic loading, reflection, or registry frameworks.
- Targets start at: one experiment executable + one test executable. Anything more requires written justification in the milestone notes.

The commands above are the intended interface; experiments are implemented only when their milestone authorizes them.

## 7. Generated artifact policy

1. Generated files (executables, objects, compiled shaders, disassembly, benchmarks, captures, converted models, tensor dumps, profiling results, intermediate test files) go only under `build/`.
2. Never write generated files to the repository root; never create ad hoc `build-*`/`output/` directories.
3. Distinct configurations get named CMake presets.
4. Generated artifacts are git-ignored; large datasets stay out of version control.
5. Durable conclusions from experiments are preserved under `docs/` (see [`CONVENTIONS.md`](CONVENTIONS.md)) — not as stray dumps.
6. Prefer reproducible generation procedures (scripts + recorded parameters).
7. No permanent backup directories — git history is the backup.

## 8. Repository hygiene

1. No `backups/`, no stray root-level source files, no unmanaged generated artifacts.
2. No independent build systems, duplicate implementations, or new languages without justification.
3. No new top-level directories without justification; no speculative infrastructure or placeholder modules.
4. Never modify upstream submodule content; never refactor unrelated code during a milestone.
5. Before adding a directory, answer: What concrete responsibility does it own? Why doesn't an existing directory fit? Is separation justified by current code? Will it contain real implementation now? If any answer is weak — do not create it.
6. Prefer extending a cohesive existing module over adding a new top-level responsibility.

`.gitignore` maintains the ignore policy; check `git status` before committing. Never delete unknown untracked files, never stage indiscriminately to make the tree look clean.

## 9. Architectural evolution

| Stage | Introduce | Do not introduce |
|---|---|---|
| Early (M01/M05) | Minimal CLI, basic D3D12, tests, research docs | inference engine, model loaders, tensor frameworks, extra backends |
| Numerical (M06–M09) | core numerical types, CPU references, WMMA kernels, quantization ops | generic math library beyond needs |
| Inference (M10/M11) | network execution, required operators, model representation, weight management | general-purpose ML runtime |
| FSR 4 (M12–M15) | model logic, temporal resources, pre/post-processing, profiling extensions | new architectural layers before extending existing modules |

Extend existing modules before creating new ones; architectural refactors only when a milestone requires them, never opportunistically.

---

Note: `.references/topics/architecture.md` is a *research topic* (RDNA 3 ISA facts) — unrelated to this contract despite the name.
