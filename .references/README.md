# .references — research collection

Compact, flat reference workspace supporting the quattro project (native INT8 inference on RDNA 3 / gfx1101, case study AMD FSR 4). Claim tagging follows [`docs/CONVENTIONS.md`](../docs/CONVENTIONS.md) (`[D]` documented, `[S]` source observation, `[R]` reverse-engineering, `[M]` measurement, `[H]` hypothesis, `[U]` unknown).

```text
.references/
├── README.md        this file
├── sources/         external repositories, pinned as git submodules
├── topics/          one starting document per research topic
└── findings.md      discoveries that influence engineering decisions
```

## Source hierarchy (highest first)

1. Official AMD documentation (ISA guides, ROCm docs, GPUOpen).
2. Official AMD open-source repositories (GPUOpen, ROCm).
3. Public FSR 4 source archives.
4. Independent optimization/implementation projects.
5. Technical articles and benchmarks.
6. Reverse-engineering observations (supplementary only).

## Sources (git submodules, pinned)

Upstream code is never modified. Every repository is pinned to one commit; update pins only deliberately.

| Repository | Pin | Purpose | Relevant paths | Limitations |
|---|---|---|---|---|
| [GPUOpen-LibrariesAndSDKs/FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) | `60f4ea81909200d8542eca14dccb2628b763a9a3` (tag `v2.3.0`) | Official SDK: FSR 2/3 shader conventions, D3D12 infrastructure, Cauldron samples, technique docs | `Kits/FidelityFX/upscalers/`, `Kits/FidelityFX/include/`, `Samples/`, `Tools/`, `docs/` | ~411 MB. **No FSR4 source** — verified: v2.3.0 ships only FSR4 docs/media (`docs/techniques/media/super-resolution-ml/`); FSR 4.1 is distributed closed. AMD FidelityFX SDK EULA, not OSI-licensed. |
| [Rolaand-Jayz/FSR-4.0.2-reference](https://github.com/Rolaand-Jayz/FSR-4.0.2-reference) | `c5e34b4b7128ffeb152f80cbdb6d715d577ec289` | Archival mirror of AMD's **MIT-licensed** FidelityFX SDK 2.0.0 / FSR 4.0.2 source — the only public source view of FSR 4 internals (INT8 + FP8 models, ML2Code runtime, generated HLSL) | `Kits/FidelityFX/upscalers/fsr4/` (source), `.../fsr4/internal/shaders/` (model HLSL + weights), `.../fsr4/dx12/ml2code_runtime/`, `.../fsr4/dx12/ffx_provider_fsr4_dx12.cpp` | Community mirror; provenance is self-declared ("AMD-published MIT source") — spot-check against other archives before high-stakes claims. Reflects FSR 4.0.2 (model `v07`), **not** 4.1. Companion RE project: [RE-of-FSR-4.1.0-Upscaling](https://github.com/Rolaand-Jayz/RE-of-FSR-4.1.0-Upscaling) (not submoded yet). |
| [lhl/fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization) | `49015b72104eb19c1563db77af68ada9a6c254df` | Independent INT8/FP8 HIP microkernel benchmarks + FSR4 HLSL source audit (RDNA 3.5 `gfx1151` primary, `gfx1100` cross-check) | `README.md` (main findings), `gfx1100/`, `fsr4-src/`, `OPTIMIZATION_RESULTS.md` | No license → read-only reference, never copy code. Microkernel-level results, not full-FSR4 frame times; gfx1151 (iGPU) primary target — treat conclusions as hypotheses on gfx1101 until re-measured. |
| [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) | `2ef91896bcdc4d26624f952e5c905c787cd9bc9e` | Official AMD tool generating WMMA/SWMMAC operand layouts, encodings, throughput for RDNA 3/4 | `./` (small Python tool), docs in `README` | Encoding-mapping tool only; no runtime benchmarks. |

## Identified, deliberately not added yet

| Repository | Reason to defer |
|---|---|
| [GPUOpen-Tools/radeon_gpu_analyzer](https://github.com/GPUOpen-Tools/radeon_gpu_analyzer) (MIT, ~574 MB) | Offline HLSL→RDNA ISA compiler — the tool of choice for M2/M3 ISA verification, but large and not needed before then. Use its release binaries; add as submodule only if source study is required. |
| [ROCm/rocm-examples](https://github.com/ROCm/rocm-examples) (MIT, ~46 MB) | HIP examples — relevant only if/when a HIP-comparison milestone starts. |
| GPUOpen organization (misc.) | No additional repo with a clear active purpose beyond the four above; revisit per milestone. |

## Key external documents (no submodule needed)

- [AMD RDNA 3 ISA reference](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf) — via [AMD GPU architecture documentation hub](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html) (also RDNA 3.5/4 ISA, whitepapers, perf counters)
- [RDNA 3 dense WMMA builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html) — official builtin signatures + cycle counts for theoretical peak
- [AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation) + [machine-readable ISA specs](https://gpuopen.com/machine-readable-isa/)
- [AMD RDNA 3 MES scheduling document](https://gpuopen.com/download/documentation/micro_engine_scheduler.pdf)
- [Accelerator and GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) — official gfx1101 mapping
- [OCP 8-bit Floating Point Specification (OFP8) Rev 1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) / [Rev 1.1](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)
- DirectX SM 6.9 / Cooperative Vector / Linear Algebra: [SM 6.9 retail](https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more), [Cooperative Vector](https://devblogs.microsoft.com/directx/cooperative-vector), [Evolving DirectX for the ML era](https://devblogs.microsoft.com/directx/evolving-directx-for-the-ml-era-on-windows/), [AMD GPUOpen GDC 2026](https://gpuopen.com/learn/amd-microsoft-gdc-2026)

## Rules

1. Keep the structure flat; no indexes, databases, or tooling.
2. Submodules: pinned commits only; never modify working trees inside `sources/`.
3. Link upstream with the pinned commit hash for traceability.
4. Add a repository only when it serves an active or explicitly next milestone.
5. Summarize; do not duplicate upstream documentation here.
