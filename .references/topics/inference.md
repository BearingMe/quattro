# Inference execution (ML2Code, generated HLSL, D3D12)

Status: starting document. Tags per [docs/CONVENTIONS.md](../../docs/CONVENTIONS.md). Access dates 2026-10-27.

## Established facts

- [S] FSR 4 models are compiled **offline** (ONNX → HLSL) via AMD's **ML2Code** toolchain; the runtime side lives in the archive:
  - `Kits/FidelityFX/upscalers/fsr4/dx12/ml2code_runtime/` (runtime)
  - `Kits/FidelityFX/upscalers/fsr4/dx12/ffx_provider_fsr4_dx12.cpp` (D3D12 provider / dispatch contract)
  Source: mirror @ `c5e34b4` (paths from its README, verified present in local tree).
- [S] Generated compute shaders are resolution-specialized (`passes_1080/2160/4320.hlsl`) and variant-specialized (i8/fp8 × quality tier); INT8 passes define `DOT4_ENABLED 1` and use `dot4add_i8packed` in fused convolution operators; FP8 execution is WMMA-based via AMD intrinsics extension.
  Source: pinned `c5e34b4`, `include/gpu/fsr4/post_common.hlsli`, `internal/shaders/*` (see [wmma.md](wmma.md), [quantization.md](quantization.md)).
- [S-lhl] Generated code characteristics (from their HLSL audit): NHWC 3D/4D tensors, nested `[unroll]` loops over (kx, ky, input channels, output features in groups of 4), fast paths for common shapes (8-channel input, 16-feature output), store-time quantization (`round(vs * rcpScale)` in 4.1-era source), mixed thread groups (`32x1x1`, `64x1x1`, `8x8x1`).
  Source: lhl repo @ `49015b7` README — mark as audit-derived; re-verify against 4.0.2 archive locally.
- [D] The public SDK's D3D12 infrastructure (Cauldron samples, FSR 2/3 techniques) documents resource conventions for GPU work in this ecosystem. Source: FidelityFX-SDK submodule @ `60f4ea8`.

## Relevant implementation details

- [S] No Vulkan/CUDA anywhere in the FSR 4 D3D12 path — INT8 uses standard HLSL intrinsics; FP8/WMMA uses `AmdExtD3DShaderIntrinsics*` (driver extension loaded via the AGS-style intrinsic mechanism).
- [U] How `initializers.bin` is bound (raw buffer? typed?), and how scales are applied per layer — read the provider + runtime next.

## Unknowns

- [U] Dispatch graph construction order, barrier strategy, and resource lifetimes in `ffx_provider_fsr4_dx12.cpp`.
- [U] Whether ML2Code emits anything WMMA-related for the **INT8** variants (or INT8 is dot4-only in 4.0.2).
- [U] ML2Code toolchain availability (is the generator itself public? presumably not) — if not, our reconstruction compiles its own HLSL.

## Questions requiring experiments

1. Read `ffx_provider_fsr4_dx12.cpp` + `ml2code_runtime/` end-to-end; write a one-page execution-graph summary (tensor → pass → dispatch → barrier) (M6 groundwork).
2. Extract per-pass constant-buffer layouts from generated HLSL (feeds EXP-002 schema decode).

## Sources

Inline; pinned submodules per [README](../README.md).
