# AMD AmdExtD3D headers — vendored provenance

Vendored unmodified from the pinned mirror
`.references/sources/FSR-4.0.2-reference` @ `c5e34b4`
(`Kits/FidelityFX/api/internal/dx12/AmdExtD3D/`) on 2026-10-27 for M06.

Files:
- `AmdExtD3D.h` — `IAmdExtD3DFactory` + `AmdExtD3DCreateInterface` typedef.
- `AmdExtD3DShaderIntrinsics.h` — opcode constants, magic-code encoding, WMMA opcodes (incl. `WMMA_I32_16X16X16_I8` = 0xc).
- `AmdExtD3DShaderIntrinsicsApi.h` — `IAmdExtD3DShaderIntrinsics{,1}` runtime interface.
- `AmdExtD3DShaderAnalyzerApi.h` — `IAmdExtD3DShaderAnalyzer{,1,2,3}` (ISA disassembly on live device).
- `AmdExtD3DShaderIntrinsics.hlsl`, `AmdExtD3DShaderIntrinsicsMatrixOps.hlsl`, `AmdExtD3DShaderIntrinsicsMatrixOpsHelper.hlsl` — HLSL-side intrinsics implementation.

Reason for vendoring: self-contained build (no shader-include dependency on
submodule state); source of truth recorded above; do not edit — upstream
changes go through re-vendoring with a new provenance note. AMD copyright
headers preserved (permissive FidelityFX license as distributed in the mirror).

## M08 addition — ml2code_runtime (2026-10-27)

Vendored unmodified from the same pinned mirror `c5e34b4`
(`Kits/FidelityFX/upscalers/fsr4/dx12/ml2code_runtime/`) for M08:

- `config.hlsli`, `globals.hlsli`, `storage.hlsli` — tensor struct templates
  (`ML2C_DECL_TENSOR` / `ML2C_DECL_QUANTIZED_TENSOR`), `ComputeShaderParams`,
  poison/alignment macros, storage adapters.
- `tensor_int8.hlsli`, `tensor_float16.hlsli`, `tensor_float32.hlsli` —
  byte-addressed load/store helpers (`Load4hA`, `Store4i8A`, …).
- `operators/int8_NHWC/QuantizeLinear.hlsli` — AMD's INT8 quantization
  operator (the C09 reference: `pack_clamp_s8(int4(round(v * rcpScale)))`).

Reason: the M08 `quantization` experiment executes AMD's shipped operator
code on gfx1101 to validate the CPU quantization contract against the real
implementation, exactly as the generated FSR 4 passes call it. Include root
is `src/gpu/amd/` (shaders include `"ml2code_runtime/..."`). Probe compile
flags match the shipped model header: `-O3 -enable-16bit-types -HV 2021`.
