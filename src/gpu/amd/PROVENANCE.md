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
