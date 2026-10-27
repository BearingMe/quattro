# EXP-004: Can native RDNA 3 INT8 WMMA be reached from our D3D12/HLSL frontend on gfx1101 — and does it execute correctly?

Date: 2026-10-27
Milestone: M06

## Question

Which compute path emits and executes **native INT8 WMMA** (16×16×16 tiles, signed INT8 operands, INT32 accumulation) on the RX 7700 XT from the project's D3D12/HLSL frontend — and is it correct against an independent CPU INT32 reference, ISA-verified, and timed?

## Hypothesis

The AMD shader-intrinsics extension (`AmdWaveMatrix*`, the route FSR 4.0.2 shipped — findings F2) is supported by driver 32.0.32015.7029 on gfx1101, and DXC-compiled HLSL using it produces the native `v_wmma_i32_16x16x16_iu8` instruction with bit-exact INT8×INT8→INT32 results.

## Reference sources

- [S] FSR-4.0.2-reference @ `c5e34b4` — `AmdExtD3D/AmdExtD3DShaderIntrinsics{,Api,MatrixOps,MatrixOpsHelper}.h(lsl)` (vendored unmodified into `src/gpu/amd/`, see its `PROVENANCE.md`); usage idiom `pre_wmma.hlsl` L212–243 (`numthreads(32,1,1)`, `Load`/`AmdWaveMatrixMultiply`/`Store`).
- [D] RDNA 3 ISA reference (Feb 2023): WMMA instructions, Wave32, INT8→INT32 operand types.
- [D] amd_matrix_instruction_calculator (pinned `2ef9189`): authoritative operand-layout generator.
- [M] `amdxc64.dll` exports `AmdExtD3DCreateInterface` on this system (checked via dumpbin).

## Method

1. **Runtime contract** (`src/gpu/amdext.{h,cpp}`): load `amdxc64.dll` → `AmdExtD3DCreateInterface(device, IID_IAmdExtD3DFactory)` → `IAmdExtD3DShaderIntrinsics` → `GetInfo` + `CheckSupport(WaveMatrix)` + `Enable()`; `IAmdExtD3DShaderAnalyzer` for driver-compiler ISA disassembly of live pipelines.
2. **Probe shaders** (`src/gpu/shaders/int8_wmma.hlsl`, compiled `-T cs_6_6 -enable-16bit-types`): one 16×16×16 tile per wave (`numthreads(32,1,1)` = Wave32); `AmdWaveMatrixLoad` from a byte-address buffer (row-major, 16 B/row for A/B int8, 64 B/row for the int32 accumulator), one `AmdWaveMatrixMultiply`, `Store` back. Four entry points = four signedness modes: I8×I8, U8×U8, I8×U8, U8×I8 (opcodes `WMMA_I32_16X16X16_{I8,U8,IU8,UI8}`). Nonzero `initC` is loaded into the accumulator — accumulation semantics are part of the validation.
3. **Control** (`int8_dp4a.hlsl`): the same tile via SM 6.4 `dot4add_i8packed` (one output element per thread, 4 packed dots each) — AMD's own FSR-4 INT8 workhorse (F1), and the negative ISA control.
4. **Experiment** (`rdna3-int8.exe run wmma-int8`): M05 device (debug layer + GPU-based validation), tile data spanning the full byte range (negatives included) + nonzero initC; per probe: analyzer-created PSO → ISA disassembly saved to `build/reports/m06/<probe>.isa` → dispatch (1 validation + 100 timed with timestamp pairs) → bit-exact compare vs `reference::CpuInt8Matmul16x16x16` → info-queue verdict.

## Result

[M] **PASS on hardware — both Debug and Release configurations.** Exit 0, CTest 5/5 both presets.

```
amd extensions: intrinsics yes, analyzer yes
intrinsics: waveSize 32 (min 32, max 64), WaveMatrix support: yes
wmma i8*i8   | correct OK (0 mismatches) | wmma x 1 dot4 x 0 | min 13.720 us
wmma u8*u8   | correct OK (0 mismatches) | wmma x 1 dot4 x 0 | min 20.760 us
wmma i8*u8   | correct OK (0 mismatches) | wmma x 1 dot4 x 0 | min 20.720 us
wmma u8*i8   | correct OK (0 mismatches) | wmma x 1 dot4 x 0 | min 20.800 us
dp4a control | correct OK (0 mismatches) | wmma x 0 dot4 x 4 | min   3.240 us
diagnostics: no unexplained error-severity messages (128 known-benign GBAV
             intrinsics-dummy reports classified, see below)
```

### ISA evidence (C05/C06) — `build/reports/m06/int8_wmma_i8.isa`

The driver's shader-analyzer (same compiler as live dispatch) disassembles the WMMA probe to, in full:

```
asic(GFX11)  type(CS)  vgpr_count(24)  wave_size(32)
  ; A-tile fragment loads: buffer_load_b128 + 8x buffer_load_u8, then
  ; v_perm_b32 / ds_bpermute_b32 redistribution to WMMA operand layout
  ; C-fragment (accumulator init) loads: 8x buffer_load_b32 -> v[8:15]
  v_wmma_i32_16x16x16_iu8  v[0:7], neg_lo(v[0:3]), neg_lo(v[20:23]), v[8:15]
  ; result stores: 8x buffer_store_b32 (v[0:7] fragments)
```

- [M] The **native RDNA 3 INT8 WMMA instruction** is present and is the only matrix instruction: `v_wmma_i32_16x16x16_iu8`, Wave32 (`UC_VERSION_W32_BIT`), gfx11 encoding `CC444000`.
- [M] **Signedness maps to `neg_lo` operand modifiers**: i8×i8 = `neg_lo(A), neg_lo(B)`; u8×u8 = none; i8×u8 = `neg_lo(A)`; u8×i8 = `neg_lo(B)` (verified in all four saved ISA files). All four modes are bit-exact vs the CPU reference including negative operands and nonzero accumulate-init.
- [M] The control probe's ISA contains **zero** `wmma` and 4× `dot4` (`v_dot4` family) — the classification is not vacuous.
- [M] No `buffer_atomic` instructions remain: every magic-code atomic was patched into fragment loads/stores. The 128 GPU-based-validation reports per run ("Root descriptor access out of bounds", root param 2 = the extension's designated dummy UAV at space `0x7FFF0ADE`, addresses `0x5N0000xx` = the magic codes) are **shadow-pass false positives over the pre-patch IL**; the executed ISA never touches that UAV. Only this exact message class is classified benign (see filter in `experiments.cpp`); any other error still fails the probe.

### Timing (C07, single-tile latency — not throughput)

[M] One 16×16×16 tile per dispatch: WMMA min 13.7–20.8 µs vs DP4A 3.2 µs. For this latency-bound shape the WMMA path pays its operand-gathering prologue (`ds_bpermute` chains) with a single instruction of matrix work. This is the expected shape of the result and **not** a throughput verdict: the M07 comparison uses multi-tile workloads where the 4096 MAC/wave-instruction advantage can amortize the prologue. Both configurations measure **with the debug layer and GPU-based validation enabled** (project rule, always on) — GBAV's shadow pass adds overhead inside the timestamp pairs; clean-path baselines must be re-measured before M07 comparisons are treated as absolute.

[M] Idempotency: the probe kernels re-load all inputs per dispatch (the accumulator init comes from the input SRV, not the output tile), so the 100-dispatch timed loop's final snapshot must equal the post-warmup snapshot — verified bit-for-bit in every probe (free on-hardware determinism check).

## Compute-path decision record (C08)

**Chosen path: D3D12/HLSL + AMD shader-intrinsics extension** — the objective (native INT8 WMMA from the project's existing D3D12 frontend) is met without leaving HLSL/DXC:

- Shader: SM 6.6 + `-enable-16bit-types`, vendored `AmdExtD3D` headers (pinned provenance), `AmdWaveMatrix<I8/U8,16,16>` × `AmdWaveMatrix<I32,16,16>` accumulator, `numthreads(32,1,1)`.
- Runtime: `amdxc64.dll` (ships with the Adrenalin driver) via `AmdExtD3DCreateInterface`; `CheckSupport(WaveMatrix)` = S_OK, `Enable()` called.
- Root signature must carry the intrinsics dummy UAV at `register(u0, space 0x7FFF0ADE)` (bound to a real 16-byte buffer).
- ISA evidence path: the driver's `IAmdExtD3DShaderAnalyzer` extension — same compiler as live dispatch, on-device, automatable. RGA is **not needed** for M06+.

Limitations, recorded:
1. AMD-proprietary HLSL extension — not portable to non-AMD adapters (acceptable: gfx1101 is the fixed target; the portable DP4A fallback remains validated side-by-side).
2. GBAV emits false-positive OOB reports on the dummy UAV (classified with evidence; debug layer stays on).
3. All signedness modes lower to the single hardware `iu8` opcode + `neg_lo` modifiers.
4. FP8/SWMMA intrinsics exist in the same headers (for later milestones) but are unprobed.
5. SM 6.9 Cooperative Vector was **not** probed: it requires a newer standalone DXC + Agility SDK, and the AMD intrinsics path already satisfies the milestone objective through the sanctioned driver-documented route. Recorded as a future option, not a blocker.

## Debugging trail (recorded per conventions)

1. `AmdExtD3DCreateInterface(nullptr, ...)` → `E_POINTER`; the factory must be created **on the device** (pOuter = device).
2. DXC rejects the vendored headers at `cs_6_6` without `-enable-16bit-types` (`asuint16(half)` not viable — `half` is not `float16_t` without the flag).
3. `GetShaderIsaCode` size query crashed (0xC0000005) with `pDisassembly = nullptr`; passing a valid (empty) `AmdExtD3DPipelineDisassembly` struct fixed the two-call protocol.
4. `GetComputeShaderStats` lives on `IAmdExtD3DShaderAnalyzer1`, not the base interface.
5. `CreateCommittedResource` with temporaries (`&BufferDesc(...)`) — C++ rejects `&rvalue`; named locals.

## Interpretation

- [M] All seven M06 gate items hold on hardware: CPU-INT32-matching results (×4 signedness modes), negative operands validated, accumulation verified (nonzero initC), `v_wmma_i32_16x16x16_iu8` confirmed in driver ISA, instruction executes on gfx1101, timing reproducible (Debug/Release consistent), limitations documented (this decision record).
- [S] The path is AMD's own: FSR 4.0.2 shipped exactly these headers in `pre_wmma/post_wmma.hlsl` — our probe reproduces their mechanism from the pinned sources.

## Next

- M07: FP16 WMMA (F16→F32 accumulator opcode `WMMA_F32_16X16X16_F16`) on the same harness; multi-tile INT8-vs-FP16 throughput comparison; register/LDS stats from the analyzer.
- M09+: reuse `matmul.cpp` shapes or generalize the runner when real kernel requirements exist.

## Files

- `src/gpu/amdext.{h,cpp}`, `src/gpu/matmul.{h,cpp}`, `src/gpu/amd/` (vendored headers + `PROVENANCE.md`)
- `src/gpu/shaders/int8_wmma.hlsl`, `src/gpu/shaders/int8_dp4a.hlsl`
- `src/reference/reference.{h,cpp}` (INT8 matmul reference ×4 modes), `src/experiments/experiments.{h,cpp}`, `src/main.cpp`
- `CMakeLists.txt` (5 new CSO targets, `-enable-16bit-types`, include paths, `experiment_wmma_int8` test)
- ISA evidence: `build/reports/m06/*.isa` (generated; not committed)
