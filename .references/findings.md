# Findings

Discoveries that influence engineering decisions. One entry per finding: observation, evidence, confidence, engineering implication. Tag classes per [docs/CONVENTIONS.md](../docs/CONVENTIONS.md).

---

## F1 — FSR 4 INT8 convolutions run on the standard HLSL `dot4add_i8packed` intrinsic

- **Observation:** The INT8 (and parts of the FP8-variant) generated passes execute packed INT8 dot products via the SM 6.4 intrinsic `dot4add_i8packed`, gated by `DOT4_ENABLED`, accumulating into per-lane accumulators.
- **Evidence:** [S] FSR-4.0.2-reference @ `c5e34b4` — `Kits/FidelityFX/upscalers/fsr4/include/gpu/fsr4/post_common.hlsli` L76–218; `DOT4_ENABLED` in `fsr4_model_v07_fp8_no_scale_passes_1080.hlsl` L10. Corroborated by [S-lhl] audit (passes 1–13 dominated by `dot4add_i8packed`).
- **Confidence:** High (direct source reading).
- **Implication:** Our M06 INT8 baseline (`dot4add_i8packed` + INT32 accumulation) is faithful to AMD's own RDNA-era approach — no exotic API required for the workhorse path. WMMA is the *FP8/RDNA4* path, plus a candidate optimization for us (M06).

## F2 — AMD reached WMMA from D3D12 HLSL via driver intrinsics extension, before SM 6.9

- **Observation:** `pre_wmma.hlsl`/`post_wmma.hlsl` use `AmdWaveMatrixA/B/Accumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_{F16,FP8,F32},16,16>` + `AmdWaveMatrixMultiply` — AMD's HLSL shader-intrinsics extension. M06 proved the route end-to-end on gfx1101: `CheckSupport(WaveMatrix)` = S_OK on driver 32.0.32015.7029 via `amdxc64.dll`; INT8/U8 operand formats + I32 accumulator exist (`WMMA_I32_16X16X16_{I8,U8,IU8,UI8}`); DXC 1.8.2502 compiles at `cs_6_6 -enable-16bit-types`; the driver analyzer disassembles live pipelines to `v_wmma_i32_16x16x16_iu8` (Wave32, gfx11), with signedness carried by `neg_lo` operand modifiers; results bit-exact vs CPU INT32 across all four signedness modes (EXP-004).
- **Evidence:** [S] FSR-4.0.2-reference @ `c5e34b4` — `Kits/FidelityFX/upscalers/fsr4/internal/shaders/post_wmma.hlsl` L90–194, `pre_wmma.hlsl` L212–238, `Kits/FidelityFX/api/internal/dx12/AmdExtD3D/*` headers (vendored, `src/gpu/amd/`); [M] EXP-004 + `build/reports/m06/*.isa`.
- **Confidence:** High (executed and ISA-verified on gfx1101).
- **Implication:** The WMMA route for M07+ is settled: D3D12/HLSL + AMD intrinsics, no HIP, no toolchain upgrade. SM 6.9 CoopVec remains unprobed (needs newer DXC + Agility SDK) — future option, not needed for the objective.

## F3 — The FSR 4 v07 model is tiny and pass-split

- **Observation:** Weights ≈ 87.1 KB (INT8) / 127.0 KB (FP8) per quality tier; ~14 model + 13 post dispatches @1080p; ~20 MB scratch; NHWC, 7 input channels.
- **Evidence:** [S] directory listing of all 12 `initializers.bin` files @ `c5e34b4`; [S-lhl] lhl repo @ `49015b7` (pass counts, scratch, channels).
- **Confidence:** High (weights measured locally; pass counts from the lhl audit, verified against the provider's dispatch code — see the [M04 architecture report](../docs/findings/2026-10-27-fsr4-architecture.md) §3.1).
- **Implication:** Execution is likely dispatch/memory-overhead-sensitive, not FLOP-bound → our harness must measure launch + memory movement early (M05), and pass fusion is a later optimization, not a first-order concern.

## F4 — Public FidelityFX SDK 2.3.0 does not contain FSR 4 source

- **Observation:** Glob for `*fsr4*` in SDK @ `60f4ea8` (tag v2.3.0) finds only docs/media images; `Kits/FidelityFX/upscalers/` holds `fsr3` + `include` only.
- **Evidence:** [S] local inspection.
- **Confidence:** High.
- **Implication:** The MIT 4.0.2 mirror is the only source-level FSR 4 reference; FSR 4.1 RDNA 3 details require the companion RE project or black-box observation (M04). Keep SDK submodule for D3D12/FSR2/3 conventions only.
- **Update (2026-10-27, EXP-002):** The same SDK ships the closed FSR 4 runtime as *signed binaries* in `Kits/FidelityFX/signedbin/` (`amd_fidelityfx_upscaler_dx12.dll` 4.1.1.2740, loader 2.3.0.2740) — a legitimate 4.1 artifact was obtainable after all, and the official prebuilt sample ran it on gfx1101 (see F8). The 4.0.2 mirror remains source-only reference.

## F5 — gfx1101 is the correct AMD ISA target for this card

- **Observation:** RX 7700 XT = gfx1101, 54 CU, wave32/64, LDS 128 KiB/WGP-class.
- **Evidence:** [D] AMD ROCm gpu-arch-specs; [M] local device IDs match (see [environment baseline](findings/2026-10-27-environment-baseline.md)).
- **Confidence:** High.
- **Implication:** All ISA inspection, RGA targets, and HIP comparisons must target `gfx1101`; ignore gfx1100/gfx1151-specific tuning claims.

## F6 — Quantization policy matters more than instruction choice (harness evidence)

- **Observation:** Per-iteration requantization regressed INT8 +194% / FP8 +476%; scalar INT8 I/O beat packed dot by ~32%; loop unrolling +12% (gfx1151 microkernels). On gfx1101 (M07, EXP-005): driver-generated WMMA chains are **overhead-bound** (2.4–3.1× time for 4× K) while DP4A is compute-bound-ish (4.9–5.2×) — DP4A wins **4.2–4.7× at 16×16×64** but only **2.2–2.7× at 16×16×256** (the WMMA prologue amortizes with K); the per-tile fragment-gather (buffer loads + `ds_bpermute`) dominates the WMMA path unless operands are LDS-staged (FSR4's own `pre_wmma.hlsl` pattern).
- **Evidence:** [M] lhl/fsr4-rdna3-optimization @ `49015b7` (gfx1151, cv < 1% protocol); [M] EXP-005 on gfx1101 (100 timed dispatches × 2 configs × 2 K values, ISA-verified kernels).
- **Confidence:** High for gfx1101 at these shapes; the gfx1151 scalar-vs-packed claim remains unproven here.
- **Implication:** Design rule for our pipeline: accumulate INT32/FP32, quantize once at store; stage WMMA operands through LDS and reuse tiles before WMMA beats DP4A (the crossover is shape-dependent — measure, don't assume). Re-validate scalar-vs-packed on gfx1101 in M09/M15. FP16-vs-INT8 WMMA: no meaningful timing difference at these shapes (register cost differs: 83 vs 52 VGPRs in the GEMM kernels).

## F7 — Sanctioned HLSL WMMA path is arriving (SM 6.9 CoopVec → DX Linear Algebra)

- **Observation:** Microsoft/AMD document Cooperative Vector (SM 6.9) and DirectX Linear Algebra as HLSL access to AMD WMMA cores.
- **Evidence:** [D] devblogs.microsoft.com (SM 6.9 retail 2026-02; LinAlg 2026-03), gpuopen.com GDC 2026 post.
- **Confidence:** High that the path exists; [U] whether our driver/gfx1101 exposes it (SDK dxc 1.8.2502 also too old to compile SM 6.9).
- **Implication:** M06 requires: newer standalone DXC + Agility SDK + `CheckFeatureSupport` probe; keep DP4A path as the portable fallback.

## F8 � The official FSR 4.1 runtime runs unmodified on gfx1101: ~1.36 ms at Quality 720p?1080p (M03 oracle)

- **Observation:** FSR SDK 2.3.0's signed `amd_fidelityfx_upscaler_dx12.dll` (4.1.1.2740) enumerates upscaler versions **4.1.2\* (default), 4.1.1, 3.1.5, 2.3.4** and executes its default **FSR 4.1.2 ML upscaler** on the RX 7700 XT via the official prebuilt `FidelityFX_FSR` sample: 1280�707 ? 1920�1061, Quality preset (qualityMode 1), **avg 1.3645 / 1.3528 ms** across two 600-frame runs (? � 0.9%). Captured frames valid (sharp Sponza, readable inscriptions).
- **Evidence:** [M] EXP-002 (`docs/experiments/EXP-002-fsr4-baseline.md`), outputs archived in `build/captures/m03/`; version enumeration in `Cauldron.log`; `GPU`/`DriverVersion` recorded in benchmark JSONs.
- **Confidence:** High (two independent runs, in-band device identification, official binaries unmodified).
- **Implication:** (1) M03 oracle established � no 4.0.x substitute needed for the baseline; (2) whole-frame `GPUTime` label in Cauldron benchmark JSONs is broken (4e-05 ms) � use per-label timings; (3) sample exits 0xC0000005 at teardown after writing outputs (quirk, not a measurement issue); (4) 4.1.1 is UI-selectable for exact-version A/B in later milestones.

## F9 — AMD/FSR4 INT8 quantization contract measured on hardware (M08)

- **Observation:** The shipped `QuantizeLinear` operator on gfx1101 matches an independent CPU implementation bit-exactly at 4 scales (incl. the real v07 downscale scale 0.0148847): `q = pack_clamp_s8(int4(round(v * (1/scale))))` — round = ties-to-even (halfway cases confirmed on hw), clamp [-128, 127] (not ±127), zero point structurally 0 (operator params unnamed/ignored), per-tensor scale = dequantize multiplier baked into tensor metadata / generated-pass constants. Special values: NaN→0, +inf→127, −inf→−128, −0→0 — reproducible and position-independent. FP8 in v07 is activation storage via WMMA `CopySat` (scale 1.0); the FP8 and INT8 networks never meet — no FP8→INT8 conversion exists in shipped code.
- **Evidence:** [S] FSR-4.0.2-reference @ `c5e34b4` — `ml2code_runtime/operators/int8_NHWC/QuantizeLinear.hlsli` (vendored), `Add.hlsli` dequant idiom, `float8_NHWC/Conv2D_k2s2b.hlsli` CopySat, generated fp8 pass (baked scales); [M] EXP-006 (`docs/experiments/EXP-006-quantization.md`).
- **Confidence:** High (bit-exact on hardware incl. special values; exhaustive CPU-side oracles).
- **Implication:** (1) M09 kernels integrate scales exactly as `round(acc * rcpScale)` at store; (2) per-channel scaling is NOT part of the AMD contract — treat as our generic capability only; (3) NaN input to quantize cannot poison tensors (→0), but accumulate-path INT32 overflow remains unchecked by the operators — M09 must bound or detect it; (4) vendoring-and-executing AMD operator code standalone works — reuse for M09 Conv2D evidence.
