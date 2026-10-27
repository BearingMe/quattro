# Findings

Discoveries that influence engineering decisions. One entry per finding: observation, evidence, confidence, engineering implication. Tag classes per [docs/CONVENTIONS.md](../docs/CONVENTIONS.md).

---

## F1 — FSR 4 INT8 convolutions run on the standard HLSL `dot4add_i8packed` intrinsic

- **Observation:** The INT8 (and parts of the FP8-variant) generated passes execute packed INT8 dot products via the SM 6.4 intrinsic `dot4add_i8packed`, gated by `DOT4_ENABLED`, accumulating into per-lane accumulators.
- **Evidence:** [S] FSR-4.0.2-reference @ `c5e34b4` — `Kits/FidelityFX/upscalers/fsr4/include/gpu/fsr4/post_common.hlsli` L76–218; `DOT4_ENABLED` in `fsr4_model_v07_fp8_no_scale_passes_1080.hlsl` L10. Corroborated by [S-lhl] audit (passes 1–13 dominated by `dot4add_i8packed`).
- **Confidence:** High (direct source reading).
- **Implication:** Our M06 INT8 baseline (`dot4add_i8packed` + INT32 accumulation) is faithful to AMD's own RDNA-era approach — no exotic API required for the workhorse path. WMMA is the *FP8/RDNA4* path, plus a candidate optimization for us (M06).

## F2 — AMD reached WMMA from D3D12 HLSL via driver intrinsics extension, before SM 6.9

- **Observation:** `pre_wmma.hlsl`/`post_wmma.hlsl` use `AmdWaveMatrixA/B/Accumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_{F16,FP8,F32},16,16>` + `AmdWaveMatrixMultiply` — AMD's HLSL shader-intrinsics extension.
- **Evidence:** [S] FSR-4.0.2-reference @ `c5e34b4`, `internal/shaders/post_wmma.hlsl` L90–194, `pre_wmma.hlsl` L212–238.
- **Confidence:** High (code exists); medium for runtime behavior (never executed by us yet).
- **Implication:** Two candidate WMMA routes on gfx1101: (a) SM 6.9 Cooperative Vector / DX Linear Algebra (sanctioned, driver support [U]), (b) AMD extension intrinsics (what FSR4 shipped). M06 must probe both before choosing.

## F3 — The FSR 4 v07 model is tiny and pass-split

- **Observation:** Weights ≈ 87.1 KB (INT8) / 127.0 KB (FP8) per quality tier; ~14 model + 13 post dispatches @1080p; ~20 MB scratch; NHWC, 7 input channels.
- **Evidence:** [S] directory listing of all 12 `initializers.bin` files @ `c5e34b4`; [S-lhl] lhl repo @ `49015b7` (pass counts, scratch, channels).
- **Confidence:** High (weights measured locally; pass counts from lhl audit, re-verify in EXP-002).
- **Implication:** Execution is likely dispatch/memory-overhead-sensitive, not FLOP-bound → our harness must measure launch + memory movement early (M05), and pass fusion is a later optimization, not a first-order concern.

## F4 — Public FidelityFX SDK 2.3.0 does not contain FSR 4 source

- **Observation:** Glob for `*fsr4*` in SDK @ `60f4ea8` (tag v2.3.0) finds only docs/media images; `Kits/FidelityFX/upscalers/` holds `fsr3` + `include` only.
- **Evidence:** [S] local inspection.
- **Confidence:** High.
- **Implication:** The MIT 4.0.2 mirror is the only source-level FSR 4 reference; FSR 4.1 RDNA 3 details require the companion RE project or black-box observation (M04). Keep SDK submodule for D3D12/FSR2/3 conventions only.

## F5 — gfx1101 is the correct AMD ISA target for this card

- **Observation:** RX 7700 XT = gfx1101, 54 CU, wave32/64, LDS 128 KiB/WGP-class.
- **Evidence:** [D] AMD ROCm gpu-arch-specs; [M] local device IDs match (see [environment baseline](findings/2026-10-27-environment-baseline.md)).
- **Confidence:** High.
- **Implication:** All ISA inspection, RGA targets, and HIP comparisons must target `gfx1101`; ignore gfx1100/gfx1151-specific tuning claims.

## F6 — Quantization policy matters more than instruction choice (harness evidence)

- **Observation:** Per-iteration requantization regressed INT8 +194% / FP8 +476%; scalar INT8 I/O beat packed dot by ~32%; loop unrolling +12% (gfx1151 microkernels).
- **Evidence:** [M] lhl/fsr4-rdna3-optimization @ `49015b7`, measured in their HIP harness (cv < 1% protocol).
- **Confidence:** Medium-high for their harness on gfx1151; **unproven on gfx1101 and in real DXC-compiled FSR-style HLSL**.
- **Implication:** Design rule candidate for our pipeline: accumulate in INT32/FP32, quantize once at store. Re-validate scalar-vs-packed on gfx1101 (M07) before adopting.

## F7 — Sanctioned HLSL WMMA path is arriving (SM 6.9 CoopVec → DX Linear Algebra)

- **Observation:** Microsoft/AMD document Cooperative Vector (SM 6.9) and DirectX Linear Algebra as HLSL access to AMD WMMA cores.
- **Evidence:** [D] devblogs.microsoft.com (SM 6.9 retail 2026-02; LinAlg 2026-03), gpuopen.com GDC 2026 post.
- **Confidence:** High that the path exists; [U] whether our driver/gfx1101 exposes it (SDK dxc 1.8.2502 also too old to compile SM 6.9).
- **Implication:** M06 requires: newer standalone DXC + Agility SDK + `CheckFeatureSupport` probe; keep DP4A path as the portable fallback.
