# Quantization (INT8 / FP8)

Status: starting document. Tags per [docs/CONVENTIONS.md](../../docs/CONVENTIONS.md). Access dates 2026-10-27.

## Established facts

- [D] FP8 formats E4M3 (no infinities, max ±448, exponent bias 7) and E5M2 (inf/NaN, max ±57344, bias 15) are standardized in the OCP OFP8 spec.
  Source: OCP 8-bit Floating Point Specification Rev 1.0 (2023-06-20), https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf (Rev 1.1, 2026-01-29: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)
- [S] FSR 4.0.2 (model `fsr4_model_v07`) ships **two quantization families** as separate generated shader sets: `fsr4_model_v07_i8_*` (INT8) and `fsr4_model_v07_fp8_no_scale_*` (FP8 e4m3, "no per-tensor scale"), each in six quality tiers (quality, balanced, performance, ultraperf, native, drs).
  Source: `.references/sources/FSR-4.0.2-reference` @ `c5e34b4`, `Kits/FidelityFX/upscalers/fsr4/internal/shaders/`
- [S] Weight payloads are tiny: `initializers.bin` = **87.1 KB (INT8)** / **127.0 KB (FP8)** per variant tier (directory listing of all 12 files).
- [S] The INT8 path accumulates via the standard HLSL intrinsic `dot4add_i8packed` into (per-lane) accumulators, gated by `#define DOT4_ENABLED 1` in the generated passes.
  Source: pinned `c5e34b4`, `Kits/FidelityFX/upscalers/fsr4/include/gpu/fsr4/post_common.hlsli` L76–218; `internal/shaders/fsr4_model_v07_fp8_no_scale_passes_1080.hlsl` L10.
- [M] On gfx1151 (independent microkernels): per-iteration requantization is catastrophic (INT8 +194%, FP8 +476% vs quantize-once-at-store); forced per-iteration requant explains most of an apparent FP8 deficit.
  Source: lhl/fsr4-rdna3-optimization @ `49015b7`, README "Evidence Boundaries" (measured in their harness).
- [D-press] Shipping FSR 4.1 executes FP8 on RDNA 4 and INT8 on RDNA 3 (same or near-identical model).
  Source: TechSpot (2026-07-02) https://www.techspot.com/article/3142-amd-fsr-4-upscaling-rdna-3 — secondary source pending primary AMD statement.

## Important caveats

- **Do not assume AMD converts FP8 weights directly to INT8.** [U] The relationship between `v07_fp8_no_scale` and `v07_i8` weights (shared origin? separate calibration?) is unverified. Verify by decoding both `initializers.bin` payloads and comparing.
- [U] Per-tensor vs per-channel scale layout in `v07_i8` (lhl audit says "fixed-point with learned scale/bias, per-layer learned scales" for 4.1-era source — re-verify against the 4.0.2 archive locally).

## Unknowns

- [U] Binary layout of `initializers.bin` (tensor offsets, dtypes, scale placement).
- [U] Saturation/overflow behavior of the accumulation path in generated code (does any pass use saturating dot4 variants?).
- [U] Activation quantization points (which tensors are stored INT8 vs F16/F32 between passes).

## Questions requiring experiments

1. Decode `fsr4_model_v07_i8_balanced/initializers.bin` and the tensor schema; map every tensor to its consumer pass (host-side, no GPU).
2. Compare i8 vs fp8 weight payloads numerically (same tier) → evidence about FP8→INT8 relationship.
3. Propagate one random input through the INT8 pass graph on CPU using decoded weights; measure quantization error vs FP32 reference (informs error budget).

## Sources

See inline; archive pinned at `c5e34b4` (see [README](../README.md)).
