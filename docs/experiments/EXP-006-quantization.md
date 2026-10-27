# EXP-006: What exactly are AMD's FP8/INT8 quantization semantics, and does our CPU contract match the shipped operator on hardware?

Date: 2026-10-27
Milestone: M08

## Question

What are the precise FP8 (E4M3/E5M2) and symmetric-INT8 quantization semantics relevant to FSR 4's implementation — rounding, saturation, scale placement, zero point — and does an independent CPU implementation of that contract match AMD's shipped `QuantizeLinear` operator executing on gfx1101?

## Hypothesis

From the M04 architecture report (§5): FSR 4 v07 INT8 uses per-tensor symmetric scales with zero point 0 and quantize-once-at-store (`round(v * rcpScale)`); the FP8 variant ships with identity scale. The CPU contract built on those facts should match AMD's shipped operator bit-exactly on the finite domain.

## Reference sources

- [S] FSR-4.0.2-reference @ `c5e34b4` — `ml2code_runtime/operators/int8_NHWC/QuantizeLinear.hlsli` (the operator under test, vendored to `src/gpu/amd/ml2code_runtime/`), `int8_NHWC/Add.hlsli` (consumer dequant idiom `unpack_s8s16(q) * scale`), `float8_NHWC/Conv2D_k2s2b.hlsli` (FP8 = activation storage via WMMA `CopySat`, `amd_pack_sat_fp8`, `#error There is no FP8 without WMMA`), `internal/shaders/fsr4_model_v07_fp8_no_scale_passes_1080.hlsl` (scales baked as shader constants, `quantizationScale = 1.0`, weights fp16 `Tensor4h_HNWC`), `pre.hlsl` (real scale 0.0148847…).
- [D] OCP OFP8 spec Rev 1.0: E4M3 (bias 7, no inf, NaN at E15|M7, max 448) / E5M2 (bias 15, IEEE inf/NaN, max 57344).
- [D] HLSL SM 6.6: `pack_clamp_s8` (saturating pack, [-128, 127]), `round()` (round-half-to-even), `-HV 2021` compile contract from the shipped pass header.

## Method

1. **core primitives** (`src/core/numerics.{h,cpp}` — first `core` module code, dependency-clean): FP8 E4M3/E5M2 decode (exact, all 256 encodings) and encode (RNE ties-to-even, saturating — matching AMD's `CopySat`/`amd_pack_sat_fp8` "sat" semantics); symmetric INT8 `QuantizeSymmetricS8` replicating AMD's arithmetic exactly (reciprocal-then-multiply, `round()` ties-to-even, saturating clamp [-128, 127], zero point 0) plus a `TowardZero` variant (the fp32-input overload's `int4()` cast — present in the source, characterized); `DequantizeSymmetricS8`. FP16 conversions moved here from `reference` with two latent defects fixed: `F16ToF32` returned **half the correct value** for denormal encodings (wrong exponent constant 127−15−e instead of 113−e), and `F32ToF16` used approximate denormal rounding. Both are now exhaustively round-trip tested.
2. **Independent oracles** (`src/reference/reference.{h,cpp}`): FP8 tables built by a separately-written ldexp composition (`BuildF8Table`), encode-by-linear-scan with explicit even-mantissa tie rule (deliberately different algorithms than core's direct bit construction); INT8 oracle with hand-rolled floor/frac ties-to-even (independent of `nearbyint`); `ComputeQuantError` (max abs, MAE, RMSE, SQNR, clip fraction).
3. **Experiment** (`rdna3-int8.exe run quantization`):
   - Part A (C01/C02, CPU): exhaustive 256-encoding decode vs oracle, encode(decode(b)) round-trip, sign symmetry, monotonicity, hand-derived spec spot values; 60000-value encode sweeps per format vs the scan oracle.
   - Part B (C03/C04, CPU): INT8 quantize vs oracle over 4 scales × 20000 values + boundary set (±127/±128, halfway, saturators); dequant round-trip bound |e| ≤ scale/2.
   - Part C (C05, CPU): per-tensor vs per-channel scaling on outlier gaussian weights (8 ch × 512, one 20× channel).
   - Part D (C06/C07/C08, CPU): error table over 6 distributions (gaussian σ∈{0.01, 1}, uniform, laplace, relu-spiked, FSR4-normalized color) × 2 scale policies (absmax, 3σ) → `build/reports/m08/distributions.csv`.
   - Part E (C09, GPU): **AMD's shipped `QuantizeLinear` operator compiled verbatim** (`-O3 -enable-16bit-types -HV 2021`, ML2Code-style baked scale, one CSO per scale: 1.0 / 2.0 / 2⁻⁷ / 0.0148847) and dispatched on gfx1101 over a 1×1×256 fp16 tensor covering grid points, exact halfway ties (power-of-two scale), saturators, fp16 denormals, ±0; bit-exact compare vs the CPU contract. Special values (qNaN, sNaN, ±inf, −0) dispatched twice (reproducibility) plus a position-rotated dispatch (position-dependence check), recorded to `build/reports/m08/special_values.txt`.

## Result

[M] **PASS on hardware — both Debug and Release configurations.** Exit 0, CTest 7/7 both presets. Driver 32.0.32015.7029 (dated 2026-10-17, recorded in the [environment baseline](../findings/2026-10-27-environment-baseline.md)).

```
fp8 e4m3: decode 0 mismatches, round-trip 0, sign-symmetry 0, NaN encodings 2/2
fp8 e5m2: decode 0 mismatches, round-trip 0 (inf excluded: saturating encode)
fp8 encode sweeps: 0 mismatches in 120000 values
fp16 exhaustive (65536 encodings): decode 0 mismatches, round-trip 0
int8 quantize vs oracle: RNE 0, truncate 0 mismatches (4 scales × 20000 + boundaries)
int8 dequant round-trip: 0 violations of |e| <= scale/2
per-tensor vs per-channel (outlier gaussian): SQNR 34.04 vs 42.28 dB (avg)
quantize_amd (AMD's shipped operator, gfx1101):
  scale 1.0         : 0/256 mismatches vs CPU contract
  scale 2.0         : 0/256 mismatches
  scale 2^-7        : 0/256 mismatches   (exact halfway ties -> RNE confirmed on hw)
  scale 0.0148847271: 0/256 mismatches   (the real FSR4 v07 downscale scale)
special values: reproducible, position-independent (ENFORCED via rotated
  dispatch), CPU-policy match ENFORCED:
  qNaN/sNaN -> 0x00; +inf -> 0x7F (127); -inf -> 0x80 (-128); -0 -> 0x00
diagnostics: no error-severity messages
```

### Findings

- [M] **C09 — the CPU contract matches AMD's shipped operator bit-exactly on gfx1101**: finite domain (grid points, halfway ties, saturators, denormals, ±0) at 4 scales including the real FSR4 v07 scale. The halfway ties confirm `round()` = round-half-to-even on this hardware; saturation confirms `pack_clamp_s8` = clamp to [-128, 127] (not ±127).
- [M] Special values are position-independent and reproducible, and also match the CPU policy: NaN → 0, +inf → 127, −inf → −128, −0 → 0. No uncontrolled NaN propagation through the quantize operator (relevant containment fact for M09+).
- [S] **Scale placement** (AMD-behavior claim, source-evidenced): per-tensor `quantizationScale` = dequantize multiplier baked into tensor metadata and, in generated passes, as a shader constant; zero point structurally fixed at 0 (the scale/zero-point operator parameters are literally unnamed and ignored in `QuantizeLinear`). Per-channel scaling does **not** appear in the v07 operator set — our per-channel results (C) are generic measurements, clearly separated.
- [S] **FP8's role in v07**: activation storage format (F32 accum → FP8 via WMMA `CopySat` hardware convert; scale = 1.0 = "no_scale" identity), NOT a weight-precision conversion of the INT8 network — the FP8 and INT8 models are separately generated networks with different operator sets (M04 §1). No FP8→INT8 conversion exists in the shipped code; milestone scope item "FP8-to-INT8 conversion" is resolved as **not applicable to FSR 4 v07** [S] — the two families never meet in shipped code.
- [M] Generic error measurements ([M], not AMD-specific): absmax scaling of unit-scale distributions gives ~42–48 dB SQNR; per-channel scaling recovers 8.2 dB on outlier-channel data (34.0 → 42.3 dB); a 3σ scale policy on unit gaussian wastes ~40 dB (scale too coarse for the grid) — absmax is the right policy for symmetric per-tensor INT8 at these distributions.
- [M] The two latent FP16 helper defects (half-value denormal decode, approximate denormal rounding) were latent because all prior probe values were fp16 normals; the M08 exhaustive round-trip test (65536 encodings) now guards them.

### Debugging trail (recorded per conventions)

1. First run: E5M2 "decode mismatches 4, round-trip 6" — the **oracle** was wrong, not core: `BuildF8Table`'s NaN condition mishandled E5M2's E31|M1/M2 encodings (treated as finite); and the round-trip check didn't exclude ±inf, which saturating encode maps to max finite **by design**. Fixed the oracle; core was right.
2. The special-value report initially showed a bizarre alternating 0x00/0x80 pattern suggesting hardware flakiness. Root cause: my report read `bytes[i*4]` — but the int8 tensor's byte address **equals** the channel index (z-stride = 1), so I was sampling channels 4i. Re-extracting bytes[i] gave clean, position-independent, CPU-matching semantics; a position-rotated dispatch confirmed and the position-independence + CPU-policy checks are enforced (mismatch fails the experiment). The GPU was right; my sampling was wrong.
3. Adversary review (pre-commit) caught that the FP16 exhaustive round-trip claim existed only as a comment — the test loop was added to Part A (65536 encodings vs independently composed decode + round-trip identity, NaN canonicalization documented) before committing.

## Interpretation

- [M] All gate items hold: FP8 decoding passes independent reference tests (exhaustive); INT8 quantization boundaries correct (±127/±128 clamps, RNE ties verified on hardware); scaling/dequantization verified (bound holds, consumer idiom cited); saturation characterized (finite + special values, reproducible, position-independent); error metrics reported (per-distribution table); quantization metadata preserved (scale-as-dequant-multiplier contract in tensor structs); AMD-specific claims separated from generic experiments (source citations vs [M] generic tables).
- [M] This also validated the vendor-and-execute approach for AMD operator code: the shipped operator compiles standalone with the documented flags and runs correctly outside the ML2Code runtime — the same technique M09 will use for Conv2D evidence.

## Next

- M09: INT8 GEMM/convolution kernels on the validated WMMA path with these quantization semantics (scale integration at store, per the `round(result * rcpScale)` requant idiom in the Fused operators).

## Files

- `src/core/numerics.{h,cpp}` (new module: FP8/FP16 conversions, INT8 quantize)
- `src/reference/reference.{h,cpp}` (independent oracles, metrics)
- `src/gpu/shaders/quantize_probe.hlsl`, `src/gpu/amd/ml2code_runtime/` (vendored operator + tensor infra, see `PROVENANCE.md`)
- `src/experiments/experiments.{h,cpp}`, `src/main.cpp`, `CMakeLists.txt` (4 CSOs, test)
- Reports: `build/reports/m08/distributions.csv`, `build/reports/m08/special_values.txt`
