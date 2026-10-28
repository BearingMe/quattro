# FSR 4 v07 quantized-output convolutions: the compiler folds the output scale

Scope: gfx1101/RX 7700 XT, pinned v07 INT8 Quality 1080p, DXC from Windows SDK
10.0.26100.0, AMD driver **32.0.32015.7029**. Evidence:
[EXP-012](../experiments/EXP-012-fsr4-downscale.md).

## What the source says

`int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli` (`Conv_16_32`,
`Conv_32_64`, `Conv_16_48`) writes each element as

```hlsl
int16_t4 result = round((acc * weights.quantizationScale * input.quantizationScale
                         + Load4hA(bias, f)) * (1.0 / quantFactor1));
```

Three float operations after the INT32 accumulator: multiply by the weight
scale, multiply by the input scale, add the FP16 bias — then multiply by the
output-scale reciprocal, round to nearest even and saturate to INT8.

## What the shipped ISA actually does

[R] `build/reports/m12/downscale/reference-pass{3,6}.isa` show, per element, a
single `v_fmaak_f32` against an SGPR scalar and an immediate literal, followed
by `v_rndne_f32` and a narrowing clamp:

```text
v_fmaak_f32 v1, s6, v1, lit(0x3f44808b)   // scale * acc + offset
v_rndne_f32 v1, v1
v_cvt_i32_f32 / v_cvt_i32_i16 / v_med3_i32
```

No trailing multiply exists. The driver therefore folded the reciprocal
`1/qf` into **both** the scale and the bias:

```text
groupScale = fl(fl(weightScale * inputScale) / quantFactor)   // per feature half
channelOffset = fl(bias * (1 / quantFactor))
result = round(fma(acc, groupScale, channelOffset))
```

[M] Decoding the scalars confirms the derivation: pass 3's two SGPR constants
are `0x3BABC6A4` (0.005242185667157173) and `0x3B8BFE67`
(0.004272270482033491), which equal `fl(fl(sW·sIn)/qf0)` and
`fl(fl(sW·sIn)/qf1)` to the bit; the per-channel immediates equal
`fl(bias·rcp)`.

## Why this matters

[M] Substituting the source-text form into the reproduction yields
11/4,147,200 mismatching bytes (max delta 1) on the full-range fixture, all at
exact half-integer rounding boundaries. Substituting the folded form yields
**0** mismatches. The two forms are algebraically equal and numerically
different.

[S/H] The float associativity is not accidental: HLSL permits the compiler to
re-associate and contract, and at `-O3` it does. Reproducing an operator
therefore requires matching the *emitted* arithmetic, not the source
expression. This is the same lesson as
[the ConvNextBlock rounding finding](2026-10-27-fsr4-cnb-rounding.md), where
`precise`/`mad` placement decided the result; here the deciding factor is
constant folding across a reciprocal.

## Practical rules for the remaining M12 operators

1. Read the shipped pass's ISA before writing the epilogue; derive the constant
   folding instead of transcribing the C++ expression.
2. Bake compiler-folded constants explicitly (`M12_P{n}_EPI_SCALE{0,1}` in
   `build/reports/m12/weights.generated.hlsli`, produced by
   `scripts/m12_pack_model.py`) rather than re-deriving them in HLSL, so the
   result does not depend on this DXC build's folding choices.
3. Keep computing runtime terms the reference also computes at runtime — the
   per-channel `bias * rcp` offset is a single float multiply on both sides.
4. Treat any residual difference as unexplained until it is located at a
   specific instruction; never widen a tolerance to absorb it.

## Limits

- [S] Both folded group scales are derived from constants baked in the pinned
  payload; they are checksum-verified by the packer but remain an
  implementation detail of one DXC version. A different toolchain could fold
  differently, which is exactly why the values are emitted explicitly.
- [U] Whether the RDNA 4 (WMMA) path folds identically is untested; only the
  non-WMMA `navi48`-generated path in the pinned mirror was measured.
