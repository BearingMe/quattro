// core — numerical conversions and quantization primitives (M08).
// Dependency rule: core depends on nothing project-local and never touches
// GPU APIs (docs/architecture.md §3).

#pragma once

#include <cstdint>

namespace core {

// --- FP16 (IEEE 754 binary16) ---

// Decode: exact for all 65536 encodings, including denormals and NaN
// payloads (NaN payloads preserve the low bits; sign preserved).
float F16ToF32(uint16_t bits);

// Encode: round-to-nearest-even for normals and denormals. Overflow
// saturates to infinity; ±inf INPUTS also encode to the largest finite
// magnitude (saturation policy — this is why E5M2's inf encodings are
// excluded from the round-trip test); NaN propagates as qNaN (0xFE00)
// with the payload dropped. Exact on the full binary16 grid; exhaustively
// round-trip tested (all 65536 encodings, M08 experiment).
uint16_t F32ToF16(float value);

// --- FP8 (OCP OFP8 Rev 1.0) ---
//
// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. NO infinity: E'=15 with
// mantissa 7 = NaN (0x7F/0xFF only), max finite = 448. Denormals =
// m/8 * 2^-6.
// E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa. IEEE-style inf (E'=31,
// m=0) and NaN (E'=31, m!=0). Max finite = 57344. Denormals = m/4 * 2^-14.
//
// Decode is exact for every encoding. Encode rounds to nearest-even and
// SATURATES on overflow to the format's max finite value — this matches
// AMD's saturating packs ("CopySat", `amd_pack_sat_fp8`) rather than
// producing inf.

float E4M3ToF32(uint8_t bits);
float E5M2ToF32(uint8_t bits);

// Returns the nearest representable value with round-to-nearest-even ties.
// Saturates beyond the max finite magnitude. NaN in -> NaN payload
// (0x7F for both formats, input sign preserved).
uint8_t F32ToE4M3(float value);
uint8_t F32ToE5M2(float value);

// --- Symmetric INT8 quantization (the AMD/FSR 4 v07 contract, M08) ---
//
// quantize:   q = clamp_s8(round(v * (1/scale)))   — scale is the DEQUANTIZE
//             multiplier; round is round-half-to-even (HLSL `round()`);
//             clamp is the saturating pack range [-128, 127]
//             (`pack_clamp_s8`), zero point fixed at 0.
// dequantize: v = q * scale (the consumer idiom `unpack_s8s16(q) * scale`).

enum class QuantizeRounding {
    RoundNearestEven,  // HLSL round() semantics — the AMD contract
    TowardZero,        // HLSL int4 cast semantics (present in the fp32
                       // QuantizeLinear overload; characterized, not the
                       // v07 activation path)
};

// `scale` must be finite and > 0; NaN/Inf/zero scale returns 0 (caller
// validates — see the experiment's metadata checks).
int8_t QuantizeSymmetricS8(float value, float scale, QuantizeRounding rounding);
float DequantizeSymmetricS8(int8_t q, float scale);

} // namespace core
