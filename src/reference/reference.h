// reference — CPU implementations for GPU validation. Must never touch GPU
// APIs (docs/architecture.md §3).

#pragma once

#include <cstddef>
#include <cstdint>

// Numerical conversions live in core (architecture.md §3); reference
// re-exports them for experiment call sites.
#include "core/numerics.h"

namespace reference {

// Independent CPU reference for the M05 harness workload. Bit-exactness is
// expected: both sides perform the same fp32 addition per element in the same
// order — no reassociation anywhere.
void CpuVectorAdd(const float* a, const float* b, float* out, size_t count);

// Returns the number of elements where `gpu` and `cpu` differ bit-wise.
size_t CountMismatches(const float* gpu, const float* cpu, size_t count);

// M06 — 16x16x16 INT8 matrix multiply, C = A*B + initC, INT32 accumulation.
// Operand signedness per mode (matches the WMMA opcode table):
//   0 = i8*i8, 1 = u8*u8, 2 = i8(a)*u8(b), 3 = u8(a)*i8(b).
// `a`/`b` are 256-byte row-major tiles; `initC`/`out` are 16x16 int32.
void CpuInt8Matmul16x16x16(const uint8_t* a, const uint8_t* b, const int32_t* initC,
                           int32_t* out, uint32_t mode);

// Bit-wise mismatch count for element counts of int32 outputs.
size_t CountI32Mismatches(const int32_t* gpu, const int32_t* cpu, size_t count);

// FP16 conversions moved to core (numerics.h, included above). The M08 work
// fixed two latent defects in the versions this header used to declare:
// F16ToF32 returned half the correct value for denormal encodings, and
// F32ToF16 used approximate denormal rounding. Both are now exhaustively
// round-trip tested (all 65536 encodings) as part of the M08 gate.

// M07 — 16x16x16 FP16 matrix multiply, C = A*B + initC, accumulation in
// `accBitsPerElement` (16 = fp16 accumulator, 32 = fp32 accumulator).
// `a`/`b` are row-major fp16 tiles (512 bytes), `initC`/`out` match the
// accumulator width (512 bytes fp16 / 1024 bytes fp32).
// Note: the hardware intra-instruction accumulation order is not documented
// ([U]); the reference sums sequentially in fp32 (or fp16) and callers compare
// with a documented tolerance, not bit-exactness.
void CpuF16Matmul16x16x16(const uint16_t* a, const uint16_t* b, const void* initC,
                          void* out, uint32_t accBitsPerElement);

// Largest absolute element-wise difference |gpu - cpu|.
float MaxAbsDiffF32(const float* gpu, const float* cpu, size_t count);
float MaxAbsDiffF16(const uint16_t* gpu, const uint16_t* cpu, size_t count);

// --- M08: independent FP8 oracles ---
//
// The oracle deliberately uses different algorithms than core::numerics:
// tables built by separately-written ldexp composition, and a hand-rolled
// floor/frac rounding (not nearbyint). Purpose: catch implementation slips,
// not to be faster or cleaner.

// Decodes all 256 encodings of the format into `out` (index = encoding).
// Independent construction: sign × (mantissa/exp arithmetic with ldexp).
void BuildE4M3Table(float* out256);
void BuildE5M2Table(float* out256);

// Nearest-representable encode by linear scan over the full decode table
// with an explicit round-to-nearest-even tie rule on the mantissa grid.
// Overflow saturates to max finite (matches core's documented policy).
uint8_t ReferenceEncodeE4M3(float value);
uint8_t ReferenceEncodeE5M2(float value);

// Symmetric INT8 quantize oracle with the AMD arithmetic contract
// (reciprocal-then-multiply) and hand-rolled ties-to-even rounding.
int8_t ReferenceQuantizeS8(float value, float scale, bool roundNearestEven);
float ReferenceDequantizeS8(int8_t q, float scale);

// --- M08: quantization error metrics ---
struct QuantErrorStats {
    double maxAbsErr = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double sqnrDb = 0.0;    // 10*log10(signalPower / noisePower)
    double clipFrac = 0.0;  // fraction of quantized values at ±128/±127 clamp
};

// `dequant[i]` = DequantizeSymmetricS8(q[i], scale); `ref[i]` = original.
// `signalPower` = mean(ref^2). clipFrac counts |q| >= 127 entries.
QuantErrorStats ComputeQuantError(const float* ref, const int8_t* q,
                                  const float* dequant, size_t count);

} // namespace reference
