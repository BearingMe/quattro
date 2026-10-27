// reference — CPU implementations for GPU validation. Must never touch GPU
// APIs (docs/architecture.md §3).

#pragma once

#include <cstddef>
#include <cstdint>

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

// IEEE 754 binary16 conversion. F32ToF16 rounds to nearest-even for normals;
// the denormal path uses approximate rounding (round-to-nearest, ties toward
// zero) — unexercised by the probe value ranges, which stay in [-1, 1] /
// small integers where normals are exact.
uint16_t F32ToF16(float value);
float F16ToF32(uint16_t bits);

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

} // namespace reference
