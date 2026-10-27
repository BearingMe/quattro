#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace reference {

void CpuVectorAdd(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i)
        out[i] = a[i] + b[i];
}

void CpuInt8Matmul16x16x16(const uint8_t* a, const uint8_t* b, const int32_t* initC,
                           int32_t* out, uint32_t mode) {
    for (uint32_t i = 0; i < 16; ++i) {
        for (uint32_t j = 0; j < 16; ++j) {
            int32_t acc = initC[i * 16 + j];
            for (uint32_t k = 0; k < 16; ++k) {
                int32_t av;
                int32_t bv;
                switch (mode) {
                    case 1:  av = int32_t(a[i * 16 + k]); bv = int32_t(b[k * 16 + j]); break; // u8*u8
                    case 2:  av = int32_t(int8_t(a[i * 16 + k])); bv = int32_t(b[k * 16 + j]); break; // i8*u8
                    case 3:  av = int32_t(a[i * 16 + k]); bv = int32_t(int8_t(b[k * 16 + j])); break; // u8*i8
                    default: av = int32_t(int8_t(a[i * 16 + k])); bv = int32_t(int8_t(b[k * 16 + j])); break; // i8*i8
                }
                acc += av * bv;
            }
            out[i * 16 + j] = acc;
        }
    }
}

size_t CountI32Mismatches(const int32_t* gpu, const int32_t* cpu, size_t count) {
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i)
        if (gpu[i] != cpu[i])
            ++mismatches;
    return mismatches;
}

uint16_t F32ToF16(float value) {
    // IEEE 754 binary16 pack, round-to-nearest-even. Denormal/overflow inputs
    // are outside the probe's value range; they clamp to inf per hardware
    // conversion behavior — recorded, not exercised (probe uses [-1, 1]).
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 127 + 15; // fp16 bias
    uint32_t mantissa = bits & 0x007FFFFFu;
    if (exponent >= 31)
        return uint16_t(sign | 0x7C00u); // inf
    if (exponent <= 0) {
        if (exponent < -10)
            return uint16_t(sign); // underflow to zero
        // Denormal: shift the implicit bit in and round.
        mantissa |= 0x00800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        const uint32_t half = 1u << (shift - 1);
        mantissa = (mantissa + half - 1) >> shift; // approximate RNE
        return uint16_t(sign | mantissa);
    }
    uint32_t result = uint32_t(exponent) << 10 | (mantissa >> 13);
    // Round to nearest even on the truncated 13 bits.
    const uint32_t rounding = mantissa & 0x1FFFu;
    const uint32_t lsb = mantissa & 0x3FFFu;
    if (rounding > 0x1000u || (rounding == 0x1000u && (lsb & 0x2000u)))
        result += 1; // may carry into the exponent — correct behavior
    return uint16_t(sign | result);
}

float F16ToF32(uint16_t bits) {
    const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1Fu;
    const uint32_t mantissa = bits & 0x03FFu;
    uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign; // zero
        } else {
            // Denormal: normalize.
            uint32_t e = 0;
            uint32_t m = mantissa;
            while (!(m & 0x0400u)) {
                m <<= 1;
                ++e;
            }
            m &= 0x03FFu;
            out = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7F800000u | (mantissa << 13); // inf / nan
    } else {
        out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return *reinterpret_cast<const float*>(&out);
}

void CpuF16Matmul16x16x16(const uint16_t* a, const uint16_t* b, const void* initC,
                          void* out, uint32_t accBitsPerElement) {
    if (accBitsPerElement == 16) {
        const uint16_t* init = static_cast<const uint16_t*>(initC);
        uint16_t* dst = static_cast<uint16_t*>(out);
        for (uint32_t i = 0; i < 16; ++i) {
            for (uint32_t j = 0; j < 16; ++j) {
                float acc = F16ToF32(init[i * 16 + j]);
                for (uint32_t k = 0; k < 16; ++k) {
                    const float product = F16ToF32(a[i * 16 + k]) * F16ToF32(b[k * 16 + j]);
                    acc = F16ToF32(F32ToF16(acc + product)); // fp16 rounding per add
                }
                dst[i * 16 + j] = F32ToF16(acc);
            }
        }
    } else {
        const float* init = static_cast<const float*>(initC);
        float* dst = static_cast<float*>(out);
        for (uint32_t i = 0; i < 16; ++i) {
            for (uint32_t j = 0; j < 16; ++j) {
                float acc = init[i * 16 + j];
                for (uint32_t k = 0; k < 16; ++k)
                    acc += F16ToF32(a[i * 16 + k]) * F16ToF32(b[k * 16 + j]);
                dst[i * 16 + j] = acc;
            }
        }
    }
}

float MaxAbsDiffF32(const float* gpu, const float* cpu, size_t count) {
    float maxDiff = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(gpu[i] - cpu[i]);
        // NaN outputs must never pass a tolerance compare silently.
        if (std::isnan(diff))
            return diff;
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

float MaxAbsDiffF16(const uint16_t* gpu, const uint16_t* cpu, size_t count) {
    float maxDiff = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(F16ToF32(gpu[i]) - F16ToF32(cpu[i]));
        if (std::isnan(diff))
            return diff;
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

size_t CountMismatches(const float* gpu, const float* cpu, size_t count) {
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        // Bit-wise comparison: identical fp32 adds must produce identical bits.
        const uint32_t g = *reinterpret_cast<const uint32_t*>(&gpu[i]);
        const uint32_t c = *reinterpret_cast<const uint32_t*>(&cpu[i]);
        if (g != c)
            ++mismatches;
    }
    return mismatches;
}

} // namespace reference
