#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace reference {

using core::F16ToF32;
using core::F32ToF16;

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

// --- M08: independent FP8 oracles ---

namespace {

// Shared table builder: decodes an 8-bit float format by composing
// sign × exponent × mantissa with ldexp — deliberately structured
// differently from core's per-format decode.
void BuildF8Table(float* out256, uint32_t expBits, uint32_t mantBits,
                  int32_t bias, bool ieeeInf) {
    const uint32_t mantMask = (1u << mantBits) - 1u;
    const uint32_t maxExp = (1u << expBits) - 1u;
    for (uint32_t enc = 0; enc < 256; ++enc) {
        const bool negative = (enc & 0x80u) != 0;
        const uint32_t e = (enc >> mantBits) & maxExp;
        const uint32_t m = enc & mantMask;
        float v;
        if (e == maxExp && (ieeeInf || m == mantMask)) {
            // IEEE-style (E5M2): E'=max is inf (m=0) or NaN (m!=0).
            // OCP E4M3: no inf; NaN only at E'=max with all-ones mantissa.
            v = (ieeeInf && m == 0) ? std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::quiet_NaN();
        } else if (e == 0) {
            v = std::ldexp(float(m), 1 - int(bias) - int(mantBits));
        } else {
            v = std::ldexp(float(m | (mantMask + 1u)), int(e) - int(bias) - int(mantBits));
        }
        out256[enc] = negative ? -v : v;
    }
}

// Nearest-representable encode by linear scan. Ties round to the candidate
// whose last grid bit is 0 (round-to-nearest-even); overflow saturates.
uint8_t ReferenceEncodeF8(const float* table256, float value, uint32_t mantBits) {
    const uint32_t inBits = *reinterpret_cast<const uint32_t*>(&value);
    const uint8_t signBit = uint8_t((inBits >> 24) & 0x80u);
    if (std::isnan(value))
        return uint8_t(signBit | 0x7Fu);
    const float a = std::fabs(value);
    if (a == 0.0f)
        return signBit;

    uint8_t best = 0;
    float bestDiff = std::numeric_limits<float>::infinity();
    for (uint32_t enc = 0; enc < 256; ++enc) {
        const float v = table256[enc];
        if (std::isnan(v) || std::isinf(v))
            continue;
        const float diff = std::fabs(a - v);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = uint8_t(enc & 0x7Fu); // magnitude encoding
        } else if (diff == bestDiff) {
            // Exact tie: prefer the even mantissa grid position. The grid
            // value's mantissa bits are the low bits of the encoding.
            const uint32_t mantMask = (1u << mantBits) - 1u;
            if ((enc & mantMask) == 0 && (best & mantMask) != 0)
                best = uint8_t(enc & 0x7Fu);
        }
    }
    // Overflow: the scan naturally lands on max finite (nearest). Saturating
    // equals nearest here because max finite is the largest finite candidate.
    return uint8_t(best | signBit);
}

} // namespace

void BuildE4M3Table(float* out256) {
    // OCP E4M3: bias 7, 3 mantissa bits, no inf, NaN at E15|M7.
    BuildF8Table(out256, 4, 3, 7, false);
}

void BuildE5M2Table(float* out256) {
    // OCP E5M2: bias 15, 2 mantissa bits, IEEE-style inf/NaN.
    BuildF8Table(out256, 5, 2, 15, true);
}

uint8_t ReferenceEncodeE4M3(float value) {
    float table[256];
    BuildE4M3Table(table);
    return ReferenceEncodeF8(table, value, 3);
}

uint8_t ReferenceEncodeE5M2(float value) {
    float table[256];
    BuildE5M2Table(table);
    return ReferenceEncodeF8(table, value, 2);
}

// --- M08: symmetric INT8 quantize oracle ---

int8_t ReferenceQuantizeS8(float value, float scale, bool roundNearestEven) {
    if (!(scale > 0.0f) || !std::isfinite(scale))
        return 0;
    if (std::isnan(value))
        return 0;
    // Same arithmetic contract as AMD: reciprocal, multiply, round.
    const float scaled = value * (1.0f / scale);
    float rounded;
    if (roundNearestEven) {
        // Hand-rolled ties-to-even (independent of nearbyint).
        const float fl = std::floor(scaled);
        const float frac = scaled - fl;
        if (frac > 0.5f)
            rounded = fl + 1.0f;
        else if (frac < 0.5f)
            rounded = fl;
        else
            rounded = (std::fmod(fl, 2.0f) == 0.0f) ? fl : fl + 1.0f;
    } else {
        rounded = std::trunc(scaled);
    }
    if (rounded >= 127.0f)
        return 127;
    if (rounded <= -128.0f)
        return -128;
    return int8_t(rounded);
}

float ReferenceDequantizeS8(int8_t q, float scale) {
    return float(q) * scale;
}

QuantErrorStats ComputeQuantError(const float* ref, const int8_t* q,
                                  const float* dequant, size_t count) {
    QuantErrorStats stats;
    if (count == 0)
        return stats;
    double noisePower = 0.0;
    double signalPower = 0.0;
    size_t clipped = 0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = double(dequant[i]) - double(ref[i]);
        stats.maxAbsErr = std::max(stats.maxAbsErr, std::fabs(diff));
        stats.mae += std::fabs(diff);
        noisePower += diff * diff;
        signalPower += double(ref[i]) * double(ref[i]);
        if (q[i] >= 127 || q[i] <= -128)
            ++clipped;
    }
    stats.mae /= double(count);
    stats.rmse = std::sqrt(noisePower / double(count));
    stats.clipFrac = double(clipped) / double(count);
    stats.sqnrDb = noisePower > 0.0
                       ? 10.0 * std::log10(signalPower / noisePower)
                       : std::numeric_limits<double>::infinity();
    return stats;
}

} // namespace reference
