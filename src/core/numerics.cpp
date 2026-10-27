#include "numerics.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace core {

// --- FP16 ---

float F16ToF32(uint16_t bits) {
    const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1Fu;
    const uint32_t mantissa = bits & 0x03FFu;
    uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign; // zero (preserves -0)
        } else {
            // Denormal: value = mantissa * 2^-24. Normalized: after e shifts
            // the implicit bit (0x0400) is set, value = (1 + frac) * 2^(-14 - e),
            // so the fp32 exponent field is 127 - 14 - e = 113 - e.
            uint32_t e = 0;
            uint32_t m = mantissa;
            while (!(m & 0x0400u)) {
                m <<= 1;
                ++e;
            }
            out = sign | ((113 - e) << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7F800000u | (mantissa << 13); // inf / NaN (payload kept)
    } else {
        out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return *reinterpret_cast<const float*>(&out);
}

uint16_t F32ToF16(float value) {
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    if (std::isnan(value))
        return uint16_t(sign | 0x7E00u); // qNaN
    const uint32_t mantissa24 = (bits & 0x007FFFFFu) | 0x00800000u; // implicit bit in
    const int32_t fp16Exp = int32_t((bits >> 23) & 0xFFu) - 112;    // fp16 exponent field
    if (fp16Exp >= 31)
        return uint16_t(sign | 0x7C00u); // overflow -> inf
    if (fp16Exp <= 0) {
        if (fp16Exp < -10)
            return uint16_t(sign); // magnitude < 2^-25: rounds to zero (RNE)
        // Denormal target: k * 2^-24 with k = mantissa24 >> (14 - fp16Exp),
        // rounded to nearest even on the dropped bits. shift in [14, 24].
        const uint32_t shift = uint32_t(14 - fp16Exp);
        uint32_t k = mantissa24 >> shift;
        const uint32_t rem = mantissa24 & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (k & 1u)))
            k += 1;
        if (k >= 0x400u) // rounded up into the normal range (min normal 2^-14)
            return uint16_t(sign | (1u << 10));
        return uint16_t(sign | uint16_t(k));
    }
    uint32_t result = (uint32_t(fp16Exp) << 10) | ((mantissa24 & 0x007FFFFFu) >> 13);
    const uint32_t rounding = mantissa24 & 0x1FFFu;
    if (rounding > 0x1000u || (rounding == 0x1000u && ((mantissa24 >> 13) & 1u)))
        result += 1; // may carry into the exponent — correct behavior
    return uint16_t(sign | uint16_t(result));
}

// --- FP8 ---

float E4M3ToF32(uint8_t bits) {
    const bool negative = (bits & 0x80u) != 0;
    const uint32_t exponent = (bits >> 3) & 0x0Fu;
    const uint32_t mantissa = bits & 0x07u;
    float value;
    if (exponent == 15 && mantissa == 7) {
        value = std::numeric_limits<float>::quiet_NaN(); // OCP E4M3: no inf
    } else if (exponent == 0) {
        value = std::ldexp(float(mantissa), -9); // m/8 * 2^-6
    } else {
        value = std::ldexp(float(mantissa | 0x08u), int(exponent) - 10); // (1+m/8)*2^(e-7)
    }
    return negative ? -value : value;
}

float E5M2ToF32(uint8_t bits) {
    const bool negative = (bits & 0x80u) != 0;
    const uint32_t exponent = (bits >> 2) & 0x1Fu;
    const uint32_t mantissa = bits & 0x03u;
    float value;
    if (exponent == 31) {
        value = (mantissa == 0) ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(float(mantissa), -16); // m/4 * 2^-14
    } else {
        value = std::ldexp(float(mantissa | 0x04u), int(exponent) - 17); // (1+m/4)*2^(e-15)
    }
    return negative ? -value : value;
}

uint8_t F32ToE4M3(float value) {
    const uint32_t inBits = *reinterpret_cast<const uint32_t*>(&value);
    const uint8_t sign = uint8_t((inBits >> 24) & 0x80u);
    if (std::isnan(value))
        return uint8_t(sign | 0x7Fu); // NaN: the only all-ones encoding
    float a = std::fabs(value);
    if (a == 0.0f)
        return sign; // zero, sign preserved
    // Saturate beyond the max finite (448). Clamping first also guarantees
    // the grid below can never produce the mantissa-7-at-max-exponent NaN.
    if (a > 448.0f)
        a = 448.0f;

    const int e = std::ilogb(a);
    if (e < -6) {
        // Denormal: unit 2^-9. k = a * 2^9, RNE; k in [0, 8) here.
        const float k = std::ldexp(a, 9);
        const uint32_t q = uint32_t(std::nearbyint(k)); // ties-to-even
        if (q >= 8u)                                    // carry: min normal 2^-6
            return uint8_t(sign | (1u << 3));
        return uint8_t(sign | uint8_t(q));
    }
    const float frac = std::ldexp(a, 3 - e); // in [8, 16)
    uint32_t q = uint32_t(std::nearbyint(frac));
    int expField = e + 7;
    if (q >= 16u) { // rounded up to the next power of two
        q = 8u;
        expField += 1;
    }
    return uint8_t(sign | (uint32_t(expField) << 3) | uint8_t(q - 8u));
}

uint8_t F32ToE5M2(float value) {
    const uint32_t inBits = *reinterpret_cast<const uint32_t*>(&value);
    const uint8_t sign = uint8_t((inBits >> 24) & 0x80u);
    if (std::isnan(value))
        return uint8_t(sign | 0x7Fu); // NaN (E5M2 has many NaN encodings; m=3 chosen)
    const float a = std::fabs(value);
    if (a == 0.0f)
        return sign;
    // Saturate beyond the max finite (57344 = E30 M3).
    if (a > 57344.0f)
        return uint8_t(sign | 0x7Bu);

    const int e = std::ilogb(a);
    if (e < -14) {
        // Denormal: unit 2^-16. k = a * 2^16, RNE; k in [0, 4) here.
        const float k = std::ldexp(a, 16);
        const uint32_t q = uint32_t(std::nearbyint(k));
        if (q >= 4u) // carry: min normal 2^-14
            return uint8_t(sign | (1u << 2));
        return uint8_t(sign | uint8_t(q));
    }
    const float frac = std::ldexp(a, 2 - e); // in [4, 8)
    uint32_t q = uint32_t(std::nearbyint(frac));
    int expField = e + 15;
    if (q >= 8u) { // rounded up to the next power of two
        q = 4u;
        expField += 1;
    }
    return uint8_t(sign | (uint32_t(expField) << 2) | uint8_t(q - 4u));
}

// --- Symmetric INT8 (AMD/FSR 4 v07 contract) ---

int8_t QuantizeSymmetricS8(float value, float scale, QuantizeRounding rounding) {
    // Invalid scale metadata: quantization is undefined; emit 0 so callers
    // that forgot validation get a detectable all-zero tensor, not garbage.
    if (!(scale > 0.0f) || !std::isfinite(scale))
        return 0;
    if (std::isnan(value))
        return 0; // characterized against hardware in the M08 probe

    // Bit-exactness with AMD's operator requires the same arithmetic:
    // reciprocal first, then multiply (not division), then round.
    const float rcpScale = 1.0f / scale;
    const float scaled = value * rcpScale;
    const float rounded = rounding == QuantizeRounding::RoundNearestEven
                              ? std::nearbyint(scaled) // default FP mode: ties-to-even
                              : std::trunc(scaled);
    if (rounded >= 127.0f)
        return 127;
    if (rounded <= -128.0f)
        return -128;
    return int8_t(rounded);
}

float DequantizeSymmetricS8(int8_t q, float scale) {
    return float(q) * scale;
}

} // namespace core
