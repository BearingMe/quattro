#include "reference.h"

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
