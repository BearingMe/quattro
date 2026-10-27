#include "reference.h"

#include <cstdint>

namespace reference {

void CpuVectorAdd(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i)
        out[i] = a[i] + b[i];
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
