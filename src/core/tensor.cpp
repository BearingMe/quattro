#include "tensor.h"

#include <limits>

namespace core {

uint32_t ElementBytes(TensorType type) {
    switch (type) {
    case TensorType::I8: return 1;
    case TensorType::F16: return 2;
    case TensorType::I32: case TensorType::F32: return 4;
    }
    return 0;
}

bool ValidateTensor(const TensorView& v, uint64_t bytes, std::string& error) {
    const uint32_t b = ElementBytes(v.type);
    // Bound shader address arithmetic, allocations and dispatch dimensions.
    if (!b || !v.width || !v.height || !v.channels || bytes > (1u << 28) ||
        v.offset % b || v.pixelStride % b || v.rowStride % b ||
        uint64_t(v.channels) * b > v.pixelStride ||
        uint64_t(v.width) * v.pixelStride > v.rowStride ||
        uint64_t(v.offset) + uint64_t(v.height - 1) * v.rowStride +
            uint64_t(v.width - 1) * v.pixelStride + uint64_t(v.channels) * b > bytes) {
        error = "invalid NHWC tensor shape, strides, alignment or storage range";
        return false;
    }
    return true;
}

bool ValidateInt8Accumulation(uint64_t terms, uint64_t absBias, std::string& error) {
    const uint64_t limit = std::numeric_limits<int32_t>::max();
    if (!terms || absBias > limit || terms > (limit - absBias) / 16384) {
        error = "INT8 accumulation rejected: full-range INT32 bound exceeded";
        return false;
    }
    return true;
}

} // namespace core
