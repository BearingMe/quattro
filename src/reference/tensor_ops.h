#pragma once

#include "core/tensor.h"
#include <vector>

namespace reference {
// Caller validates descriptors. Independent scalar CPU oracle, with explicit
// FP32 rounding points and hand-written ties-to-even (not HLSL round).
std::vector<uint8_t> CpuTensorOp(const core::TensorOpDesc& desc,
                                const std::vector<uint8_t>& storage);
}
