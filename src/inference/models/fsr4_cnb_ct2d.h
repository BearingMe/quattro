#pragma once
#include "gpu/amdext.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// Model pass 13 (`float16_NHWC/Fused/CNB_CT2D`): the last ConvNextBlock of the
// bottleneck, fused with a 2x2 transposed upscale that writes the network's
// FP16 1920x1080x8 output.
constexpr size_t kCnbCt2dInputPixels = 960 * 540;
constexpr size_t kCnbCt2dInputBytes = kCnbCt2dInputPixels * 16;
constexpr size_t kCnbCt2dOutputBytes = size_t(1920) * 1080 * 8 * 2;

bool RunCnbCt2d(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                std::span<const uint8_t> input, std::span<const uint8_t> weights,
                std::vector<uint8_t>& output, std::string& isa, std::string& error);

} // namespace inference::fsr4
