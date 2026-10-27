#pragma once
#include "gpu/amdext.h"

#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// Passes 1, 2 and 12 have the same geometry but distinct constants/scales.
// Three native WMMA convolutions, with GPU-only packing and requantization.
bool RunConvNextBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                      uint32_t pass, std::span<const uint8_t> input,
                      std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                      std::vector<std::string>& convolutionIsa, std::string& error);

} // namespace inference::fsr4
