#pragma once

#include "gpu/device.h"

#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// First real model pass, not a stand-in for complete model execution.
bool RunInputConvolution(gpu::DeviceContext& context, std::span<const uint8_t> input,
                         std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                         std::string& error);

} // namespace inference::fsr4
