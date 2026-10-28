#pragma once
#include "gpu/amdext.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// Fixed geometry of one FasterNetBlock pass.
struct FnbGeometry {
    uint32_t pixels;
    uint32_t width;      // output width in pixels
    uint32_t channels;   // features (32 or 64)
    size_t inputOffset;  // byte offset of the input tensor in the scratch image
    size_t outputOffset; // byte offset of the output tensor
    uint32_t gridX;      // shipped dispatch grid
    uint32_t gridY;
};

// Passes 4, 5, 7, 8 and 10 are the five FasterNetBlocks of the pinned graph.
bool FnbGeometryForPass(uint32_t pass, FnbGeometry& geometry);

// Model passes 4/5/7/8/10 (`int8_NHWC/Fused/FasterNetBlock`), the residual
// block of the encoder/decoder stack: grouped 3x3 partial convolution, 1x1
// expand with ReLU, 1x1 contract, then a two-scale residual add.
bool RunFasterNetBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                       uint32_t pass, std::span<const uint8_t> input,
                       std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                       std::string& isa, std::string& error);

} // namespace inference::fsr4
