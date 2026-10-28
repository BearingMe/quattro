#pragma once
#include "gpu/amdext.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// Fixed geometry of one FasterNetBlock-style pass.
struct FnbGeometry {
    uint32_t pixels;
    uint32_t width;      // output width in pixels
    uint32_t channels;   // features (32 or 64)
    size_t inputOffset;  // byte offset of the input tensor in the scratch image
    size_t outputOffset; // byte offset of the output tensor
    uint32_t gridX;      // shipped dispatch grid
    uint32_t gridY;
    // Fused transposed-convolution stage (passes 9 and 11): the block output is
    // upscaled 2x and added to a skip tensor of the upscaled shape.
    uint32_t outFeatures; // 0 for plain FasterNetBlocks
    uint32_t outWidth;
    uint32_t outHeight;
    size_t skipOffset;    // byte offset of the skip tensor in the scratch image
};

// Passes 4, 5, 7, 8 and 10 are FasterNetBlocks; 9 and 11 fuse the block with a
// 2x transposed upscale and a residual add (`FNB_CT2D_ADD`).
bool FnbGeometryForPass(uint32_t pass, FnbGeometry& geometry);

// Model passes 4/5/7/8/10 (`int8_NHWC/Fused/FasterNetBlock`) and 9/11
// (`int8_NHWC/Fused/FNB_CT2D_ADD`): the residual block of the encoder/decoder
// stack, optionally fused with a 2x upscale and add.
bool RunFasterNetBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                       uint32_t pass, std::span<const uint8_t> input,
                       std::span<const uint8_t> skip, std::span<const uint8_t> weights,
                       std::vector<uint8_t>& output, std::string& isa, std::string& error);

} // namespace inference::fsr4
