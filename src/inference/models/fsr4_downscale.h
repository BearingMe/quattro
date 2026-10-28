#pragma once
#include "gpu/amdext.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// Fixed geometry of one quantized downscale convolution pass.
struct DownscaleGeometry {
    uint32_t inputPixels;
    uint32_t inputChannels;
    uint32_t pixels;   // output pixels
    uint32_t features; // output features
    uint32_t reduction; // 2*2*inputChannels
    uint32_t width;     // output width in pixels (diagnostics only)
};

// Passes 3 and 6 are the only quantized stride-2 downscale convolutions in the
// pinned v07 INT8 Quality graph. Returns false for any other pass.
bool DownscaleGeometryForPass(uint32_t pass, DownscaleGeometry& geometry);

// Model passes 3 and 6: quantized stride-2 downscale convolution
// (`int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput`).
//
// One native-WMMA INT32 GEMM over a GPU-packed 2x2 im2col operand, followed by
// the operator's two-group requantization epilogue. Every intermediate stays on
// the GPU. `isa` receives the analyzed GEMM pipeline's ISA for evidence.
bool RunDownscaleConvolution(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                             uint32_t pass, std::span<const uint8_t> input,
                             std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                             std::string& isa, std::string& error);

} // namespace inference::fsr4
