#pragma once

#include <cstdint>
#include <string>

namespace core {

enum class TensorType : uint32_t { I8, F16, I32, F32 };

// NHWC, batch=1. Strides/offset are bytes. Channel slices retain the parent's
// pixel and row strides; padding is storage, never a logical channel.
struct TensorView {
    uint32_t width = 0, height = 0, channels = 0;
    uint32_t pixelStride = 0, rowStride = 0, offset = 0;
    TensorType type = TensorType::I8;
};

enum class TensorOp : uint32_t {
    Relu, Quantize, Affine, AffineResidual, HalfOutput,
    Copy, Concat, Pad, Spatial2x
};

// Per-output-channel arithmetic metadata. Concatenation preserves bytes;
// callers retain the appropriate scale for each resulting channel group.
struct ChannelTransform {
    float inputScale = 1.0f;
    float residualScale = 1.0f;
    float reciprocalOutputScale = 1.0f;
    uint32_t biasF16 = 0;
};
static_assert(sizeof(ChannelTransform) == 16);

struct TensorOpDesc {
    TensorOp op = TensorOp::Copy;
    TensorView a, b;
    uint32_t width = 0, height = 0, channels = 0;
    TensorType outputType = TensorType::I8;
    uint32_t transformOffset = 0;
    uint32_t padX = 0, padY = 0;
    bool relu = false;
};

uint32_t ElementBytes(TensorType type);
bool ValidateTensor(const TensorView& view, uint64_t storageBytes, std::string& error);

// Conservative full-range signed-INT8 dot-product bound, including -128*-128.
// Rejects before dispatch; absBias is an accumulator-domain bound, not FP16 bias.
bool ValidateInt8Accumulation(uint64_t terms, uint64_t absBias, std::string& error);

} // namespace core
