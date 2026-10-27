#include "tensor_ops.h"
#include "compute.h"
#include "core/numerics.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace gpu {

using core::TensorOp;
using core::TensorOpDesc;
using core::ChannelTransform;

bool ValidateTensorOp(const TensorOpDesc& d, const std::vector<uint8_t>& storage,
                      std::string& error) {
    const auto fail = [&] { error = "invalid tensor operation shape, type or transform"; return false; };
    if (!core::ValidateTensor(d.a, storage.size(), error)) return false;
    const uint32_t bytes = core::ElementBytes(d.outputType);
    if (!d.width || !d.height || !d.channels || !bytes ||
        d.width > 65535 || d.height > 65535 || d.channels > 65535) return fail();
    const uint64_t count = uint64_t(d.width) * d.height * d.channels;
    if (count * bytes > (1u << 28)) return fail();
    const bool binary = d.op == TensorOp::Concat || d.op == TensorOp::AffineResidual;
    if (binary && !core::ValidateTensor(d.b, storage.size(), error)) return false;
    const bool sameSpatial = d.width == d.a.width && d.height == d.a.height;
    const bool sameShape = sameSpatial && d.channels == d.a.channels;
    const bool affine = d.op == TensorOp::Affine || d.op == TensorOp::AffineResidual;
    const bool transformed = affine || d.op == TensorOp::Quantize || d.op == TensorOp::HalfOutput;
    if (d.relu && !affine) return fail();
    if (transformed) {
        if (d.transformOffset % 4 || uint64_t(d.transformOffset) + uint64_t(d.channels) * 16 > storage.size()) return fail();
        for (uint32_t c = 0; c < d.channels; ++c) {
            ChannelTransform t;
            memcpy(&t, storage.data() + d.transformOffset + size_t(c) * 16, 16);
            if (!(t.inputScale > 0) || !std::isfinite(t.inputScale) ||
                !(t.residualScale > 0) || !std::isfinite(t.residualScale) ||
                !(t.reciprocalOutputScale > 0) || !std::isfinite(t.reciprocalOutputScale) ||
                t.biasF16 > 65535 || !std::isfinite(core::F16ToF32(uint16_t(t.biasF16)))) return fail();
            const double biasBound = std::abs(double(core::F16ToF32(uint16_t(t.biasF16))));
            const double bound = d.op == TensorOp::Quantize ? 65504.0 + biasBound :
                2147483648.0 * t.inputScale + biasBound +
                (binary ? 128.0 * t.residualScale : 0.0);
            if (bound > std::numeric_limits<float>::max() ||
                (d.op != TensorOp::HalfOutput && bound * t.reciprocalOutputScale > std::numeric_limits<float>::max())) return fail();
        }
    }
    switch (d.op) {
    case TensorOp::Relu:
        if (!sameShape || d.a.type != core::TensorType::I8 || d.outputType != d.a.type) return fail();
        break;
    case TensorOp::Quantize:
        if (!sameShape || (d.a.type != core::TensorType::F16 && d.a.type != core::TensorType::F32) ||
            d.outputType != core::TensorType::I8) return fail();
        break;
    case TensorOp::Affine: case TensorOp::AffineResidual: case TensorOp::HalfOutput:
        if (!sameShape || d.a.type != core::TensorType::I32 ||
            d.outputType != (d.op == TensorOp::HalfOutput ? core::TensorType::F16 : core::TensorType::I8)) return fail();
        if (binary && (d.b.type != core::TensorType::I8 || d.b.width != d.width ||
            d.b.height != d.height || d.b.channels != d.channels)) return fail();
        break;
    case TensorOp::Copy:
        if (!sameShape || d.outputType != d.a.type) return fail();
        break;
    case TensorOp::Concat:
        if (!sameSpatial || d.b.width != d.width || d.b.height != d.height ||
            uint64_t(d.a.channels) + d.b.channels != d.channels ||
            d.outputType != d.a.type || d.b.type != d.a.type) return fail();
        break;
    case TensorOp::Pad:
        if (uint64_t(d.a.width) + 2ull * d.padX != d.width ||
            uint64_t(d.a.height) + 2ull * d.padY != d.height ||
            d.channels != d.a.channels || d.outputType != d.a.type) return fail();
        break;
    case TensorOp::Spatial2x:
        if (uint64_t(d.a.width) * 2 != d.width || uint64_t(d.a.height) * 2 != d.height ||
            uint64_t(d.channels) * 4 != d.a.channels || d.outputType != d.a.type) return fail();
        break;
    default: return fail();
    }
    // Quantize accepts finite floating inputs only; reject unsafe FP32 math.
    if (d.op == TensorOp::Quantize) {
        for (uint32_t y = 0; y < d.height; ++y)
            for (uint32_t x = 0; x < d.width; ++x)
                for (uint32_t c = 0; c < d.channels; ++c) {
                    float value;
                    const uint32_t size = core::ElementBytes(d.a.type);
                    const auto* p = storage.data() + d.a.offset + size_t(y) * d.a.rowStride +
                                    size_t(x) * d.a.pixelStride + c * size;
                    if (size == 2) {
                        uint16_t h; memcpy(&h,p,2); value = core::F16ToF32(h);
                    } else memcpy(&value,p,4);
                    ChannelTransform t;
                    memcpy(&t,storage.data()+d.transformOffset+size_t(c)*16,16);
                    const double bound = std::abs(double(value)) + std::abs(double(core::F16ToF32(uint16_t(t.biasF16))));
                    if (!std::isfinite(value) || bound > std::numeric_limits<float>::max() ||
                        bound * t.reciprocalOutputScale > std::numeric_limits<float>::max()) return fail();
                }
    }
    return true;
}

bool RunTensorOp(DeviceContext& ctx, const TensorOpDesc& d,
                  const std::vector<uint8_t>& storage, std::vector<uint8_t>& warm,
                  std::vector<uint8_t>& repeated, std::string& error) {
    if (!ValidateTensorOp(d, storage, error)) return false;
    // Header ABI is a 32-dword SRV prefix, not root constants or a cbuffer.
    uint32_t h[32] = {};
    h[0] = uint32_t(d.op);
    const auto view = [&](uint32_t i, const core::TensorView& v) {
        h[i] = v.width; h[i+1] = v.height; h[i+2] = v.channels;
        h[i+3] = v.pixelStride; h[i+4] = v.rowStride;
        h[i+5] = v.offset + 128; h[i+6] = uint32_t(v.type);
    };
    view(1, d.a); view(8, d.b);
    h[15] = d.width; h[16] = d.height; h[17] = d.channels;
    h[18] = uint32_t(d.outputType); h[19] = d.transformOffset + 128;
    h[20] = d.padX; h[21] = d.padY; h[22] = d.relu;
    std::vector<uint8_t> blob(128 + ((storage.size() + 3) & ~size_t(3)), 0);
    memcpy(blob.data(), h, 128);
    memcpy(blob.data() + 128, storage.data(), storage.size());
    ComPtr<ID3D12RootSignature> root;
    std::vector<uint8_t> shader;
    if (!CreateGemmTileRootSignature(ctx, root, error) ||
        !LoadShaderBytecode("tensor_ops", shader, error)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC p{};
    p.pRootSignature = root.Get(); p.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(ctx.device->CreateComputePipelineState(&p, IID_PPV_ARGS(&pso)))) {
        error = "tensor_ops CreateComputePipelineState failed"; return false;
    }
    const size_t outBytes = size_t(d.width) * d.height * d.channels * core::ElementBytes(d.outputType);
    const size_t rounded = (outBytes + 3) & ~size_t(3);
    const uint32_t groups = uint32_t((rounded / 4 + 63) / 64);
    const uint32_t gridX = groups < 65535 ? groups : 65535;
    // Dispatch tiling is in the SRV ABI, so shapes larger than one D3D12 X
    // dimension (including the selected 1080p FP16 output) are supported.
    h[23] = gridX * 64;
    memcpy(blob.data(), h, 128);
    std::vector<int32_t> first, last;
    MatmulRunResult timing;
    if (!RunGemmTile(ctx, root, pso, blob.size(), rounded,
        gridX, (groups + gridX - 1) / gridX, 1, 3, blob.data(), first, last, timing, error)) return false;
    warm.resize(rounded); repeated.resize(rounded);
    memcpy(warm.data(), first.data(), rounded);
    memcpy(repeated.data(), last.data(), rounded);
    return InfoQueueClean(ctx, error);
}

} // namespace gpu
