#include "fsr4_fnb.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

namespace inference::fsr4 {

bool FnbGeometryForPass(uint32_t pass, FnbGeometry& geometry) {
    switch (pass) {
    case 4: // 480x270x32, encoder3 ResidualBlock 0
        geometry = {480 * 270, 480, 32, 8294400, 12441600, 8, 270, 0, 0, 0, 0};
        return true;
    case 5: // 480x270x32, encoder3 ResidualBlock 1 (quantized output tensor)
        geometry = {480 * 270, 480, 32, 12441600, 8294400, 8, 270, 0, 0, 0, 0};
        return true;
    case 7: // 240x135x64, bottleneck ResidualBlock 0
        geometry = {240 * 135, 240, 64, 12441600, 14515200, 4, 135, 0, 0, 0, 0};
        return true;
    case 8: // 240x135x64, bottleneck ResidualBlock 1
        geometry = {240 * 135, 240, 64, 14515200, 12441600, 4, 135, 0, 0, 0, 0};
        return true;
    case 9: // 240x135x64 -> 480x270x32, bottleneck ResidualBlock 2 + upscale
        geometry = {240 * 135, 240, 64, 12441600, 16588800, 30, 17, 32, 480, 270, 8294400};
        return true;
    case 10: // 480x270x32, decoder3 ResidualBlock 1
        geometry = {480 * 270, 480, 32, 16588800, 8294400, 8, 270, 0, 0, 0, 0};
        return true;
    case 11: // 480x270x32 -> 960x540x16, decoder3 ResidualBlock 2 + upscale
        geometry = {480 * 270, 480, 32, 8294400, 12441600, 8, 270, 16, 960, 540, 0};
        return true;
    default:
        return false;
    }
}

bool RunFasterNetBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                       uint32_t pass, std::span<const uint8_t> input,
                       std::span<const uint8_t> skip, std::span<const uint8_t> weights,
                       std::vector<uint8_t>& output, std::string& isa, std::string& error) {
    output.clear();
    isa.clear();
    FnbGeometry shape{};
    if (!FnbGeometryForPass(pass, shape)) {
        error = "unsupported FasterNetBlock pass";
        return false;
    }
    if (input.size() != size_t(shape.pixels) * shape.channels) {
        error = "unsupported FasterNetBlock input shape";
        return false;
    }
    const size_t tensorBytes = size_t(shape.pixels) * shape.channels;
    const size_t outBytes = shape.outFeatures == 0
        ? tensorBytes
        : size_t(shape.outWidth) * shape.outHeight * shape.outFeatures;
    if (shape.outFeatures != 0 && skip.size() != outBytes) {
        error = "unsupported FasterNetBlock skip tensor shape";
        return false;
    }
    if (!ValidateWeights(weights, error))
        return false;

    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(context, signature, error))
        return false;
    std::vector<uint8_t> cso;
    const std::string name = "m12/fnb" + std::to_string(pass);
    if (!gpu::LoadShaderBytecode(name.c_str(), cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    AmdExtD3DPipelineHandle handle{};
    AmdExtD3DComputeShaderStats stats{};
    if (!gpu::CreateAnalyzedComputePipeline(extensions, desc, pso, handle, stats, error))
        return false;
    std::string statsLine;
    if (!gpu::GetPipelineIsa(extensions, handle, isa, statsLine, error))
        return false;

    std::vector<uint8_t> source;
    source.reserve(input.size() + skip.size());
    source.insert(source.end(), input.begin(), input.end());
    source.insert(source.end(), skip.begin(), skip.end());
    const std::vector<gpu::BufferInit> buffers = {
        {source.size(), source.data()},    // 0: source tensor (then skip tensor)
        {weights.size(), weights.data()},  // 1: checked payload
        {outBytes, nullptr},               // 2: output tensor
        {4, nullptr},                      // 3: bound-but-unused second UAV slot
    };
    gpu::BufferPass stage{};
    stage.pso = pso.Get();
    stage.gridX = (shape.pixels + 63) / 64;
    stage.gridY = 1;
    stage.srv0 = 0;
    stage.srv1 = 1;
    stage.uav0 = 2;
    stage.uav1 = 3;
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, signature, buffers, {stage}, 3, {false}, snapshots, 2,
                                outBytes, output, error);
}

} // namespace inference::fsr4

