#include "fsr4_fnb.h"
#include "fsr4_graph.h"
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

bool AppendFasterNetBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                          DispatchPlan& plan, uint32_t pass, size_t input, size_t skip,
                          size_t weights, size_t& output, std::string& error) {
    (void)context; // PSOs are created through the supplied AMD extension interface.
    FnbGeometry shape{};
    if (!FnbGeometryForPass(pass, shape)) {
        error = "unsupported FasterNetBlock pass";
        return false;
    }
    if (!plan.signature || input >= plan.buffers.size() || weights >= plan.buffers.size() ||
        input == weights || plan.buffers[input].bytes != size_t(shape.pixels) * shape.channels) {
        error = "unsupported FasterNetBlock input shape";
        return false;
    }
    const size_t tensorBytes = size_t(shape.pixels) * shape.channels;
    const size_t outBytes = shape.outFeatures == 0
        ? tensorBytes
        : size_t(shape.outWidth) * shape.outHeight * shape.outFeatures;
    if (shape.outFeatures != 0 &&
        (skip >= plan.buffers.size() || plan.buffers[skip].bytes != outBytes)) {
        error = "unsupported FasterNetBlock skip tensor shape";
        return false;
    }
    std::vector<uint8_t> cso;
    const std::string name = "m12/fnb" + std::to_string(pass);
    if (!gpu::LoadShaderBytecode(name.c_str(), cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = plan.signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    AmdExtD3DPipelineHandle handle{};
    AmdExtD3DComputeShaderStats stats{};
    if (!gpu::CreateAnalyzedComputePipeline(extensions, desc, pso, handle, stats, error))
        return false;
    std::string isa, statsLine;
    if (!gpu::GetPipelineIsa(extensions, handle, isa, statsLine, error))
        return false;

    plan.convolutionIsa.push_back(std::move(isa));
    gpu::BufferPass stage{};
    size_t source = input;
    if (shape.outFeatures != 0) {
        source = plan.buffers.size();
        plan.buffers.push_back({tensorBytes + outBytes, nullptr});
        stage.copiesBefore = {{input, source, 0, 0, tensorBytes},
                              {skip, source, 0, tensorBytes, outBytes}};
    }
    const size_t destination = plan.buffers.size();
    plan.buffers.push_back({outBytes, nullptr});
    const size_t unused = plan.buffers.size();
    plan.buffers.push_back({4, nullptr});
    stage.pso = pso.Get();
    stage.gridX = (shape.pixels + 63) / 64;
    stage.gridY = 1;
    stage.srv0 = source;
    stage.srv1 = weights;
    stage.uav0 = destination;
    stage.uav1 = unused;
    plan.psos.push_back(pso);
    plan.passes.push_back(stage);
    output = destination;
    return true;
}

bool RunFasterNetBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                       uint32_t pass, std::span<const uint8_t> input,
                       std::span<const uint8_t> skip, std::span<const uint8_t> weights,
                       std::vector<uint8_t>& output, std::string& isa, std::string& error) {
    output.clear();
    isa.clear();
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error))
        return false;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()},
                    {skip.empty() ? 4 : skip.size(), skip.empty() ? nullptr : skip.data()}};
    size_t destination = 0;
    if (!AppendFasterNetBlock(context, extensions, plan, pass, 0, 2, 1, destination, error))
        return false;
    isa = plan.convolutionIsa.front();
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes, destination,
                                {false}, snapshots, destination,
                                plan.buffers[destination].bytes, output, error);
}

} // namespace inference::fsr4

