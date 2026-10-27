#include "fsr4_cnb.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

#include <algorithm>

namespace inference::fsr4 {

bool RunConvNextBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                      uint32_t passIndex, std::span<const uint8_t> input,
                      std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                      std::vector<std::string>& convolutionIsa, std::string& error) {
    output.clear();
    convolutionIsa.clear();
    constexpr size_t pixels = 960 * 540;
    if ((passIndex != 1 && passIndex != 2 && passIndex != 12) || input.size() != pixels * 16) {
        error = "unsupported ConvNextBlock pass or input shape";
        return false;
    }
    if (!ValidateWeights(weights, error))
        return false;
    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(context, signature, error, true))
        return false;
    const uint32_t reductions[] = {144, 16, 32};
    const uint32_t features[] = {16, 32, 16};
    size_t packedBytes = 0, accBytes = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        packedBytes = std::max(packedBytes, pixels * reductions[i] + reductions[i] * features[i] + 1024);
        accBytes = std::max(accBytes, pixels * features[i] * 4);
    }
    const std::vector<gpu::BufferInit> buffers = {
        {input.size(), input.data()}, {weights.size(), weights.data()},
        {packedBytes, nullptr}, {accBytes, nullptr},
        {pixels * 16, nullptr}, {pixels * 32, nullptr}, {pixels * 16, nullptr}, {4, nullptr},
    };
    std::vector<gpu::ComPtr<ID3D12PipelineState>> psos;
    std::vector<gpu::BufferPass> stages;
    for (uint32_t conv = 0; conv < 3; ++conv) {
        for (uint32_t step = 0; step < 3; ++step) {
            const std::string name = step == 1
                ? "m12/cnb" + std::to_string(conv) + "raw"
                : "m12/p" + std::to_string(passIndex) + "c" + std::to_string(conv) +
                      (step == 0 ? "pack" : "post");
            std::vector<uint8_t> cso;
            if (!gpu::LoadShaderBytecode(name.c_str(), cso, error))
                return false;
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = signature.Get();
            desc.CS = {cso.data(), cso.size()};
            gpu::ComPtr<ID3D12PipelineState> pso;
            if (step == 1) {
                AmdExtD3DPipelineHandle handle{};
                AmdExtD3DComputeShaderStats stats{};
                if (!gpu::CreateAnalyzedComputePipeline(extensions, desc, pso, handle, stats, error))
                    return false;
                std::string isa, statsLine;
                if (!gpu::GetPipelineIsa(extensions, handle, isa, statsLine, error))
                    return false;
                convolutionIsa.push_back(std::move(isa));
            } else if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
                error = "cannot create ConvNextBlock packing/epilogue PSO: " + name;
                return false;
            }
            gpu::BufferPass stage{};
            stage.pso = pso.Get();
            stage.srv1 = 1;
            stage.uav1 = 7;
            if (step == 0) {
                stage.srv0 = conv == 0 ? 0 : conv == 1 ? 4 : 5;
                stage.uav0 = 2;
                const size_t words = (pixels * reductions[conv] + reductions[conv] * features[conv] + 1024) / 4;
                stage.gridX = 4096;
                stage.gridY = uint32_t((words + 4096 * 64 - 1) / (4096 * 64));
            } else if (step == 1) {
                stage.srv0 = 2;
                stage.uav0 = 3;
                stage.gridX = features[conv] / 16;
                stage.gridY = uint32_t(pixels / 16);
            } else {
                stage.srv0 = 3;
                stage.uav0 = conv == 0 ? 4 : conv == 1 ? 5 : 6;
                stage.uav1 = conv == 2 ? 0 : 7;
                stage.gridX = 4096;
                stage.gridY = uint32_t((pixels + 4096 * 64 - 1) / (4096 * 64));
            }
            psos.push_back(pso);
            stages.push_back(stage);
        }
    }
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, signature, buffers, stages, 6,
                                std::vector<bool>(stages.size(), false), snapshots,
                                6, pixels * 16, output, error, true);
}

} // namespace inference::fsr4
