#include "fsr4_cnb.h"
#include "fsr4_graph.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

#include <algorithm>

namespace inference::fsr4 {

bool AppendConvNextBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                         DispatchPlan& plan, uint32_t passIndex, size_t input, size_t weights,
                         size_t& output, std::string& error) {
    constexpr size_t pixels = 960 * 540;
    if (!plan.signature || (passIndex != 1 && passIndex != 2 && passIndex != 12) ||
        input >= plan.buffers.size() || weights >= plan.buffers.size() || input == weights ||
        plan.buffers[input].bytes != pixels * 16) {
        error = "invalid ConvNextBlock dispatch inputs";
        return false;
    }
    const uint32_t reductions[] = {144, 16, 32};
    const uint32_t features[] = {16, 32, 16};
    size_t packedBytes = 0, accBytes = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        packedBytes = std::max(packedBytes, pixels * reductions[i] + reductions[i] * features[i] + 1024);
        accBytes = std::max(accBytes, pixels * features[i] * 4);
    }
    const size_t workspace = plan.buffers.size();
    plan.buffers.insert(plan.buffers.end(), {
        {packedBytes, nullptr}, {accBytes, nullptr},
        {pixels * 16, nullptr}, {pixels * 32, nullptr}, {pixels * 16, nullptr}, {4, nullptr},
    });
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
            desc.pRootSignature = plan.signature.Get();
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
                plan.convolutionIsa.push_back(std::move(isa));
                plan.convolutionStats.push_back(statsLine);
            } else if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
                error = "cannot create ConvNextBlock packing/epilogue PSO: " + name;
                return false;
            }
            gpu::BufferPass stage{};
            stage.pso = pso.Get();
            stage.srv1 = weights;
            stage.uav1 = workspace + 5;
            if (step == 0) {
                stage.srv0 = conv == 0 ? input : conv == 1 ? workspace + 2 : workspace + 3;
                stage.uav0 = workspace;
                const size_t words = (pixels * reductions[conv] + reductions[conv] * features[conv] + 1024) / 4;
                stage.gridX = 4096;
                stage.gridY = uint32_t((words + 4096 * 64 - 1) / (4096 * 64));
            } else if (step == 1) {
                stage.srv0 = workspace;
                stage.uav0 = workspace + 1;
                stage.gridX = features[conv] / 16;
                stage.gridY = uint32_t(pixels / 16);
            } else {
                stage.srv0 = workspace + 1;
                stage.uav0 = conv == 0 ? workspace + 2 : conv == 1 ? workspace + 3 : workspace + 4;
                stage.uav1 = conv == 2 ? input : workspace + 5;
                stage.gridX = 4096;
                stage.gridY = uint32_t((pixels + 4096 * 64 - 1) / (4096 * 64));
            }
            plan.psos.push_back(pso);
            plan.passes.push_back(stage);
        }
    }
    output = workspace + 4;
    return true;
}

bool RunConvNextBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                      uint32_t passIndex, std::span<const uint8_t> input,
                      std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                      std::vector<std::string>& convolutionIsa, std::string& error) {
    output.clear();
    convolutionIsa.clear();
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error, true))
        return false;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()}};
    size_t destination = 0;
    if (!AppendConvNextBlock(context, extensions, plan, passIndex, 0, 1, destination, error))
        return false;
    convolutionIsa = std::move(plan.convolutionIsa);
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes, destination,
                                std::vector<bool>(plan.passes.size(), false), snapshots,
                                destination, size_t(960) * 540 * 16, output, error, true);
}

} // namespace inference::fsr4
