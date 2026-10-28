#include "fsr4_pass0.h"
#include "fsr4_graph.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

namespace inference::fsr4 {

bool AppendInputConvolution(gpu::DeviceContext& context, DispatchPlan& plan,
                            size_t input, size_t weights, size_t& output,
                            std::string& error) {
    if (!plan.signature || input >= plan.buffers.size() || weights >= plan.buffers.size() ||
        input == weights || plan.buffers[input].bytes != size_t(1920) * 1080 * 16) {
        error = "invalid input convolution dispatch inputs";
        return false;
    }
    const size_t destination = plan.buffers.size();
    plan.buffers.push_back({size_t(960) * 540 * 16, nullptr});
    const size_t unused = plan.buffers.size();
    plan.buffers.push_back({4, nullptr});
    std::vector<uint8_t> bytecode;
    if (!gpu::LoadShaderBytecode("m12/pass0", bytecode, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = plan.signature.Get();
    desc.CS = {bytecode.data(), bytecode.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        error = "cannot create FSR4 input convolution PSO";
        return false;
    }
    gpu::BufferPass stage{};
    stage.pso = pso.Get();
    stage.gridX = (960 * 540 + 63) / 64;
    stage.srv0 = input;
    stage.srv1 = weights;
    stage.uav0 = destination;
    stage.uav1 = unused;
    plan.psos.push_back(pso);
    plan.passes.push_back(stage);
    output = destination;
    return true;
}

bool RunInputConvolution(gpu::DeviceContext& context, std::span<const uint8_t> input,
                         std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                         std::string& error) {
    output.clear();
    constexpr size_t inputBytes = size_t(1920) * 1080 * 8 * 2;
    constexpr size_t outputBytes = size_t(960) * 540 * 16;
    if (input.size() != inputBytes) {
        error = "FSR4 input convolution requires 1920x1080x8 FP16 storage";
        return false;
    }
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error))
        return false;
    plan.buffers = {
        {input.size(), input.data()}, {weights.size(), weights.data()},
    };
    size_t destination = 0;
    if (!AppendInputConvolution(context, plan, 0, 1, destination, error))
        return false;
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes,
                                destination, {false}, snapshots, destination,
                                outputBytes, output, error);
}

} // namespace inference::fsr4
