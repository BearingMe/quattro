#include "fsr4_cnb_ct2d.h"
#include "fsr4_graph.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

namespace inference::fsr4 {

bool AppendCnbCt2d(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                   DispatchPlan& plan, size_t input, size_t weights, size_t& output,
                   std::string& error) {
    (void)context; // PSOs are created through the supplied AMD extension interface.
    if (!plan.signature || input >= plan.buffers.size() || weights >= plan.buffers.size() ||
        input == weights || plan.buffers[input].bytes != kCnbCt2dInputBytes) {
        error = "unsupported CNB_CT2D input shape";
        return false;
    }
    std::vector<uint8_t> cso;
    if (!gpu::LoadShaderBytecode("m12/cnbct2d", cso, error))
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
    plan.convolutionStats.push_back(statsLine);
    const size_t destination = plan.buffers.size();
    plan.buffers.push_back({kCnbCt2dOutputBytes, nullptr});
    const size_t unused = plan.buffers.size();
    plan.buffers.push_back({4, nullptr});
    gpu::BufferPass stage{};
    stage.pso = pso.Get();
    stage.gridX = (kCnbCt2dInputPixels + 63) / 64;
    stage.gridY = 1;
    stage.srv0 = input;
    stage.srv1 = weights;
    stage.uav0 = destination;
    stage.uav1 = unused;
    plan.psos.push_back(pso);
    plan.passes.push_back(stage);
    output = destination;
    return true;
}

bool RunCnbCt2d(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                std::span<const uint8_t> input, std::span<const uint8_t> weights,
                std::vector<uint8_t>& output, std::string& isa, std::string& error) {
    output.clear();
    isa.clear();
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error))
        return false;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()}};
    size_t destination = 0;
    if (!AppendCnbCt2d(context, extensions, plan, 0, 1, destination, error))
        return false;
    isa = plan.convolutionIsa.front();
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes,
                                destination, {false}, snapshots, destination,
                                kCnbCt2dOutputBytes, output, error);
}

} // namespace inference::fsr4
