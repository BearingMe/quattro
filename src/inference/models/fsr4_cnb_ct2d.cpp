#include "fsr4_cnb_ct2d.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

namespace inference::fsr4 {

bool RunCnbCt2d(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                std::span<const uint8_t> input, std::span<const uint8_t> weights,
                std::vector<uint8_t>& output, std::string& isa, std::string& error) {
    output.clear();
    isa.clear();
    if (input.size() != kCnbCt2dInputBytes) {
        error = "unsupported CNB_CT2D input shape";
        return false;
    }
    if (!ValidateWeights(weights, error))
        return false;

    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(context, signature, error))
        return false;
    std::vector<uint8_t> cso;
    if (!gpu::LoadShaderBytecode("m12/cnbct2d", cso, error))
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

    const std::vector<gpu::BufferInit> buffers = {
        {input.size(), input.data()},     // 0: source tensor
        {weights.size(), weights.data()}, // 1: checked payload
        {kCnbCt2dOutputBytes, nullptr},   // 2: FP16 network output
        {4, nullptr},                     // 3: bound-but-unused second UAV slot
    };
    gpu::BufferPass stage{};
    stage.pso = pso.Get();
    stage.gridX = (kCnbCt2dInputPixels + 63) / 64;
    stage.gridY = 1;
    stage.srv0 = 0;
    stage.srv1 = 1;
    stage.uav0 = 2;
    stage.uav1 = 3;
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, signature, buffers, {stage}, 3, {false}, snapshots, 2,
                                kCnbCt2dOutputBytes, output, error);
}

} // namespace inference::fsr4
