#include "fsr4_pass0.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

namespace inference::fsr4 {

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
    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(context, signature, error))
        return false;
    std::vector<uint8_t> bytecode;
    if (!gpu::LoadShaderBytecode("m12/pass0", bytecode, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {bytecode.data(), bytecode.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        error = "cannot create FSR4 input convolution PSO";
        return false;
    }
    const std::vector<gpu::BufferInit> buffers = {
        {input.size(), input.data()}, {weights.size(), weights.data()},
        {outputBytes, nullptr}, {4, nullptr},
    };
    gpu::BufferPass pass{};
    pass.pso = pso.Get();
    pass.gridX = (960 * 540 + 63) / 64;
    pass.srv0 = 0;
    pass.srv1 = 1;
    pass.uav0 = 2;
    pass.uav1 = 3;
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, signature, buffers, {pass}, 3, {false},
                                snapshots, 2, outputBytes, output, error);
}

} // namespace inference::fsr4
