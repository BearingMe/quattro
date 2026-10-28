#include "fsr4_downscale.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

namespace inference::fsr4 {

bool DownscaleGeometryForPass(uint32_t pass, DownscaleGeometry& geometry) {
    if (pass == 3) {
        // 960x540x16 INT8 -> 480x270x32 INT8
        geometry = {960 * 540, 16, 480 * 270, 32, 64, 480};
        return true;
    }
    if (pass == 6) {
        // 480x270x32 INT8 -> 240x135x64 INT8
        geometry = {480 * 270, 32, 240 * 135, 64, 128, 240};
        return true;
    }
    return false;
}

bool RunDownscaleConvolution(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                             uint32_t pass, std::span<const uint8_t> input,
                             std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                             std::string& isa, std::string& error) {
    output.clear();
    isa.clear();
    DownscaleGeometry shape{};
    if (!DownscaleGeometryForPass(pass, shape)) {
        error = "unsupported downscale convolution pass";
        return false;
    }
    if (input.size() != size_t(shape.inputPixels) * shape.inputChannels) {
        error = "unsupported downscale convolution input shape";
        return false;
    }
    if (!ValidateWeights(weights, error))
        return false;
    if (shape.pixels % 16 != 0 || shape.features % 16 != 0 || shape.reduction % 16 != 0) {
        error = "downscale convolution shape is not WMMA tile aligned";
        return false;
    }

    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(context, signature, error, true))
        return false;

    const std::string prefix = "m12/ds" + std::to_string(pass);
    const size_t packedBytes =
        size_t(shape.pixels) * shape.reduction + size_t(shape.reduction) * shape.features + 1024;
    const std::vector<gpu::BufferInit> buffers = {
        {input.size(), input.data()},   // 0: source tensor
        {weights.size(), weights.data()}, // 1: checked payload (weights + FP16 bias)
        {packedBytes, nullptr},         // 2: [A: pixels*K | B: K*features | zero tile]
        {size_t(shape.pixels) * shape.features * 4, nullptr}, // 3: INT32 accumulators
        {size_t(shape.pixels) * shape.features, nullptr},      // 4: INT8 output
        {4, nullptr},                   // 5: bound-but-unused second UAV slot
    };

    std::vector<gpu::ComPtr<ID3D12PipelineState>> psos;
    std::vector<gpu::BufferPass> stages;
    const char* steps[] = {"pack", "raw", "post"};
    for (uint32_t step = 0; step < 3; ++step) {
        std::vector<uint8_t> cso;
        const std::string name = prefix + steps[step];
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
            std::string statsLine;
            if (!gpu::GetPipelineIsa(extensions, handle, isa, statsLine, error))
                return false;
        } else if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
            error = "cannot create downscale convolution PSO: " + name;
            return false;
        }

        gpu::BufferPass stage{};
        stage.pso = pso.Get();
        stage.srv1 = 1;
        stage.uav1 = 5;
        if (step == 0) {
            stage.srv0 = 0;
            stage.uav0 = 2;
            const size_t words = packedBytes / 4;
            stage.gridX = 4096;
            stage.gridY = uint32_t((words + 4096 * 64 - 1) / (4096 * 64));
        } else if (step == 1) {
            stage.srv0 = 2;
            stage.uav0 = 3;
            stage.gridX = shape.features / 16;
            stage.gridY = shape.pixels / 16;
        } else {
            stage.srv0 = 3;
            stage.uav0 = 4;
            stage.gridX = 4096;
            stage.gridY = uint32_t((shape.pixels + 4096 * 64 - 1) / (4096 * 64));
        }
        psos.push_back(pso);
        stages.push_back(stage);
    }

    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, signature, buffers, stages, 5,
                                std::vector<bool>(stages.size(), false), snapshots, 4,
                                size_t(shape.pixels) * shape.features, output, error, true);
}

} // namespace inference::fsr4
