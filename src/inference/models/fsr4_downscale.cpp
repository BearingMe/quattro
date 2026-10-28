#include "fsr4_downscale.h"
#include "fsr4_graph.h"
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

bool AppendDownscale(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                      DispatchPlan& plan, uint32_t pass, size_t input, size_t weights,
                      size_t& output, std::string& error) {
    DownscaleGeometry shape{};
    if (!DownscaleGeometryForPass(pass, shape)) {
        error = "unsupported downscale convolution pass";
        return false;
    }
    if (!plan.signature || input >= plan.buffers.size() || weights >= plan.buffers.size() ||
        input == weights || plan.buffers[input].bytes != size_t(shape.inputPixels) * shape.inputChannels) {
        error = "unsupported downscale convolution input shape";
        return false;
    }
    if (shape.pixels % 16 != 0 || shape.features % 16 != 0 || shape.reduction % 16 != 0) {
        error = "downscale convolution shape is not WMMA tile aligned";
        return false;
    }

    const std::string prefix = "m12/ds" + std::to_string(pass);
    const size_t packedBytes =
        size_t(shape.pixels) * shape.reduction + size_t(shape.reduction) * shape.features + 1024;
    const size_t workspace = plan.buffers.size();
    plan.buffers.insert(plan.buffers.end(), {
        {packedBytes, nullptr},         // 2: [A: pixels*K | B: K*features | zero tile]
        {size_t(shape.pixels) * shape.features * 4, nullptr}, // 3: INT32 accumulators
        {size_t(shape.pixels) * shape.features, nullptr},      // 4: INT8 output
        {4, nullptr},                   // 5: bound-but-unused second UAV slot
    });
    const char* steps[] = {"pack", "raw", "post"};
    for (uint32_t step = 0; step < 3; ++step) {
        std::vector<uint8_t> cso;
        const std::string name = prefix + steps[step];
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
            error = "cannot create downscale convolution PSO: " + name;
            return false;
        }

        gpu::BufferPass stage{};
        stage.pso = pso.Get();
        stage.srv1 = weights;
        stage.uav1 = workspace + 3;
        if (step == 0) {
            stage.srv0 = input;
            stage.uav0 = workspace;
            const size_t words = packedBytes / 4;
            stage.gridX = 4096;
            stage.gridY = uint32_t((words + 4096 * 64 - 1) / (4096 * 64));
        } else if (step == 1) {
            stage.srv0 = workspace;
            stage.uav0 = workspace + 1;
            stage.gridX = shape.features / 16;
            stage.gridY = shape.pixels / 16;
        } else {
            stage.srv0 = workspace + 1;
            stage.uav0 = workspace + 2;
            stage.gridX = 4096;
            stage.gridY = uint32_t((shape.pixels + 4096 * 64 - 1) / (4096 * 64));
        }
        plan.psos.push_back(pso);
        plan.passes.push_back(stage);
    }

    output = workspace + 2;
    return true;
}

bool RunDownscaleConvolution(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                             uint32_t pass, std::span<const uint8_t> input,
                             std::span<const uint8_t> weights, std::vector<uint8_t>& output,
                             std::string& isa, std::string& error) {
    output.clear();
    isa.clear();
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error, true))
        return false;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()}};
    size_t destination = 0;
    if (!AppendDownscale(context, extensions, plan, pass, 0, 1, destination, error))
        return false;
    isa = plan.convolutionIsa.front();
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes, destination,
                                std::vector<bool>(plan.passes.size(), false), snapshots, destination,
                                plan.buffers[destination].bytes, output, error, true);
}

} // namespace inference::fsr4
