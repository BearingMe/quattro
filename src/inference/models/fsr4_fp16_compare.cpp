#include "fsr4_graph.h"
#include "gpu/compute.h"

namespace inference::fsr4 {

bool UseFp16MatrixComparison(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                             DispatchPlan& plan, std::vector<size_t>& boundaries,
                             std::vector<std::string>& stats, std::string& error) {
    if (boundaries.size() != 14 || plan.passes.size() != 42) {
        error = "FP16 comparison expects the complete fixed 42-dispatch graph";
        return false;
    }
    std::vector<gpu::BufferPass> expanded;
    std::vector<size_t> newBoundaries;
    size_t first = 0;
    uint32_t replacements = 0;
    for (uint32_t pass = 0; pass < 14; ++pass) {
        const bool matrix = pass == 1 || pass == 2 || pass == 3 || pass == 6 || pass == 12;
        for (size_t stage = first; stage <= boundaries[pass]; ++stage) {
            auto raw = plan.passes[stage];
            if (!matrix || (stage - first) % 3 != 1) {
                expanded.push_back(raw);
                continue;
            }
            const uint32_t conv = uint32_t((stage - first) / 3);
            const uint32_t m = pass == 3 ? 129600 : pass == 6 ? 32400 : 518400;
            const uint32_t k = pass == 3 ? 64 : pass == 6 ? 128 : conv == 0 ? 144 : conv == 1 ? 16 : 32;
            const uint32_t n = pass == 3 ? 32 : pass == 6 ? 64 : conv == 1 ? 32 : 16;
            const std::string key = pass == 3 || pass == 6 ? "ds" + std::to_string(pass) : "cnb" + std::to_string(conv);
            const size_t packetBytes = size_t(m) * k + size_t(k) * n + 1024;
            const size_t converted = plan.buffers.size();
            plan.buffers.push_back({packetBytes * 2, nullptr});
            std::vector<uint8_t> cso;
            gpu::ComPtr<ID3D12PipelineState> conversion, product;
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = plan.signature.Get();
            if (!gpu::LoadShaderBytecode(("m15/" + key + "-convert").c_str(), cso, error)) return false;
            desc.CS = {cso.data(), cso.size()};
            if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&conversion)))) {
                error = "FP16 comparison packet-conversion PSO failed"; return false;
            }
            if (!gpu::LoadShaderBytecode(("m15/" + key + "-raw").c_str(), cso, error)) return false;
            desc.CS = {cso.data(), cso.size()};
            AmdExtD3DPipelineHandle handle{};
            AmdExtD3DComputeShaderStats rawStats{};
            if (!gpu::CreateAnalyzedComputePipeline(extensions, desc, product, handle, rawStats, error)) {
                error = "FP16 " + key + " PSO: " + error; return false;
            }
            std::string isa, line;
            if (!gpu::GetPipelineIsa(extensions, handle, isa, line, error)) return false;
            if (isa.find("v_wmma_f32_16x16x16_f16") == std::string::npos) {
                error = "FP16 comparison did not emit native FP16 WMMA"; return false;
            }
            stats.push_back("pass" + std::to_string(pass) + ".c" + std::to_string(conv) + ": " + line + "\n" + isa);
            gpu::BufferPass convert = raw;
            convert.pso = conversion.Get();
            convert.uav0 = converted;
            convert.gridX = 4096;
            convert.gridY = uint32_t((packetBytes / 4 + 4096 * 64 - 1) / (4096 * 64));
            raw.pso = product.Get();
            raw.srv0 = converted;
            expanded.push_back(convert);
            expanded.push_back(raw);
            plan.psos.push_back(conversion);
            plan.psos.push_back(product);
            ++replacements;
        }
        first = boundaries[pass] + 1;
        newBoundaries.push_back(expanded.size() - 1);
    }
    plan.passes = std::move(expanded);
    boundaries = std::move(newBoundaries);
    if (replacements != 11) {
        error = "FP16 comparison replaced an unexpected number of matrix stages";
        return false;
    }
    return true;
}

} // namespace inference::fsr4
