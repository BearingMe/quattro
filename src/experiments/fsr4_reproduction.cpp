// M12 incremental reproduction checks. These are deliberately separate from
// the reference-only fsr4-oracle experiment and do not claim the full M12 gate.
#include "experiments.h"
#include "inference/models/fsr4_pass0.h"
#include "inference/models/fsr4_cnb.h"
#include "inference/models/fsr4_weights.h"
#include "core/numerics.h"
#include "gpu/compute.h"
#include "gpu/graph.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace experiments {
namespace {

std::filesystem::path RepositoryRoot() {
    wchar_t path[MAX_PATH]{};
    const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (length == 0 || length == MAX_PATH)
        return {};
    return std::filesystem::path(path).parent_path().parent_path().parent_path();
}

bool ReferencePass0(gpu::DeviceContext& ctx, const std::vector<uint8_t>& input,
                    std::vector<uint8_t>& output, std::string& error) {
    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(ctx, signature, error))
        return false;
    std::vector<uint8_t> cso;
    if (!gpu::LoadShaderBytecode("m12ref/pass0", cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    if (FAILED(ctx.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        error = "cannot create reference pass0 PSO";
        return false;
    }
    // Reference pass0 reads t0 and writes u1. Its rounded 68-row group grid
    // can write beyond the logical 540 rows, so retain explicit tail storage.
    const std::vector<gpu::BufferInit> buffers = {
        {input.size(), input.data()}, {4, nullptr}, {4, nullptr},
        {size_t(960) * 544 * 16, nullptr},
    };
    gpu::BufferPass pass{};
    pass.pso = pso.Get();
    pass.gridX = 120;
    pass.gridY = 68;
    pass.srv0 = 0;
    pass.srv1 = 1;
    pass.uav0 = 2;
    pass.uav1 = 3;
    std::vector<std::vector<uint8_t>> snapshots;
    return gpu::RunBufferPasses(ctx, signature, buffers, {pass}, 3, {false}, snapshots,
                                3, size_t(960) * 540 * 16, output, error);
}

bool ReferenceCnb(gpu::DeviceContext& ctx, uint32_t index, const std::vector<uint8_t>& input,
                   std::vector<uint8_t>& output, std::string& error) {
    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(ctx, signature, error))
        return false;
    std::vector<uint8_t> cso;
    const std::string name = "m12ref/pass" + std::to_string(index);
    if (!gpu::LoadShaderBytecode(name.c_str(), cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    if (FAILED(ctx.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        error = "cannot create isolated reference ConvNextBlock PSO";
        return false;
    }
    const size_t inputOffset = index == 1 ? 0 : index == 2 ? 8294400 : 12441600;
    const size_t outputOffset = index == 1 ? 8294400 : 0;
    std::vector<uint8_t> scratch(20736000 + 1024, 0);
    memcpy(scratch.data() + inputOffset, input.data(), input.size());
    const std::vector<gpu::BufferInit> buffers = {
        {4, nullptr}, {4, nullptr}, {4, nullptr}, {scratch.size(), scratch.data()},
    };
    gpu::BufferPass pass{};
    pass.pso = pso.Get();
    pass.gridX = 15;
    pass.gridY = 540;
    pass.srv0 = 0;
    pass.srv1 = 1;
    pass.uav0 = 2;
    pass.uav1 = 3;
    std::vector<std::vector<uint8_t>> snapshots;
    std::vector<uint8_t> raw;
    if (!gpu::RunBufferPasses(ctx, signature, buffers, {pass}, 3, {false}, snapshots,
                              3, outputOffset + input.size(), raw, error))
        return false;
    output.assign(raw.begin() + outputOffset, raw.end());
    return true;
}

bool SaveNumericIsa(gpu::DeviceContext& ctx, gpu::AmdExtensions& ext,
                    const std::string& shader, const std::filesystem::path& file,
                    bool wmmaSignature, std::string& error) {
    gpu::ComPtr<ID3D12RootSignature> signature;
    std::vector<uint8_t> cso;
    if (!gpu::CreateBufferPassRootSignature(ctx, signature, error, wmmaSignature) ||
        !gpu::LoadShaderBytecode(shader.c_str(), cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    AmdExtD3DPipelineHandle handle{};
    AmdExtD3DComputeShaderStats stats{};
    if (!gpu::CreateAnalyzedComputePipeline(ext, desc, pso, handle, stats, error))
        return false;
    std::string isa, statsLine;
    if (!gpu::GetPipelineIsa(ext, handle, isa, statsLine, error))
        return false;
    std::ofstream out(file, std::ios::binary);
    out << isa;
    if (!out) {
        error = "cannot write numeric ISA report";
        return false;
    }
    return true;
}

} // namespace

int RunFsr4Weights() {
    using namespace inference::fsr4;
    std::vector<uint8_t> weights;
    std::string error;
    if (!LoadWeights(RepositoryRoot() / "build/reports/m12/weights.bin", weights, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (WeightTable().size() != 78 || ScalarTable().size() != 14)
        return 1;
    auto corrupted = weights;
    corrupted[corrupted.size() / 2] ^= 1;
    if (ValidateWeights(corrupted, error) || error.empty())
        return 1;
    corrupted = weights;
    corrupted.pop_back();
    if (ValidateWeights(corrupted, error) || error.empty())
        return 1;
    corrupted = weights;
    corrupted.push_back(0);
    if (ValidateWeights(corrupted, error) || error.empty())
        return 1;
    std::vector<uint8_t> missing{1, 2, 3};
    if (LoadWeights(RepositoryRoot() / "build/reports/m12/does-not-exist.bin", missing, error) ||
        !missing.empty() || error.empty())
        return 1;
    std::printf("FSR4 weights: PASS | %zu tensors | %zu B | sha256 %s\n",
                WeightTable().size(), weights.size(), WeightPayloadSha256());
    std::printf("corrupt/truncated/extra/missing payloads: rejected\n");
    return 0;
}

int RunFsr4Pass0() {
    using namespace inference::fsr4;
    std::vector<uint8_t> weights;
    std::string error;
    if (!LoadWeights(RepositoryRoot() / "build/reports/m12/weights.bin", weights, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    gpu::DeviceContext ctx;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("adapter: %ls\n", ctx.adapter.name.c_str());
    bool allOk = true;
    for (uint32_t fixture = 0; fixture < 3; ++fixture) {
        std::vector<uint8_t> paddedInput(size_t(1920) * 1088 * 16, 0);
        uint32_t seed = 0x5EED4F12u;
        for (size_t pixel = 0; pixel < size_t(1920) * 1080; ++pixel) {
            for (uint32_t c = 0; c < 7; ++c) {
                seed = seed * 1664525u + 1013904223u;
                float value = 0.0f;
                if (fixture == 1 && pixel == size_t(540) * 1920 + 960)
                    value = c & 1 ? -1.0f : 1.0f;
                if (fixture == 2)
                    value = float(int((seed >> 16) % 4095u) - 2047) / 1024.0f;
                const uint16_t bits = core::F32ToF16(value);
                memcpy(paddedInput.data() + pixel * 16 + c * 2, &bits, 2);
            }
        }
        const std::span<const uint8_t> validInput(paddedInput.data(), size_t(1920) * 1080 * 16);
        std::vector<uint8_t> output, repeated, reference;
        if (!RunInputConvolution(ctx, validInput, weights, output, error) ||
            !RunInputConvolution(ctx, validInput, weights, repeated, error) ||
            !ReferencePass0(ctx, paddedInput, reference, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        size_t mismatches = 0;
        int maxDelta = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            if (output[i] != reference[i]) {
                ++mismatches;
                const int delta = std::abs(int(int8_t(output[i])) - int(int8_t(reference[i])));
                if (delta > maxDelta)
                    maxDelta = delta;
            }
        }
        const bool deterministic = output == repeated;
        std::printf("pass0 fixture %u: %zu/%zu mismatches | max INT8 delta %d | deterministic %s\n",
                    fixture, mismatches, output.size(), maxDelta, deterministic ? "yes" : "NO");
        allOk = allOk && mismatches == 0 && deterministic;
        if (fixture == 2) {
            std::ofstream file(RepositoryRoot() / "build/reports/m12/own-pass0.bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(output.data()), std::streamsize(output.size()));
            if (!file)
                return 1;
        }
    }
    std::string queueErrors;
    if (!gpu::InfoQueueClean(ctx, queueErrors)) {
        std::fprintf(stderr, "%s\n", queueErrors.c_str());
        return 1;
    }
    std::printf("FSR4 pass0: %s (only 1/14 passes; not the M12 gate)\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}

int RunFsr4Cnb() {
    using namespace inference::fsr4;
    // Regression for the review finding: root2 is the actual weight SRV,
    // not this signature's intrinsic dummy. Error filtering must fail closed.
    const std::string diagnostic = "GPU-BASED VALIDATION: Root descriptor access out of bounds "
        "QUATTRO_WMMA_INTRINSIC_DUMMY Root Descriptor Type: UAV Root Parameter Index: [4] "
        "Highest byte offset from view start accessed: [1342177324] "
        "Bytes available from view start based on remaining resource size: 16.";
    const auto recognized = [](const std::string& text) {
        D3D12_MESSAGE message{};
        message.ID = D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_ROOT_DESCRIPTOR_ACCESS_OUT_OF_BOUNDS;
        message.pDescription = text.c_str();
        return gpu::IsBufferPassWmmaDiagnostic(message);
    };
    if (!recognized(diagnostic))
        return 1;
    for (const auto& replacement : {std::pair{"Index: [4]", "Index: [2]"},
                                    std::pair{"Type: UAV", "Type: SRV"},
                                    std::pair{"QUATTRO_WMMA_INTRINSIC_DUMMY", "REAL_WEIGHTS"},
                                    std::pair{"[1342177324]", "[1024]"},
                                    std::pair{"size: 16.", "size: 136848."}}) {
        std::string changed = diagnostic;
        changed.replace(changed.find(replacement.first), std::strlen(replacement.first), replacement.second);
        if (recognized(changed))
            return 1;
    }
    std::vector<uint8_t> weights;
    std::string error;
    if (!LoadWeights(RepositoryRoot() / "build/reports/m12/weights.bin", weights, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    gpu::DeviceContext ctx;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    gpu::AmdExtensions ext;
    if (!gpu::LoadAmdExtensions(ctx, ext, error) || !ext.waveMatrixSupported) {
        std::fprintf(stderr, "native WMMA unavailable: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }
    struct ExtensionLifetime {
        gpu::AmdExtensions& value;
        ~ExtensionLifetime() { gpu::ReleaseAmdExtensions(value); }
    } lifetime{ext};
    const auto directory = RepositoryRoot() / "build/reports/m12/cnb";
    std::filesystem::create_directories(directory);
    bool ok = true;
    for (uint32_t pass : {1u, 2u, 12u}) {
        if (!SaveNumericIsa(ctx, ext, "m12ref/pass" + std::to_string(pass),
                            directory / ("reference-pass" + std::to_string(pass) + ".isa"), false, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        for (uint32_t conv = 0; conv < 3; ++conv) {
            if (!SaveNumericIsa(ctx, ext, "m12/p" + std::to_string(pass) + "c" + std::to_string(conv) + "post",
                                directory / ("post-pass" + std::to_string(pass) + "conv" + std::to_string(conv) + ".isa"), true, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
        }
        for (uint32_t fixture = 0; fixture < 2; ++fixture) {
            std::vector<uint8_t> input(size_t(960) * 540 * 16, 0);
            uint32_t seed = 0x5EED4F12u;
            if (fixture == 1) {
                for (uint8_t& value : input) {
                    seed = seed * 1664525u + 1013904223u;
                    value = uint8_t(seed >> 24); // covers all signed INT8 values
                }
            }
            std::vector<uint8_t> output, reference;
            std::vector<std::string> isa;
            if (!RunConvNextBlock(ctx, ext, pass, input, weights, output, isa, error) ||
                !ReferenceCnb(ctx, pass, input, reference, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
            size_t different = 0;
            int maxDelta = 0;
            for (size_t i = 0; i < output.size(); ++i) {
                if (output[i] != reference[i]) {
                    ++different;
                    maxDelta = std::max(maxDelta, std::abs(int(int8_t(output[i])) - int(int8_t(reference[i]))));
                }
            }
            bool native = isa.size() == 3;
            for (size_t conv = 0; conv < isa.size(); ++conv) {
                native = native && isa[conv].find("v_wmma_i32_16x16x16_iu8") != std::string::npos;
                std::ofstream file(directory / ("pass" + std::to_string(pass) + "conv" +
                                   std::to_string(conv) + ".isa"), std::ios::binary);
                file << isa[conv];
                if (!file)
                    return 1;
            }
            std::printf("ConvNext pass%u fixture%u: %zu/%zu mismatches | max delta %d | native WMMA %s\n",
                        pass, fixture, different, output.size(), maxDelta, native ? "yes" : "NO");
            ok = ok && different == 0 && native;
            if (fixture == 1) {
                std::vector<uint8_t> repeated;
                std::vector<std::string> repeatedIsa;
                if (!RunConvNextBlock(ctx, ext, pass, input, weights, repeated, repeatedIsa, error)) {
                    std::fprintf(stderr, "%s\n", error.c_str());
                    return 1;
                }
                const bool deterministic = output == repeated;
                std::printf("  repeated native execution: %s\n", deterministic ? "bit-identical" : "DIFFERENT");
                ok = ok && deterministic;
            }
            if (different != 0) {
                for (size_t i = 0, shown = 0; i < output.size() && shown < 12; ++i) {
                    if (output[i] != reference[i]) {
                        std::printf("  mismatch pixel(%zu,%zu) c%zu: own %d reference %d\n",
                                    (i / 16) % 960, (i / 16) / 960, i % 16,
                                    int(int8_t(output[i])), int(int8_t(reference[i])));
                        ++shown;
                    }
                }
            }
        }
    }
    std::string queueErrors;
    // A prior strict run retained the exact magic access at root4. Exempt
    // only that named dummy/size/address/ID pattern; root2 weights remain fatal.
    size_t intrinsicMessages = 0;
    if (!gpu::InfoQueueCleanWithFilter(ctx, gpu::IsBufferPassWmmaDiagnostic, intrinsicMessages, queueErrors)) {
        std::fprintf(stderr, "%s\n", queueErrors.c_str());
        return 1;
    }
    std::printf("ConvNextBlocks: %s | exact named root4 intrinsic messages %zu; tensor errors fatal\n",
                ok ? "PASS" : "FAIL", intrinsicMessages);
    return ok ? 0 : 1;
}

} // namespace experiments
