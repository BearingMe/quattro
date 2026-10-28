// M12 incremental reproduction checks. These are deliberately separate from
// the reference-only fsr4-oracle experiment and do not claim the full M12 gate.
#include "experiments.h"
#include "inference/models/fsr4_pass0.h"
#include "inference/models/fsr4_cnb.h"
#include "inference/models/fsr4_downscale.h"
#include "inference/models/fsr4_fnb.h"
#include "inference/models/fsr4_cnb_ct2d.h"
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

// The checked weight payload starts with the pinned initializer buffer
// verbatim; scripts/m12_pack_model.py appends the embedded arrays after it.
// Reference passes that read AMD's `InitializerBuffer` bind this prefix.
constexpr size_t kInitializerBytes = 89216;

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

// Isolated execution of one shipped model pass over a synthetic scratch image.
// `inputOffset`/`outputOffset` follow the pass's own descriptor; the caller
// supplies the tensor it wants compared.
bool ReferenceIsolatedPass(gpu::DeviceContext& ctx, const std::string& shaderName,
                           size_t inputOffset, size_t outputOffset, uint32_t gridX, uint32_t gridY,
                           const std::vector<uint8_t>& input,
                           std::span<const uint8_t> initializers, size_t outputBytes,
                           std::vector<uint8_t>& output, std::string& error,
                           std::span<const uint8_t> skip = {}, size_t skipOffset = 0);

bool ReferenceDownscale(gpu::DeviceContext& ctx, uint32_t index,
                        const std::vector<uint8_t>& input,
                        std::span<const uint8_t> initializers, size_t outputBytes,
                        std::vector<uint8_t>& output, std::string& error) {
    // Pass 3 reads scratch offset 0 and writes 8,294,400; pass 6 reads
    // 8,294,400 and writes 12,441,600. Neither overlaps the other's extent.
    const size_t inputOffset = index == 3 ? 0 : 8294400;
    const size_t outputOffset = index == 3 ? 8294400 : 12441600;
    const uint32_t gridX = index == 3 ? 8 : 4;
    const uint32_t gridY = index == 3 ? 270 : 135;
    return ReferenceIsolatedPass(ctx, "m12ref/pass" + std::to_string(index), inputOffset,
                                 outputOffset, gridX, gridY, input, initializers, outputBytes,
                                 output, error);
}

bool ReferenceIsolatedPass(gpu::DeviceContext& ctx, const std::string& shaderName,
                           size_t inputOffset, size_t outputOffset, uint32_t gridX, uint32_t gridY,
                           const std::vector<uint8_t>& input,
                           std::span<const uint8_t> initializers, size_t outputBytes,
                           std::vector<uint8_t>& output, std::string& error,
                           std::span<const uint8_t> skip, size_t skipOffset) {
    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(ctx, signature, error))
        return false;
    std::vector<uint8_t> cso;
    const std::string& name = shaderName;
    if (!gpu::LoadShaderBytecode(name.c_str(), cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    if (FAILED(ctx.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        error = "cannot create isolated reference downscale PSO";
        return false;
    }
    // t1 is the model's `InitializerBuffer`: passes 6/7/8 take their weights
    // from there, so a dummy binding faults the GPU instead of returning
    // garbage. Passes whose weights are embedded constants ignore it.
    std::vector<uint8_t> scratch(20736000 + 1024, 0);
    memcpy(scratch.data() + inputOffset, input.data(), input.size());
    if (!skip.empty())
        memcpy(scratch.data() + skipOffset, skip.data(), skip.size());
    const std::vector<gpu::BufferInit> buffers = {
        {4, nullptr}, {initializers.size(), initializers.data()}, {4, nullptr},
        {scratch.size(), scratch.data()},
    };
    gpu::BufferPass pass{};
    pass.pso = pso.Get();
    pass.gridX = gridX;
    pass.gridY = gridY;
    pass.srv0 = 0;
    pass.srv1 = 1;
    pass.uav0 = 2;
    pass.uav1 = 3;
    std::vector<std::vector<uint8_t>> snapshots;
    std::vector<uint8_t> raw;
    if (!gpu::RunBufferPasses(ctx, signature, buffers, {pass}, 3, {false}, snapshots,
                              3, outputOffset + outputBytes, raw, error))
        return false;
    output.assign(raw.begin() + outputOffset, raw.begin() + outputOffset + outputBytes);
    return true;
}

// Pass 13 writes the network's FP16 output to u0 rather than to the scratch
// image, so it needs its own isolated-reference path.
bool ReferenceCnbCt2d(gpu::DeviceContext& ctx, std::span<const uint8_t> initializers,
                      const std::vector<uint8_t>& input, std::vector<uint8_t>& output,
                      std::string& error) {
    gpu::ComPtr<ID3D12RootSignature> signature;
    if (!gpu::CreateBufferPassRootSignature(ctx, signature, error))
        return false;
    std::vector<uint8_t> cso;
    if (!gpu::LoadShaderBytecode("m12ref/pass13", cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {cso.data(), cso.size()};
    gpu::ComPtr<ID3D12PipelineState> pso;
    if (FAILED(ctx.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        error = "cannot create isolated reference CNB_CT2D PSO";
        return false;
    }
    std::vector<uint8_t> scratch(20736000 + 1024, 0);
    memcpy(scratch.data(), input.data(), input.size());
    // The shipped store is unpredicated and writes 8 rows past 1080
    // (240 KiB); reserve that documented tail so the reference is executed
    // unmodified, and read back only the logical extent.
    constexpr size_t kOutputTail = size_t(8) * 1920 * 8 * 2;
    const std::vector<gpu::BufferInit> buffers = {
        {4, nullptr}, {initializers.size(), initializers.data()},
        {inference::fsr4::kCnbCt2dOutputBytes + kOutputTail, nullptr},
        {scratch.size(), scratch.data()},
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
    return gpu::RunBufferPasses(ctx, signature, buffers, {pass}, 3, {false}, snapshots, 2,
                                inference::fsr4::kCnbCt2dOutputBytes, output, error);
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
    const std::span<const uint8_t> initializers = std::span<const uint8_t>(weights)
                                                     .first(kInitializerBytes);
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
    // Pass 13 (CNB_CT2D): the only pass that writes the network's FP16 output
    // to u0 instead of the scratch, so it gets its own reference path.
    {
        const auto pass13Dir = RepositoryRoot() / "build/reports/m12/cnb-ct2d";
        std::filesystem::create_directories(pass13Dir);
        if (!SaveNumericIsa(ctx, ext, "m12ref/pass13", pass13Dir / "reference-pass13.isa",
                            false, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        for (uint32_t fixture = 0; fixture < 2; ++fixture) {
            std::vector<uint8_t> input(inference::fsr4::kCnbCt2dInputBytes, 0);
            uint32_t seed = 0x5EED4F12u;
            if (fixture == 1) {
                for (uint8_t& value : input) {
                    seed = seed * 1664525u + 1013904223u;
                    value = uint8_t(seed >> 24);
                }
            }
            std::vector<uint8_t> output, repeated, reference;
            std::string passIsa, repeatedIsa;
            const bool ownOk = RunCnbCt2d(ctx, ext, input, weights, output, passIsa, error);
            const bool repeatOk = ownOk &&
                RunCnbCt2d(ctx, ext, input, weights, repeated, repeatedIsa, error);
            const bool referenceOk = repeatOk &&
                ReferenceCnbCt2d(ctx, initializers, input, reference, error);
            if (!ownOk || !repeatOk || !referenceOk) {
                std::fprintf(stderr, "pass13 fixture %u: %s\n", fixture, error.c_str());
                return 1;
            }
            if (const char* dump = std::getenv("M12_DUMP"); dump && *dump == '1') {
                const std::string stem = "pass13-fixture" + std::to_string(fixture);
                for (const auto& [suffix, data] :
                     {std::pair<const char*, const std::vector<uint8_t>*>{"-input.bin", &input},
                      {"-own.bin", &output},
                      {"-ref.bin", &reference}}) {
                    std::ofstream file(pass13Dir / (stem + suffix), std::ios::binary);
                    file.write(reinterpret_cast<const char*>(data->data()),
                               std::streamsize(data->size()));
                    if (!file)
                        return 1;
                }
            }
            size_t different = 0;
            double maxDelta = 0.0;
            for (size_t i = 0; i + 1 < output.size(); i += 2) {
                uint16_t a = 0, b = 0;
                memcpy(&a, output.data() + i, 2);
                memcpy(&b, reference.data() + i, 2);
                if (a != b) {
                    ++different;
                    maxDelta = std::max(maxDelta,
                                        double(std::abs(core::F16ToF32(a) - core::F16ToF32(b))));
                }
            }
            const bool deterministic = output == repeated;
            std::printf("CNB_CT2D pass13 fixture%u: %zu/%zu halfword mismatches | "
                        "max |delta| %g | deterministic %s\n",
                        fixture, different, output.size() / 2, maxDelta,
                        deterministic ? "yes" : "NO");
            ok = ok && different == 0 && deterministic;
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

int RunFsr4Downscale() {
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
    const auto directory = RepositoryRoot() / "build/reports/m12/downscale";
    std::filesystem::create_directories(directory);
    bool ok = true;
    for (uint32_t pass : {3u, 6u}) {
        DownscaleGeometry shape{};
        if (!DownscaleGeometryForPass(pass, shape)) {
            std::fprintf(stderr, "no downscale geometry for pass %u\n", pass);
            return 1;
        }
        // The shipped pass's own ISA is the numeric-oracle evidence: it shows
        // the exact float sequence its requantization epilogue lowers to.
        if (!SaveNumericIsa(ctx, ext, "m12ref/pass" + std::to_string(pass),
                            directory / ("reference-pass" + std::to_string(pass) + ".isa"),
                            false, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        const size_t inputBytes = size_t(shape.inputPixels) * shape.inputChannels;
        const size_t outputBytes = size_t(shape.pixels) * shape.features;
        std::string passIsa;
        for (uint32_t fixture = 0; fixture < 2; ++fixture) {
            // Fixture 1 covers the full signed INT8 range, fixture 0 the
            // all-zero case; both are fed byte-identically to our pass and to
            // the isolated shipped pass.
            std::vector<uint8_t> input(inputBytes, 0);
            uint32_t seed = 0x5EED4F12u;
            if (fixture == 1) {
                for (uint8_t& value : input) {
                    seed = seed * 1664525u + 1013904223u;
                    value = uint8_t(seed >> 24);
                }
            }
            std::vector<uint8_t> output, repeated, reference;
            std::string repeatedIsa;
            const std::span<const uint8_t> initializers = std::span<const uint8_t>(weights)
                                                             .first(kInitializerBytes);
            const bool ownOk =
                RunDownscaleConvolution(ctx, ext, pass, input, weights, output, passIsa, error);
            const bool repeatOk = ownOk &&
                RunDownscaleConvolution(ctx, ext, pass, input, weights, repeated, repeatedIsa, error);
            const bool referenceOk = repeatOk &&
                ReferenceDownscale(ctx, pass, input, initializers, outputBytes, reference, error);
            if (!ownOk || !repeatOk || !referenceOk) {
                std::fprintf(stderr, "downscale pass %u fixture %u: %s\n", pass, fixture,
                             error.c_str());
                return 1;
            }
            // Optional captures (M12_DUMP=1), same convention as the reference
            // oracle: explicit dumps only, always under build/reports/m12/.
            if (const char* dump = std::getenv("M12_DUMP"); dump && *dump == '1') {
                const std::string stem =
                    "pass" + std::to_string(pass) + "-fixture" + std::to_string(fixture);
                for (const auto& [suffix, data] :
                     {std::pair<const char*, const std::vector<uint8_t>*>{"-input.bin", &input},
                      {"-own.bin", &output},
                      {"-ref.bin", &reference}}) {
                    std::ofstream file(directory / (stem + suffix), std::ios::binary);
                    file.write(reinterpret_cast<const char*>(data->data()),
                               std::streamsize(data->size()));
                    if (!file)
                        return 1;
                }
            }
            size_t different = 0;
            int maxDelta = 0;
            for (size_t i = 0; i < output.size(); ++i) {
                if (output[i] != reference[i]) {
                    ++different;
                    maxDelta = std::max(
                        maxDelta, std::abs(int(int8_t(output[i])) - int(int8_t(reference[i]))));
                }
            }
            const bool deterministic = output == repeated;
            const bool native = passIsa.find("v_wmma_i32_16x16x16_iu8") != std::string::npos;
            std::printf("Downscale pass%u fixture%u: %zu/%zu mismatches | max delta %d | "
                        "deterministic %s | native WMMA %s\n",
                        pass, fixture, different, output.size(), maxDelta,
                        deterministic ? "yes" : "NO", native ? "yes" : "NO");
            ok = ok && different == 0 && deterministic && native;
            for (size_t i = 0, shown = 0; i < output.size() && shown < 12; ++i) {
                if (output[i] == reference[i])
                    continue;
                const size_t pixel = i / shape.features;
                std::printf("  mismatch pixel(%zu,%zu) c%zu: own %d reference %d\n",
                            pixel % shape.width, pixel / shape.width, i % shape.features,
                            int(int8_t(output[i])), int(int8_t(reference[i])));
                ++shown;
            }
        }
        std::ofstream file(directory / ("pass" + std::to_string(pass) + ".isa"), std::ios::binary);
        file << passIsa;
        if (!file)
            return 1;
    }
    std::string queueErrors;
    size_t intrinsicMessages = 0;
    if (!gpu::InfoQueueCleanWithFilter(ctx, gpu::IsBufferPassWmmaDiagnostic, intrinsicMessages,
                                       queueErrors)) {
        std::fprintf(stderr, "%s\n", queueErrors.c_str());
        return 1;
    }
    std::printf("Downscale convolutions: %s | exact named root4 intrinsic messages %zu; "
                "tensor errors fatal\n",
                ok ? "PASS" : "FAIL", intrinsicMessages);
    return ok ? 0 : 1;
}

int RunFsr4Fnb() {
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
    const auto directory = RepositoryRoot() / "build/reports/m12/fnb";
    std::filesystem::create_directories(directory);
    const std::span<const uint8_t> initializers = std::span<const uint8_t>(weights)
                                                     .first(kInitializerBytes);
    bool ok = true;
    for (uint32_t pass : {4u, 5u, 7u, 8u, 9u, 10u, 11u}) {
        FnbGeometry shape{};
        if (!FnbGeometryForPass(pass, shape)) {
            std::fprintf(stderr, "no FasterNetBlock geometry for pass %u\n", pass);
            return 1;
        }
        // Pass 11's shipped 32/1 overload stores without a logical-size
        // predicate and corrupts its own output band; the reproduction
        // predicates, so it is compared against the labelled causal control.
        const std::string referenceShader =
            pass == 11 ? "m12control/pass11" : "m12ref/pass" + std::to_string(pass);
        if (!SaveNumericIsa(ctx, ext, referenceShader,
                            directory / ("reference-pass" + std::to_string(pass) + ".isa"),
                            false, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        const size_t tensorBytes = size_t(shape.pixels) * shape.channels;
        const size_t outBytes = shape.outFeatures == 0
            ? tensorBytes
            : size_t(shape.outWidth) * shape.outHeight * shape.outFeatures;
        std::string passIsa;
        for (uint32_t fixture = 0; fixture < 2; ++fixture) {
            std::vector<uint8_t> input(tensorBytes, 0);
            std::vector<uint8_t> skip(outBytes, 0);
            uint32_t seed = 0x5EED4F12u;
            uint32_t skipSeed = 0x5EED4F13u;
            if (fixture == 1) {
                for (uint8_t& value : input) {
                    seed = seed * 1664525u + 1013904223u;
                    value = uint8_t(seed >> 24);
                }
                for (uint8_t& value : skip) {
                    skipSeed = skipSeed * 1664525u + 1013904223u;
                    value = uint8_t(skipSeed >> 24);
                }
            }
            std::vector<uint8_t> output, repeated, reference;
            std::string repeatedIsa;
            const bool ownOk =
                RunFasterNetBlock(ctx, ext, pass, input, skip, weights, output, passIsa, error);
            const bool repeatOk = ownOk &&
                RunFasterNetBlock(ctx, ext, pass, input, skip, weights, repeated, repeatedIsa,
                                  error);
            const bool referenceOk = repeatOk && ReferenceIsolatedPass(
                ctx, referenceShader, shape.inputOffset, shape.outputOffset, shape.gridX,
                shape.gridY, input, initializers, outBytes, reference, error,
                skip, shape.skipOffset);
            if (!ownOk || !repeatOk || !referenceOk) {
                std::fprintf(stderr, "fnb pass %u fixture %u: %s\n", pass, fixture, error.c_str());
                return 1;
            }
            if (const char* dump = std::getenv("M12_DUMP"); dump && *dump == '1') {
                const std::string stem =
                    "pass" + std::to_string(pass) + "-fixture" + std::to_string(fixture);
                for (const auto& [suffix, data] :
                     {std::pair<const char*, const std::vector<uint8_t>*>{"-input.bin", &input},
                      {"-own.bin", &output},
                      {"-ref.bin", &reference}}) {
                    std::ofstream file(directory / (stem + suffix), std::ios::binary);
                    file.write(reinterpret_cast<const char*>(data->data()),
                               std::streamsize(data->size()));
                    if (!file)
                        return 1;
                }
            }
            size_t different = 0;
            int maxDelta = 0;
            for (size_t i = 0; i < output.size(); ++i) {
                if (output[i] != reference[i]) {
                    ++different;
                    maxDelta = std::max(
                        maxDelta, std::abs(int(int8_t(output[i])) - int(int8_t(reference[i]))));
                }
            }
            const bool deterministic = output == repeated;
            const bool dp4a = passIsa.find("v_dot4_i32_iu8") != std::string::npos;
            std::printf("FasterNet pass%u fixture%u: %zu/%zu mismatches | max delta %d | "
                        "deterministic %s | DP4A %s\n",
                        pass, fixture, different, output.size(), maxDelta,
                        deterministic ? "yes" : "NO", dp4a ? "yes" : "NO");
            ok = ok && different == 0 && deterministic && dp4a;
            for (size_t i = 0, shown = 0; i < output.size() && shown < 8; ++i) {
                if (output[i] == reference[i])
                    continue;
                const size_t pixel = i / shape.channels;
                std::printf("  mismatch pixel(%zu,%zu) c%zu: own %d reference %d\n",
                            pixel % shape.width, pixel / shape.width, i % shape.channels,
                            int(int8_t(output[i])), int(int8_t(reference[i])));
                ++shown;
            }
        }
        std::ofstream file(directory / ("pass" + std::to_string(pass) + ".isa"), std::ios::binary);
        file << passIsa;
        if (!file)
            return 1;
    }
    std::string queueErrors;
    size_t intrinsicMessages = 0;
    if (!gpu::InfoQueueCleanWithFilter(ctx, gpu::IsBufferPassWmmaDiagnostic, intrinsicMessages,
                                       queueErrors)) {
        std::fprintf(stderr, "%s\n", queueErrors.c_str());
        return 1;
    }
    std::printf("FasterNetBlocks: %s | exact named root4 intrinsic messages %zu\n",
                ok ? "PASS" : "FAIL", intrinsicMessages);
    return ok ? 0 : 1;
}

} // namespace experiments




