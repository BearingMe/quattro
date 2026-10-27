// M11 — standalone INT8 neural inference on gfx1101: the fixed two-layer
// network (EXP-009) executed end-to-end on the GPU with native WMMA GEMMs and
// requantization, GPU-resident intermediates, CPU comparison at every tensor,
// repeated execution and end-to-end GPU timing.

#include "experiments.h"

#include "../gpu/amdext.h"
#include "../gpu/compute.h"
#include "../gpu/device.h"
#include "../gpu/matmul.h"
#include "../gpu/sequence.h"
#include "../reference/fixed_network.h"
#include "core/tensor.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace experiments {
namespace {

std::filesystem::path ReportsDirM11() {
    wchar_t exePathW[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exePathW, MAX_PATH) == 0)
        return {};
    return std::filesystem::path(exePathW).parent_path() / ".." / "reports" / "m11";
}

bool SaveReport(const std::filesystem::path& dir, const std::string& name, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        return false;
    std::ofstream file(dir / name, std::ios::binary);
    if (!file)
        return false;
    file.write(content.data(), std::streamsize(content.size()));
    return bool(file);
}

// FNV-1a over the bytes — recorded so the exact staged data is checkable
// against the experiment doc without committing binaries.
uint64_t DataHash(const void* data, size_t bytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

size_t CountIsaLines(const std::string& isa, const char* needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = isa.find(needle, pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    return count;
}

} // namespace

int RunFixedNetworkGpu() {
    using namespace reference;
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool ok = true;
    std::string error;

    // --- C03: quantized weights, generated deterministically and validated
    // before any GPU submission. Fixed seed; recorded hashes below make the
    // staged data checkable against EXP-009. ---
    constexpr unsigned W = FixedNetworkWidth;
    static_assert(FixedNetworkElements == W * W);
    uint32_t seed = 0xC0FFEE11u;
    const auto nextI8 = [&seed](int lo, int hi) {
        seed = seed * 1664525u + 1013904223u;
        return int8_t(lo + int((seed >> 8) % uint32_t(hi - lo + 1)));
    };
    FixedNetworkBytes input{}, firstWeights{}, secondWeights{};
    for (unsigned i = 0; i < FixedNetworkElements; ++i) {
        input[i] = nextI8(-100, 100);
        firstWeights[i] = nextI8(-100, 100);
        secondWeights[i] = nextI8(-100, 100);
    }
    // Weight contract: exact sizes (static), values in the signed INT8 domain
    // (trivially true), and accumulation safety for the fixed K (enforced).
    if (!core::ValidateInt8Accumulation(W, 0, error)) {
        std::fprintf(stderr, "error: weight contract: %s\n", error.c_str());
        return 1;
    }
    std::printf("weights: 2 x %ux%u int8 (K-major), input %ux%u int8 | fnv1a in=%016llx w1=%016llx w2=%016llx\n",
                W, W, W, W, (unsigned long long)DataHash(input.data(), input.size()),
                (unsigned long long)DataHash(firstWeights.data(), firstWeights.size()),
                (unsigned long long)DataHash(secondWeights.data(), secondWeights.size()));

    // --- C02: CPU reference over the same staged data. ---
    const FixedNetworkReference ref = CpuFixedNetwork(input, firstWeights, secondWeights);

    // --- Device + extensions (analyzer needed for WMMA ISA evidence). ---
    gpu::DeviceContext ctx;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::printf("adapter: %ls (vid=0x%04X did=0x%04X)\n", ctx.adapter.name.c_str(),
                ctx.adapter.vendorId, ctx.adapter.deviceId);

    gpu::AmdExtensions ext;
    if (!gpu::LoadAmdExtensions(ctx, ext, error) || !error.empty()) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }
    if (!ext.analyzer) {
        std::fprintf(stderr, "error: shader analyzer unavailable - ISA evidence impossible\n");
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    gpu::ComPtr<ID3D12RootSignature> rootSignature;
    if (!gpu::CreateGemmTileRootSignature(ctx, rootSignature, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    // Analyzer PSOs (dispatchable, M09 pattern) + ISA evidence.
    const auto LoadKernelPso = [&](const char* csoName, bool requireWmma)
        -> gpu::ComPtr<ID3D12PipelineState> {
        std::vector<uint8_t> cso;
        if (!gpu::LoadShaderBytecode(csoName, cso, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            return nullptr;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = rootSignature.Get();
        desc.CS = {cso.data(), cso.size()};
        gpu::ComPtr<ID3D12PipelineState> pso;
        AmdExtD3DPipelineHandle handle = nullptr;
        AmdExtD3DComputeShaderStats stats{};
        if (!gpu::CreateAnalyzedComputePipeline(ext, desc, pso, handle, stats, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            return nullptr;
        }
        std::string isa;
        std::string statsLine;
        if (gpu::GetPipelineIsa(ext, handle, isa, statsLine, error)) {
            SaveReport(ReportsDirM11(), std::string(csoName) + ".isa", isa);
            const size_t wmmaCount = CountIsaLines(isa, "v_wmma_i32_16x16x16_iu8");
            std::printf("kernel %-24s | wmma x%zu | %s\n", csoName, wmmaCount, statsLine.c_str());
            if (requireWmma && wmmaCount == 0) {
                std::fprintf(stderr, "error: native WMMA missing from %s ISA\n", csoName);
                ok = false;
            }
        } else {
            std::fprintf(stderr, "error: ISA unavailable for %s: %s\n", csoName, error.c_str());
            ok = false;
        }
        return pso;
    };

    std::printf("kernels (analyzer PSOs, ISA evidence -> build/reports/m11/):\n");
    const gpu::ComPtr<ID3D12PipelineState> gemmRaw = LoadKernelPso("gemm_raw_m16k16n16", true);
    const gpu::ComPtr<ID3D12PipelineState> requant = LoadKernelPso("gemm_requant_m16n16", false);
    const gpu::ComPtr<ID3D12PipelineState> requantRelu = LoadKernelPso("gemm_requant_m16n16_relu", false);
    if (!ok) {
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    // --- Fixed buffers (EXP-009 layout; requant1 writes the hidden directly
    // into GEMM2's A region, so no copy between layers). ---
    std::vector<uint8_t> gemm1In(size_t(W) * W + size_t(W) * W + 1024); // A | B(w1) | zeroTile
    memcpy(gemm1In.data(), input.data(), input.size());
    memcpy(gemm1In.data() + input.size(), firstWeights.data(), firstWeights.size());
    std::vector<uint8_t> gemm2In(size_t(W) * W + size_t(W) * W + 1024); // A(hidden) | B(w2) | zeroTile
    memcpy(gemm2In.data() + W * W, secondWeights.data(), secondWeights.size());
    std::vector<uint8_t> acc1(size_t(W) * W * 4 + W * 2, 0); // acc | fp16 bias (zeros)
    std::vector<uint8_t> acc2(size_t(W) * W * 4 + W * 2, 0);

    const std::vector<gpu::SequenceBufferDesc> buffers = {
        {gemm1In.size(), gemm1In.data()},
        {acc1.size(), acc1.data()},
        {gemm2In.size(), gemm2In.data()},
        {acc2.size(), acc2.data()},
        {size_t(W) * W, nullptr}, // network output (int8, zero-filled)
    };
    const std::vector<gpu::DispatchStage> stages = {
        {gemmRaw.Get(), 1, 1, 0, 1},        // acc1 = input x w1
        {requantRelu.Get(), 1, 1, 1, 2},    // hidden (ReLU) -> GEMM2's A region
        {gemmRaw.Get(), 1, 1, 2, 3},        // acc2 = hidden x w2
        {requant.Get(), 1, 1, 3, 4},        // output
    };
    const std::vector<gpu::SequenceSnapshotDesc> snapshots = {
        {1, 0, size_t(W) * W * 4}, // acc1 (C05)
        {2, 0, size_t(W) * W},     // hidden (C05)
        {3, 0, size_t(W) * W * 4}, // acc2 (C05)
    };

    // Discarded warm cycle (cross-run clock state dominates otherwise — M07/M09).
    {
        std::vector<std::vector<uint8_t>> warmSnaps;
        std::vector<uint8_t> outWarmup, outTimed;
        gpu::MatmulRunResult discard;
        if (!gpu::RunDispatchSequence(ctx, rootSignature, buffers, stages, snapshots, 4,
                                      size_t(W) * W, 1, 1, warmSnaps, outWarmup, outTimed,
                                      discard, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            gpu::ReleaseAmdExtensions(ext);
            return 1;
        }
    }

    // Measured run: 3 warmup networks, 100 timed networks.
    std::vector<std::vector<uint8_t>> snaps;
    std::vector<uint8_t> outWarmup, outTimed;
    gpu::MatmulRunResult timing;
    if (!gpu::RunDispatchSequence(ctx, rootSignature, buffers, stages, snapshots, 4,
                                  size_t(W) * W, 3, 100, snaps, outWarmup, outTimed,
                                  timing, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }
    if (snaps.size() != 3) {
        std::fprintf(stderr, "error: missing intermediate snapshots\n");
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    // --- C05: intermediates vs the CPU reference (bit-exact by construction:
    // INT32-exact WMMA accumulation and exact power-of-two requantization). ---
    const auto compareI32 = [&](const char* label, const std::vector<uint8_t>& gpu,
                                const FixedNetworkAccumulators& cpu) {
        size_t mismatches = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            int32_t gpuValue;
            memcpy(&gpuValue, gpu.data() + i * 4, 4);
            if (gpuValue != int32_t(cpu[i])) {
                if (mismatches < 4)
                    std::fprintf(stderr, "  %s[%zu]: gpu %d vs cpu %lld\n", label, i,
                                 int(gpuValue), (long long)cpu[i]);
                ++mismatches;
            }
        }
        std::printf("%-10s | %zu mismatches vs CPU INT64 reference\n", label, mismatches);
        if (mismatches)
            ok = false;
    };
    const auto compareI8 = [&](const char* label, const std::vector<uint8_t>& gpu,
                               const FixedNetworkBytes& cpu) {
        size_t mismatches = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            const int gpuValue = int8_t(gpu[i]);
            if (gpuValue != int(cpu[i])) {
                if (mismatches < 4)
                    std::fprintf(stderr, "  %s[%zu]: gpu %d vs cpu %d\n", label, i,
                                 gpuValue, int(cpu[i]));
                ++mismatches;
            }
        }
        std::printf("%-10s | %zu mismatches vs CPU reference\n", label, mismatches);
        if (mismatches)
            ok = false;
    };
    compareI32("acc1", snaps[0], ref.firstAcc);
    compareI8("hidden", snaps[1], ref.hidden);
    compareI32("acc2", snaps[2], ref.secondAcc);

    // --- C06: final output, warmup AND timed snapshots vs CPU. ---
    compareI8("output", outWarmup, ref.output);
    if (outWarmup != outTimed) {
        std::fprintf(stderr, "error: timed output differs from warmup output\n");
        ok = false;
    }

    // --- C07: end-to-end GPU timing (upload/readback excluded). ---
    std::printf("network 2x(%ux%ux%u INT8 WMMA + requant) | min %.2f us | avg %.2f us | "
                "max %.2f us over %u timed executions (3 warmup)\n",
                W, W, W, timing.minMs * 1000.0, timing.avgMs * 1000.0, timing.maxMs * 1000.0,
                timing.iterations);

    // --- Intermediate inspection artifacts (build/reports/m11/). ---
    const bool saved =
        SaveReport(ReportsDirM11(), "network-input.bin",
                   std::string(reinterpret_cast<const char*>(input.data()), input.size())) &&
        SaveReport(ReportsDirM11(), "weights-first.bin",
                   std::string(reinterpret_cast<const char*>(firstWeights.data()), firstWeights.size())) &&
        SaveReport(ReportsDirM11(), "weights-second.bin",
                   std::string(reinterpret_cast<const char*>(secondWeights.data()), secondWeights.size())) &&
        SaveReport(ReportsDirM11(), "hidden-gpu.bin", std::string(snaps[1].begin(), snaps[1].end())) &&
        SaveReport(ReportsDirM11(), "output-gpu.bin", std::string(outWarmup.begin(), outWarmup.end()));
    if (!saved) {
        std::fprintf(stderr, "error: report write failed\n");
        ok = false;
    }

    std::string queueErrors;
    size_t benignCount = 0;
    if (!gpu::InfoQueueCleanGbavBenign(ctx, benignCount, queueErrors)) {
        std::fprintf(stderr, "%s", queueErrors.c_str());
        ok = false;
    } else {
        std::printf("diagnostics: no unexplained error-severity messages (%zu known-benign "
                    "GBAV intrinsics-dummy reports, see EXP-004)\n", benignCount);
    }

    gpu::ReleaseAmdExtensions(ext);
    std::printf("%s\n", ok ? "fixed-network inference: PASS" : "fixed-network inference: FAIL");
    return ok ? 0 : 1;
}

} // namespace experiments