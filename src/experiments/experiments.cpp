#include "experiments.h"

#include "../gpu/amdext.h"
#include "../gpu/compute.h"
#include "../gpu/device.h"
#include "../gpu/matmul.h"
#include "../reference/reference.h"
#include "core/numerics.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

namespace experiments {
namespace {

void PrintAdapter(const gpu::AdapterInfo& adapter) {
    std::printf("adapter: %ls (vid=0x%04X did=0x%04X, %zu MiB, luid 0x%llX)\n",
                adapter.name.c_str(), adapter.vendorId, adapter.deviceId,
                adapter.dedicatedVideoMemory >> 20,
                (unsigned long long)adapter.luidHighLow);
}

} // namespace

namespace {

// Counts occurrences of `needle` (lowercase) in the ISA text.
size_t CountIsaLines(const std::string& isa, const char* needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = isa.find(needle, pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    return count;
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
    return true;
}

std::filesystem::path ReportsDir() {
    wchar_t exePathW[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exePathW, MAX_PATH) == 0)
        return {};
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    return exeDir / ".." / "reports" / "m06";
}

std::filesystem::path ReportsDirM08() {
    wchar_t exePathW[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exePathW, MAX_PATH) == 0)
        return {};
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    return exeDir / ".." / "reports" / "m08";
}

std::filesystem::path ReportsDirM09() {
    wchar_t exePathW[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exePathW, MAX_PATH) == 0)
        return {};
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    return exeDir / ".." / "reports" / "m09";
}

} // namespace

int RunWmmaInt8() {
    // Unbuffered stdout: crash diagnostics must not be lost in the pipe.
    setvbuf(stdout, nullptr, _IONBF, 0);
    // --- Deterministic tile data covering negatives, positives, and a
    // nonzero accumulator (accumulate semantics are part of the gate). ---
    uint8_t a[256];
    uint8_t b[256];
    int32_t initC[256];
    int32_t expected[4][256];
    for (uint32_t i = 0; i < 256; ++i) {
        a[i] = uint8_t((i * 37u + 11u) & 0xFFu);        // full byte range → negative int8s included
        b[i] = uint8_t((i * 53u + 29u) & 0xFFu);
        initC[i] = int32_t(i) * 13 - 100;
    }
    reference::CpuInt8Matmul16x16x16(a, b, initC, expected[0], 0); // i8*i8
    reference::CpuInt8Matmul16x16x16(a, b, initC, expected[1], 1); // u8*u8
    reference::CpuInt8Matmul16x16x16(a, b, initC, expected[2], 2); // i8*u8
    reference::CpuInt8Matmul16x16x16(a, b, initC, expected[3], 3); // u8*i8

    uint8_t input[256 + 256 + 1024];
    memcpy(input, a, 256);
    memcpy(input + 256, b, 256);
    memcpy(input + 512, initC, 1024);

    // --- Device + extensions. ---
    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    PrintAdapter(ctx.adapter);

    gpu::AmdExtensions ext;
    if (!gpu::LoadAmdExtensions(ctx, ext, error) || !error.empty()) {
        // LoadAmdExtensions reports degraded-but-usable states via non-empty
        // error (e.g. Enable failed); for this probe any extension anomaly is
        // fatal — a silently unpatched compile must never reach dispatch.
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }
    std::printf("amd extensions: intrinsics %s, analyzer %s\n",
                ext.intrinsics ? "yes" : "NO", ext.analyzer ? "yes" : "NO");
    if (ext.intrinsics)
        std::printf("intrinsics: waveSize %u (min %u, max %u), WaveMatrix support: %s\n",
                    ext.intrinsicsInfo.waveSize, ext.intrinsicsInfo.minWaveSize,
                    ext.intrinsicsInfo.maxWaveSize,
                    ext.waveMatrixSupported ? "yes" : "NO");
    if (!ext.waveMatrixSupported || !ext.analyzer) {
        std::fprintf(stderr, "error: WaveMatrix intrinsics or analyzer unsupported — "
                             "WMMA probe cannot be validated on this driver\n");
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    gpu::ComPtr<ID3D12RootSignature> rootSignature;
    if (!gpu::CreateGemmTileRootSignature(ctx, rootSignature, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    const std::filesystem::path reportsDir = ReportsDir();
    struct Probe {
        const char* cso;       // compiled shader object name (shaders/<name>.cso)
        const char* label;
        uint32_t mode;         // CPU reference mode
        bool expectWmma;
    };
    const Probe probes[] = {
        {"int8_wmma_i8",   "wmma i8*i8",          0, true},
        {"int8_wmma_u8",   "wmma u8*u8",          1, true},
        {"int8_wmma_iu8",  "wmma i8*u8",          2, true},
        {"int8_wmma_ui8",  "wmma u8*i8",          3, true},
        {"int8_dp4a",      "dp4a i8*i8 (control)", 0, false},
    };

    bool ok = true;
    for (const Probe& probe : probes) {
        std::vector<uint8_t> cso;
        if (!gpu::LoadShaderBytecode(probe.cso, cso, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rootSignature.Get();
        psoDesc.CS = {cso.data(), cso.size()};

        gpu::ComPtr<ID3D12PipelineState> pso;
        AmdExtD3DPipelineHandle handle = nullptr;
        AmdExtD3DComputeShaderStats stats{};
        if (!gpu::CreateAnalyzedComputePipeline(ext, psoDesc, pso, handle, stats, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }

        std::string isa;
        std::string statsLine;
        if (!gpu::GetPipelineIsa(ext, handle, isa, statsLine, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }
        const std::string isaFile = std::string(probe.cso) + ".isa";
        const bool saved = SaveReport(reportsDir, isaFile, isa);
        const size_t wmmaLines = CountIsaLines(isa, "wmma");
        const size_t dot4Lines = CountIsaLines(isa, "dot4");

        gpu::MatmulRunResult timing;
        std::vector<int32_t> outWarmup;
        std::vector<int32_t> outTimed;
        if (!gpu::RunGemmTile(ctx, rootSignature, pso, sizeof(input), sizeof(initC),
                              /*gridX=*/1, /*gridY=*/1, /*warmup=*/1, /*timed=*/100,
                              input, outWarmup, outTimed, timing, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }

        const size_t mismatches =
            probe.mode < 4 ? reference::CountI32Mismatches(outWarmup.data(), expected[probe.mode], 256)
                           : reference::CountI32Mismatches(outWarmup.data(), expected[0], 256);
        // Free idempotency check: every probe kernel reloads its inputs and
        // overwrites the whole tile per dispatch, so the post-timed snapshot
        // (timing-only loop) must equal the post-warmup snapshot bit-for-bit.
        const size_t idempotencyDelta =
            reference::CountI32Mismatches(outTimed.data(), outWarmup.data(), 256);
        const bool isaOk = probe.expectWmma ? (wmmaLines > 0) : (wmmaLines == 0);
        const bool modeOk = (mismatches == 0) && (idempotencyDelta == 0) && isaOk && saved;

        std::printf("%-22s | correct %s (%4zu mismatches) | idempotent %s | wmma x%3zu dot4 x%3zu | %s\n",
                    probe.label, mismatches == 0 ? "OK  " : "FAIL", mismatches,
                    idempotencyDelta == 0 ? "OK" : "FAIL", wmmaLines, dot4Lines, statsLine.c_str());
        std::printf("%-22s | gpu time: min %.3f us, avg %.3f us, max %.3f us | isa saved: %s\n",
                    "", timing.minMs * 1000.0, timing.avgMs * 1000.0, timing.maxMs * 1000.0,
                    saved ? isaFile.c_str() : "FAILED");

        if (!modeOk) {
            std::fprintf(stderr, "error: probe %s failed (correctness %s, idempotency %s, isa %s, saved %s)\n",
                         probe.label, mismatches == 0 ? "ok" : "FAIL",
                         idempotencyDelta == 0 ? "ok" : "FAIL",
                         isaOk ? "ok" : "FAIL", saved ? "ok" : "FAIL");
            ok = false;
            break;
        }
    }

    if (ok) {
        std::string queueErrors;
        size_t benignCount = 0;
        // GPU-based validation shadow passes flag the AMD intrinsics dummy UAV
        // (root param 2, the extension-designated binding at space 0x7FFF0ADE)
        // as "Root descriptor access out of bounds" at the magic-code addresses
        // (0x5N0000xx). Known false positive, evidence in EXP-004: the final
        // ISA (build/reports/m06/*.isa) contains no buffer atomics at all —
        // the driver patched every magic atomic into fragment loads/stores and
        // v_wmma — so the executed code never touches that UAV; GBAV
        // instruments the pre-patch IL access pattern. Only this exact message
        // class is treated as benign; any other error still fails the probe.
        const auto benignIntrinsicsDummy =
            [](const D3D12_MESSAGE& msg) {
                if (!msg.pDescription)
                    return false;
                const char* desc = msg.pDescription;
                return strstr(desc, "GPU-BASED VALIDATION") != nullptr &&
                       strstr(desc, "Root descriptor access out of bounds") != nullptr &&
                       strstr(desc, "Root Parameter Index: [2]") != nullptr;
            };
        if (!gpu::InfoQueueCleanWithFilter(ctx, benignIntrinsicsDummy, benignCount, queueErrors)) {
            std::fprintf(stderr, "%s", queueErrors.c_str());
            ok = false;
        } else {
            std::printf("diagnostics: no unexplained error-severity messages (%zu known-benign "
                        "GBAV intrinsics-dummy reports classified, see EXP-004)\n",
                        benignCount);
        }
    }

    gpu::ReleaseAmdExtensions(ext);
    std::printf("%s\n", ok ? "int8 wmma probe: PASS" : "int8 wmma probe: FAIL");
    return ok ? 0 : 1;
}

int RunWmmaFp16() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    // --- FP16 single-tile correctness (C01/C02): values in [-1, 1] keep the
    // fp16-accumulator error bounded; full byte-range magnitudes are not the
    // fp16 probe's point (documented tolerance compare, not bit-exact). ---
    uint16_t fa[256];
    uint16_t fb[256];
    for (uint32_t i = 0; i < 256; ++i) {
        // All arithmetic in float: (i % 17) - 8 in uint32 would underflow.
        const float av = float(i % 17) * 0.125f - 1.0f;                    // [-1, 1]
        const float bv = float((i * 7u) % 13u) * 0.125f - 0.75f;           // [-0.75, 0.75]
        fa[i] = core::F32ToF16(av);
        fb[i] = core::F32ToF16(bv);
    }
    float initF32[256];
    for (uint32_t i = 0; i < 256; ++i)
        initF32[i] = float(i % 9) - 4.0f;
    uint16_t initF16[256];
    for (uint32_t i = 0; i < 256; ++i)
        initF16[i] = core::F32ToF16(initF32[i]);

    float expectedF32acc[256];
    uint16_t expectedF16acc[256];
    reference::CpuF16Matmul16x16x16(fa, fb, initF32, expectedF32acc, 32);
    reference::CpuF16Matmul16x16x16(fa, fb, initF16, expectedF16acc, 16);

    uint8_t f32accInput[512 + 512 + 1024];
    uint8_t f16accInput[512 + 512 + 512];
    memcpy(f32accInput, fa, 512);
    memcpy(f32accInput + 512, fb, 512);
    memcpy(f32accInput + 1024, initF32, 1024);
    memcpy(f16accInput, fa, 512);
    memcpy(f16accInput + 512, fb, 512);
    memcpy(f16accInput + 1024, initF16, 512);

    // Tolerances: per-product is exact in fp32 (11-bit mantissas -> 22-bit
    // product); the [U] hardware accumulation order differs from the CPU's
    // sequential order by at most the order-swap rounding budget. 0.05 covers
    // 16 products of magnitude <= 1 with partial sums <= 16 in fp16-accum;
    // the fp32-accum tolerance is far tighter in practice (printed).
    constexpr float kTolF32acc = 2e-3f;
    constexpr float kTolF16acc = 0.06f;

    // --- Device + extensions (same path as M06). ---
    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    PrintAdapter(ctx.adapter);

    gpu::AmdExtensions ext;
    if (!gpu::LoadAmdExtensions(ctx, ext, error) || !error.empty()) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }
    if (!ext.waveMatrixSupported || !ext.analyzer) {
        std::fprintf(stderr, "error: WaveMatrix intrinsics or analyzer unsupported\n");
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    gpu::ComPtr<ID3D12RootSignature> rootSignature;
    if (!gpu::CreateGemmTileRootSignature(ctx, rootSignature, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    const std::filesystem::path reportsDir = ReportsDir();
    struct TileProbe {
        const char* cso;
        const char* label;
        bool fp16Accumulator;
        size_t outputBytes;
        const void* input;
        size_t inputBytes;
    };
    const TileProbe tileProbes[] = {
        {"fp16_wmma_f32acc", "fp16 wmma ->f32 acc", false, 1024, f32accInput, sizeof(f32accInput)},
        {"fp16_wmma_f16acc", "fp16 wmma ->f16 acc", true, 512, f16accInput, sizeof(f16accInput)},
    };

    bool ok = true;
    for (const TileProbe& probe : tileProbes) {
        std::vector<uint8_t> cso;
        if (!gpu::LoadShaderBytecode(probe.cso, cso, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rootSignature.Get();
        psoDesc.CS = {cso.data(), cso.size()};

        gpu::ComPtr<ID3D12PipelineState> pso;
        AmdExtD3DPipelineHandle handle = nullptr;
        AmdExtD3DComputeShaderStats stats{};
        if (!gpu::CreateAnalyzedComputePipeline(ext, psoDesc, pso, handle, stats, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }
        std::string isa;
        std::string statsLine;
        if (!gpu::GetPipelineIsa(ext, handle, isa, statsLine, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }
        const std::string isaFile = std::string(probe.cso) + ".isa";
        const bool saved = SaveReport(reportsDir, isaFile, isa);
        const size_t wmmaF32 = CountIsaLines(isa, "v_wmma_f32_16x16x16_f16");
        const size_t wmmaF16 = CountIsaLines(isa, "v_wmma_f16_16x16x16_f16");

        gpu::MatmulRunResult timing;
        std::vector<int32_t> outWarmup;
        std::vector<int32_t> outTimed;
        if (!gpu::RunGemmTile(ctx, rootSignature, pso, probe.inputBytes, probe.outputBytes,
                              1, 1, 1, 100, probe.input,
                              outWarmup, outTimed, timing, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }

        float maxDiff = 0.0f;
        float tolerance = 0.0f;
        if (probe.fp16Accumulator) {
            maxDiff = reference::MaxAbsDiffF16(reinterpret_cast<const uint16_t*>(outWarmup.data()),
                                               expectedF16acc, 256);
            tolerance = kTolF16acc;
        } else {
            maxDiff = reference::MaxAbsDiffF32(reinterpret_cast<const float*>(outWarmup.data()),
                                               expectedF32acc, 256);
            tolerance = kTolF32acc;
        }
        const bool correct = maxDiff <= tolerance;
        const bool isaOk = probe.fp16Accumulator ? (wmmaF16 > 0 && wmmaF32 == 0)
                                                 : (wmmaF32 > 0 && wmmaF16 == 0);
        std::printf("%-22s | maxdiff %.6g (tol %.6g) %s | wmma_f32 x%zu wmma_f16 x%zu | %s\n",
                    probe.label, maxDiff, tolerance, correct ? "OK  " : "FAIL",
                    wmmaF32, wmmaF16, statsLine.c_str());
        std::printf("%-22s | gpu time: min %.3f us, avg %.3f us | isa saved: %s\n",
                    "", timing.minMs * 1000.0, timing.avgMs * 1000.0,
                    saved ? isaFile.c_str() : "FAILED");

        if (!correct || !isaOk || !saved) {
            std::fprintf(stderr, "error: fp16 tile probe %s failed\n", probe.label);
            ok = false;
            break;
        }
    }

    // --- Multi-tile comparison: 64 GEMMs of 16x16xK, K in {64, 256}; the same
    // arithmetic on three paths (int8 WMMA / fp16 WMMA(f32acc) / int8 DP4A).
    // K-scaling distinguishes compute-bound (time ~ K) from overhead-bound
    // (flat) behavior (C07). ---
    if (ok) {
        constexpr uint32_t kGemms = 64;
        constexpr uint32_t kTileCounts[2] = {4, 16}; // K = 64, K = 256
        struct GemmKernel {
            const char* csoK64;
            const char* csoK256;
            const char* label;
            bool fp16;
            uint32_t threadsPerGemm;
        };
        const GemmKernel gemmKernels[] = {
            {"gemm_int8_wmma_k64", "gemm_int8_wmma_k256", "int8 wmma", false, 32},
            {"gemm_fp16_wmma_k64", "gemm_fp16_wmma_k256", "fp16 wmma", true, 32},
            {"gemm_int8_dp4a_k64", "gemm_int8_dp4a_k256", "int8 dp4a", false, 256},
        };

        struct GemmResult {
            double minUs[2];
            double avgUs[2];
            double minMs[2];
            double avgMs[2];
            size_t vgpr = 0;
            size_t lds = 0;
        };
        GemmResult results[3] = {};

        for (uint32_t ki = 0; ki < 2; ++ki) {
            const uint32_t kTiles = kTileCounts[ki];
            // Build one input blob per kernel type (int8 layout shared by
            // WMMA + DP4A; fp16 layout separate).
            std::vector<uint8_t> in8(kGemms * (2u * 16u * 16u * kTiles + 1024u));
            std::vector<uint8_t> inF16(kGemms * (2u * 32u * 16u * kTiles + 1024u));
            std::vector<int32_t> ref8(kGemms * 256);
            std::vector<float> refF16(kGemms * 256);
            for (uint32_t g = 0; g < kGemms; ++g) {
                uint8_t* base8 = in8.data() + g * (2u * 16u * 16u * kTiles + 1024u);
                uint8_t* baseF = inF16.data() + g * (2u * 32u * 16u * kTiles + 1024u);
                int32_t* c8 = reinterpret_cast<int32_t*>(base8 + 2u * 16u * 16u * kTiles);
                float* cF = reinterpret_cast<float*>(baseF + 2u * 32u * 16u * kTiles);

                int32_t acc8[256];
                float accF[256];
                for (uint32_t e = 0; e < 256; ++e) {
                    acc8[e] = int32_t((g * 256u + e) % 7u) - 3;
                    accF[e] = float((g * 256u + e) % 5u) - 2.0f;
                    c8[e] = acc8[e];
                    cF[e] = accF[e];
                }
                for (uint32_t t = 0; t < kTiles; ++t) {
                    uint8_t* a8 = base8 + t * 512u;
                    uint8_t* b8 = a8 + 256u;
                    uint16_t* aF = reinterpret_cast<uint16_t*>(baseF + t * 1024u);
                    uint16_t* bF = reinterpret_cast<uint16_t*>(baseF + t * 1024u + 512u);
                    for (uint32_t e = 0; e < 256; ++e) {
                        a8[e] = uint8_t(((g * 256u + e + t) * 37u + 11u) & 0xFFu);
                        b8[e] = uint8_t(((g * 256u + e + t) * 53u + 29u) & 0xFFu);
                        aF[e] = core::F32ToF16(float(e % 17) * 0.125f - 1.0f);
                        bF[e] = core::F32ToF16(float((e * 7u) % 13u) * 0.125f - 0.75f);
                    }
                    int32_t tile8[256];
                    reference::CpuInt8Matmul16x16x16(a8, b8, acc8, tile8, 0);
                    memcpy(acc8, tile8, sizeof(acc8));
                    reference::CpuF16Matmul16x16x16(aF, bF, accF, accF, 32);
                }
                memcpy(ref8.data() + g * 256, acc8, sizeof(acc8));
                memcpy(refF16.data() + g * 256, accF, sizeof(accF));
            }

            for (uint32_t m = 0; m < 3; ++m) {
                const GemmKernel& kernel = gemmKernels[m];
                const char* csoName = ki == 0 ? kernel.csoK64 : kernel.csoK256;
                const uint8_t* input = kernel.fp16 ? inF16.data() : in8.data();
                const size_t inputBytes = kernel.fp16 ? inF16.size() : in8.size();
                // Per-kernel verdicts: a failing kernel is reported but does not
                // abort the loop — full comparison data per run.
                bool kernelOk = true;

                std::vector<uint8_t> cso;
                if (!gpu::LoadShaderBytecode(csoName, cso, error)) {
                    std::fprintf(stderr, "error: %s\n", error.c_str());
                    ok = false;
                    break;
                }
                D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
                psoDesc.pRootSignature = rootSignature.Get();
                psoDesc.CS = {cso.data(), cso.size()};
                gpu::ComPtr<ID3D12PipelineState> pso;
                AmdExtD3DPipelineHandle handle = nullptr;
                AmdExtD3DComputeShaderStats stats{};
                if (!gpu::CreateAnalyzedComputePipeline(ext, psoDesc, pso, handle, stats, error)) {
                    std::fprintf(stderr, "error: %s\n", error.c_str());
                    ok = false;
                    break;
                }
                if (ki == 0) {
                    results[m].vgpr = stats.usageStats.numUsedVgprs;
                    results[m].lds = stats.usageStats.ldsUsageSizeInBytes;
                }
                // ISA evidence per (kernel, K) configuration.
                {
                    std::string isa;
                    std::string statsLine;
                    if (gpu::GetPipelineIsa(ext, handle, isa, statsLine, error))
                        SaveReport(reportsDir, std::string(csoName) + ".isa", isa);
                    else {
                        std::fprintf(stderr, "warning: ISA unavailable for %s: %s\n",
                                     csoName, error.c_str());
                        ok = false;
                    }
                }

                gpu::MatmulRunResult timing;
                std::vector<int32_t> outWarmup;
                std::vector<int32_t> outTimed;
                if (!gpu::RunGemmTile(ctx, rootSignature, pso, inputBytes, kGemms * 1024u,
                                      kGemms, 1, 10, 100, input,
                                      outWarmup, outTimed, timing, error)) {
                    std::fprintf(stderr, "error: %s\n", error.c_str());
                    ok = false;
                    break;
                }

                // Correctness: int8 paths bit-exact; fp16 within tolerance (256
                // sequential adds at magnitude <= 258: order-swap bound ~8e-3).
                if (kernel.fp16) {
                    const float maxDiff = reference::MaxAbsDiffF32(
                        reinterpret_cast<const float*>(outWarmup.data()), refF16.data(),
                        size_t(kGemms) * 256);
                    // NaN maxDiff must fail: use the not-less form.
                    if (!(maxDiff <= 0.05f)) {
                        std::fprintf(stderr, "error: fp16 gemm correctness maxdiff %.6g (tol 0.05)\n",
                                     maxDiff);
                        kernelOk = false;
                    }
                } else {
                    const size_t mismatches = reference::CountI32Mismatches(
                        outWarmup.data(), ref8.data(), size_t(kGemms) * 256);
                    if (mismatches != 0) {
                        std::fprintf(stderr, "error: int8 gemm (%s) correctness %zu mismatches\n",
                                     csoName, mismatches);
                        for (uint32_t d = 0; d < 6 && d < size_t(kGemms) * 256; ++d)
                            std::fprintf(stderr, "  [%u] gpu %d vs ref %d\n", d,
                                         outWarmup[d], ref8[d]);
                        kernelOk = false;
                    }
                }
                if (!kernelOk)
                    ok = false;
                results[m].minUs[ki] = timing.minMs * 1000.0;
                results[m].avgUs[ki] = timing.avgMs * 1000.0;
                results[m].minMs[ki] = timing.minMs;
                results[m].avgMs[ki] = timing.avgMs;
            }
        }

        if (ok) {
            std::printf("\nmulti-tile comparison: %u gemms of 16x16xK, 10 warmup + 100 timed dispatches\n",
                        kGemms);
            std::printf("%-12s %8s %10s %10s %10s %10s %6s\n",
                        "kernel", "vgpr", "min@K64", "avg@K64", "min@K256", "avg@K256", "scale");
            for (uint32_t m = 0; m < 3; ++m) {
                const double scale = results[m].minUs[0] > 0.0
                                         ? results[m].minUs[1] / results[m].minUs[0]
                                         : 0.0;
                std::printf("%-12s %8zu %9.2fus %9.2fus %9.2fus %9.2fus %5.2fx\n",
                            gemmKernels[m].label, results[m].vgpr,
                            results[m].minUs[0], results[m].avgUs[0],
                            results[m].minUs[1], results[m].avgUs[1], scale);
            }
            std::printf("theoretical: 16x16x16 tile = 4096 MACs per wmma wave-instruction; "
                        "dp4a = 4 MACs per lane-instruction (per-lane outputs, 16 dots/tile-row)\n");
        }
    }

    if (ok) {
        std::string queueErrors;
        size_t benignCount = 0;
        const auto benignIntrinsicsDummy =
            [](const D3D12_MESSAGE& msg) {
                if (!msg.pDescription)
                    return false;
                const char* desc = msg.pDescription;
                return strstr(desc, "GPU-BASED VALIDATION") != nullptr &&
                       strstr(desc, "Root descriptor access out of bounds") != nullptr &&
                       strstr(desc, "Root Parameter Index: [2]") != nullptr;
            };
        if (!gpu::InfoQueueCleanWithFilter(ctx, benignIntrinsicsDummy, benignCount, queueErrors)) {
            std::fprintf(stderr, "%s", queueErrors.c_str());
            ok = false;
        } else {
            std::printf("diagnostics: no unexplained error-severity messages (%zu known-benign "
                        "GBAV intrinsics-dummy reports classified, see EXP-004)\n",
                        benignCount);
        }
    }

    gpu::ReleaseAmdExtensions(ext);
    std::printf("%s\n", ok ? "fp16 wmma + comparison: PASS" : "fp16 wmma + comparison: FAIL");
    return ok ? 0 : 1;
}

int RunQuantization() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    bool ok = true;

    // --- Part A (C01/C02): exhaustive FP8 decode + encode vs independent oracles. ---
    float e4m3[256];
    float e5m2[256];
    reference::BuildE4M3Table(e4m3);
    reference::BuildE5M2Table(e5m2);
    {
        size_t decodeMismatch = 0;
        size_t roundTripMismatch = 0;
        size_t nanCount = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            const float decoded = core::E4M3ToF32(uint8_t(b));
            const float oracle = e4m3[b];
            const bool bothNan = std::isnan(decoded) && std::isnan(oracle);
            if (!bothNan && decoded != oracle)
                ++decodeMismatch;
            if (bothNan)
                ++nanCount;
            // Round-trip: encode(decode(b)) == b for every non-NaN encoding.
            if (!bothNan && core::F32ToE4M3(decoded) != b)
                ++roundTripMismatch;
        }
        // Sign symmetry: decode(b ^ 0x80) == -decode(b) for all non-NaN.
        size_t signMismatch = 0;
        for (uint32_t b = 1; b < 128; ++b) {
            if (!(std::isnan(e4m3[b]) || std::isnan(e4m3[b | 0x80])) &&
                e4m3[b | 0x80] != -e4m3[b])
                ++signMismatch;
        }
        // Monotonicity across the finite positive range 0x01..0x7E.
        for (uint32_t b = 1; b < 0x7E; ++b)
            if (!(e4m3[b] < e4m3[b + 1])) {
                ++decodeMismatch;
                std::fprintf(stderr, "  e4m3 monotonicity break at 0x%02X\n", b);
            }
        // Hand-derived spec spot values (OCP OFP8).
        struct Spot { uint8_t enc; float value; };
        const Spot e4m3Spots[] = {
            {0x00, 0.0f}, {0x80, -0.0f}, {0x38, 1.0f}, {0x37, 0.9375f},
            {0x07, 7.0f / 512.0f}, {0x40, 2.0f}, {0x78, 256.0f}, {0x7E, 448.0f},
            {0x7F, std::numeric_limits<float>::quiet_NaN()},
        };
        for (const Spot& spot : e4m3Spots) {
            const float got = core::E4M3ToF32(spot.enc);
            if (std::isnan(spot.value) ? !std::isnan(got) : got != spot.value) {
                std::fprintf(stderr, "  e4m3 spot value 0x%02X: got %.9g expected %.9g\n",
                             spot.enc, got, spot.value);
                ++decodeMismatch;
            }
        }

        std::printf("fp8 e4m3: decode mismatches %zu, round-trip mismatches %zu, "
                    "sign-symmetry mismatches %zu, NaN encodings %zu/2\n",
                    decodeMismatch, roundTripMismatch, signMismatch, nanCount);
        if (decodeMismatch || roundTripMismatch || signMismatch || nanCount != 2)
            ok = false;
    }
    {
        size_t decodeMismatch = 0;
        size_t roundTripMismatch = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            const float decoded = core::E5M2ToF32(uint8_t(b));
            const float oracle = e5m2[b];
            const bool bothNan = std::isnan(decoded) && std::isnan(oracle);
            // Saturating encode maps ±inf onto max finite by design — those
            // two encodings cannot round-trip (documented policy).
            const bool isInf = std::isinf(decoded);
            if (!bothNan && decoded != oracle)
                ++decodeMismatch;
            if (!bothNan && !isInf && core::F32ToE5M2(decoded) != b)
                ++roundTripMismatch;
        }
        struct Spot { uint8_t enc; float value; };
        const Spot e5m2Spots[] = {
            {0x38, 0.5f}, {0x3C, 1.0f}, {0x44, 4.0f}, {0x7B, 57344.0f},
            {0x01, std::ldexp(1.0f, -16)}, {0x7C, std::numeric_limits<float>::infinity()},
        };
        for (const Spot& spot : e5m2Spots) {
            const float got = core::E5M2ToF32(spot.enc);
            if (got != spot.value) {
                std::fprintf(stderr, "  e5m2 spot value 0x%02X: got %.9g expected %.9g\n",
                             spot.enc, got, spot.value);
                ++decodeMismatch;
            }
        }
        std::printf("fp8 e5m2: decode mismatches %zu, round-trip mismatches %zu\n",
                    decodeMismatch, roundTripMismatch);
        if (decodeMismatch || roundTripMismatch)
            ok = false;
    }
    // FP16 exhaustive validation (the conversions M06/M07 rely on): all 65536
    // encodings — decode vs an independently-composed formula, and
    // encode(decode(b)) identity for every non-NaN encoding (NaN payloads are
    // canonicalized to qNaN by design — documented in numerics.h).
    {
        size_t decodeMismatch = 0;
        size_t roundTripMismatch = 0;
        for (uint32_t b = 0; b < 65536; ++b) {
            const uint16_t bits = uint16_t(b);
            const float decoded = core::F16ToF32(bits);
            const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
            const uint32_t expField = (bits >> 10) & 0x1Fu;
            const uint32_t mant = bits & 0x03FFu;
            float oracle;
            if (expField == 0)
                oracle = std::ldexp(float(mant), -24) * (sign ? -1.0f : 1.0f);
            else if (expField == 31)
                oracle = (mant == 0)
                             ? (sign ? -std::numeric_limits<float>::infinity()
                                     : std::numeric_limits<float>::infinity())
                             : std::numeric_limits<float>::quiet_NaN();
            else
                oracle = std::ldexp(float(mant | 0x0400u), int(expField) - 25)
                         * (sign ? -1.0f : 1.0f);
            const bool bothNan = std::isnan(decoded) && std::isnan(oracle);
            if (!bothNan && decoded != oracle)
                ++decodeMismatch;
            if (!bothNan && core::F32ToF16(decoded) != bits)
                ++roundTripMismatch;
        }
        std::printf("fp16 exhaustive (65536 encodings): decode mismatches %zu, "
                    "round-trip mismatches %zu\n",
                    decodeMismatch, roundTripMismatch);
        if (decodeMismatch || roundTripMismatch)
            ok = false;
    }
    // Encode sweeps vs the linear-scan oracle (RNE): dense ranges around the
    // tie-prone magnitudes plus a wide random sweep. Deterministic LCG.
    {
        uint32_t seed = 0x12345678u;
        const auto next01 = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return float(seed >> 8) / float(1 << 24);
        };
        size_t e4Mismatch = 0;
        size_t e5Mismatch = 0;
        for (uint32_t i = 0; i < 60000; ++i) {
            float v;
            if (i < 20000) {
                // Dense around the E4M3 max-finite / boundary magnitudes.
                v = (next01() - 0.5f) * 1200.0f;
            } else if (i < 40000) {
                // Dense around E5M2 boundary magnitudes.
                v = (next01() - 0.5f) * 130000.0f;
            } else {
                // Tiny denormal-range magnitudes.
                v = (next01() - 0.5f) * std::ldexp(1.0f, -6);
            }
            if (core::F32ToE4M3(v) != reference::ReferenceEncodeE4M3(v))
                ++e4Mismatch;
            if (core::F32ToE5M2(v) != reference::ReferenceEncodeE5M2(v))
                ++e5Mismatch;
        }
        std::printf("fp8 encode sweeps: e4m3 mismatches %zu, e5m2 mismatches %zu (60000 values each)\n",
                    e4Mismatch, e5Mismatch);
        if (e4Mismatch || e5Mismatch)
            ok = false;
    }

    // --- Part B (C03/C04): symmetric INT8 quantize vs independent oracle. ---
    {
        const float scales[] = {1.0f, 0.0078125f, 0.014884727f, 2.0f};
        size_t rneMismatch = 0;
        size_t truncMismatch = 0;
        uint32_t seed = 0xDEADBEEFu;
        const auto next01 = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return float(seed >> 8) / float(1 << 24);
        };
        for (const float scale : scales) {
            for (uint32_t i = 0; i < 20000; ++i) {
                const float v = (next01() - 0.5f) * 600.0f * scale;
                if (core::QuantizeSymmetricS8(v, scale, core::QuantizeRounding::RoundNearestEven) !=
                    reference::ReferenceQuantizeS8(v, scale, true))
                    ++rneMismatch;
                if (core::QuantizeSymmetricS8(v, scale, core::QuantizeRounding::TowardZero) !=
                    reference::ReferenceQuantizeS8(v, scale, false))
                    ++truncMismatch;
            }
            // Deliberate boundary values around the clamp and halfway points.
            const float boundary[] = {-130.0f, -128.5f, -128.0f, -127.5f, -127.0f, -0.5f, 0.0f,
                                      0.5f, 127.0f, 127.49f, 127.5f, 128.0f, 1000.0f};
            for (const float b : boundary) {
                const float v = b * scale;
                if (core::QuantizeSymmetricS8(v, scale, core::QuantizeRounding::RoundNearestEven) !=
                    reference::ReferenceQuantizeS8(v, scale, true))
                    ++rneMismatch;
            }
        }
        std::printf("int8 quantize vs oracle: RNE mismatches %zu, truncate mismatches %zu\n",
                    rneMismatch, truncMismatch);
        if (rneMismatch || truncMismatch)
            ok = false;

        // Dequant round-trip bound: |dequant(q) - v| <= scale/2 (+fp32 eps).
        size_t boundViolations = 0;
        for (const float scale : scales) {
            for (uint32_t i = 0; i < 20000; ++i) {
                const float v = (next01() - 0.5f) * 100.0f * scale;
                const int8_t q = core::QuantizeSymmetricS8(
                    v, scale, core::QuantizeRounding::RoundNearestEven);
                const float back = core::DequantizeSymmetricS8(q, scale);
                if (std::fabs(back - v) > scale * 0.5f + scale * 1e-6f)
                    ++boundViolations;
            }
        }
        std::printf("int8 dequant round-trip: %zu violations of the |e| <= scale/2 bound\n",
                    boundViolations);
        if (boundViolations)
            ok = false;
    }

    // --- Part C (C05) + Part D (C06/C07/C08): synthetic distributions,
    // per-tensor vs per-channel, error metrics. All generic experiments —
    // the AMD-behavior claims live in the source citations + Part E. ---
    {
        uint32_t seed = 0x51A7B3C9u;
        const auto next01 = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return float(seed >> 8) / float(1 << 24);
        };
        const auto nextGauss = [&next01] {
            // Box-Muller (deterministic LCG input).
            const float u1 = std::max(next01(), 1e-12f);
            const float u2 = next01();
            return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
        };

        constexpr uint32_t kChannels = 8;
        constexpr uint32_t kPerChannel = 512;
        constexpr uint32_t kCount = kChannels * kPerChannel;

        // C05: gaussian weights with one 20x outlier channel.
        {
            std::vector<float> w(kCount);
            for (uint32_t c = 0; c < kChannels; ++c)
                for (uint32_t i = 0; i < kPerChannel; ++i)
                    w[c * kPerChannel + i] = nextGauss() * (c == 3 ? 20.0f : 1.0f);

            float dequant[kCount];
            int8_t q[kCount];

            // Per-tensor: scale = maxAbs/127 across ALL channels.
            float globalMax = 0.0f;
            for (float v : w)
                globalMax = std::max(globalMax, std::fabs(v));
            const float tensorScale = globalMax / 127.0f;
            for (uint32_t i = 0; i < kCount; ++i) {
                q[i] = core::QuantizeSymmetricS8(w[i], tensorScale,
                                                 core::QuantizeRounding::RoundNearestEven);
                dequant[i] = core::DequantizeSymmetricS8(q[i], tensorScale);
            }
            const reference::QuantErrorStats tensorStats =
                reference::ComputeQuantError(w.data(), q, dequant, kCount);

            // Per-channel: one scale per channel.
            double sqnrSum = 0.0;
            size_t perChannelClipped = 0;
            float perChannelMaxErr = 0.0f;
            for (uint32_t c = 0; c < kChannels; ++c) {
                float cMax = 0.0f;
                for (uint32_t i = 0; i < kPerChannel; ++i)
                    cMax = std::max(cMax, std::fabs(w[c * kPerChannel + i]));
                const float cScale = cMax / 127.0f;
                for (uint32_t i = 0; i < kPerChannel; ++i) {
                    const size_t idx = c * kPerChannel + i;
                    q[idx] = core::QuantizeSymmetricS8(w[idx], cScale,
                                                       core::QuantizeRounding::RoundNearestEven);
                    dequant[idx] = core::DequantizeSymmetricS8(q[idx], cScale);
                }
                const reference::QuantErrorStats cs =
                    reference::ComputeQuantError(w.data() + c * kPerChannel, q + c * kPerChannel,
                                                 dequant + c * kPerChannel, kPerChannel);
                sqnrSum += cs.sqnrDb;
                perChannelClipped += size_t(cs.clipFrac * kPerChannel);
                perChannelMaxErr = std::max(perChannelMaxErr, float(cs.maxAbsErr));
            }
            std::printf("per-tensor vs per-channel (outlier gaussian, 8ch): "
                        "SQNR %.2f dB vs %.2f dB (avg), clip %.3f%% vs %.3f%%\n",
                        tensorStats.sqnrDb, sqnrSum / kChannels,
                        tensorStats.clipFrac * 100.0, double(perChannelClipped) / kCount * 100.0);
        }

        // C06/C07/C08: distribution x scale-policy error table.
        struct Dist {
            const char* name;
            std::vector<float> values;
        };
        std::vector<Dist> dists;
        {
            std::vector<float> g01(kCount);
            std::vector<float> g1(kCount);
            std::vector<float> uni(kCount);
            std::vector<float> lap(kCount);
            std::vector<float> rel(kCount);
            std::vector<float> col(kCount);
            for (uint32_t i = 0; i < kCount; ++i) {
                const float g = nextGauss();
                g01[i] = g * 0.01f;
                g1[i] = g;
                uni[i] = next01() * 2.0f - 1.0f;
                const float u = next01() - 0.5f;
                lap[i] = std::copysign(-std::log(std::max(std::fabs(u), 1e-12f)) * 0.1f, u);
                rel[i] = std::max(0.0f, nextGauss() * 0.5f);
                col[i] = next01() * 2.0f - 1.0f; // FSR4 normalizes color to [-1, 1]
            }
            dists.push_back({"gauss sigma=0.01", std::move(g01)});
            dists.push_back({"gauss sigma=1", std::move(g1)});
            dists.push_back({"uniform [-1,1]", std::move(uni)});
            dists.push_back({"laplace 0.1", std::move(lap)});
            dists.push_back({"relu(gauss*0.5)", std::move(rel)});
            dists.push_back({"fsr4 color [-1,1]", std::move(col)});
        }

        std::string report = "distribution,scale_policy,scale,max_abs_err,mae,rmse,sqnr_db,clip_pct\n";
        for (const Dist& d : dists) {
            float absMax = 0.0f;
            double mean = 0.0;
            for (float v : d.values) {
                absMax = std::max(absMax, std::fabs(v));
                mean += v;
            }
            mean /= kCount;
            // Second moment for the sigma-based scale policy.
            double var = 0.0;
            for (float v : d.values)
                var += (v - mean) * (v - mean);
            var /= kCount;
            const float sigma = float(std::sqrt(var));

            const float policyScales[2] = {absMax / 127.0f, std::max(sigma * 3.0f, 1e-9f)};
            const char* policyNames[2] = {"absmax", "3sigma"};
            for (uint32_t p = 0; p < 2; ++p) {
                const float scale = policyScales[p];
                std::vector<int8_t> q(kCount);
                std::vector<float> dequant(kCount);
                for (uint32_t i = 0; i < kCount; ++i) {
                    q[i] = core::QuantizeSymmetricS8(d.values[i], scale,
                                                     core::QuantizeRounding::RoundNearestEven);
                    dequant[i] = core::DequantizeSymmetricS8(q[i], scale);
                }
                const reference::QuantErrorStats s =
                    reference::ComputeQuantError(d.values.data(), q.data(), dequant.data(), kCount);
                char line[256];
                std::snprintf(line, sizeof(line), "%s,%s,%.9g,%.6g,%.6g,%.6g,%.2f,%.3f\n",
                              d.name, policyNames[p], scale, s.maxAbsErr, s.mae, s.rmse,
                              s.sqnrDb, s.clipFrac * 100.0);
                report += line;
            }
        }
        SaveReport(ReportsDirM08(), "distributions.csv", report);
        std::printf("distribution error table: %zu rows -> build/reports/m08/distributions.csv\n",
                    dists.size() * 2);
    }

    // --- Part E (C09): AMD's shipped QuantizeLinear on gfx1101 vs the CPU
    // contract. Finite inputs: bit-exact required. Special inputs (NaN/Inf):
    // characterization only (recorded, reproducibility-enforced). ---
    {
        gpu::DeviceContext ctx;
        std::string error;
        if (!gpu::CreateComputeDevice(ctx, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        PrintAdapter(ctx.adapter);

        gpu::ComPtr<ID3D12RootSignature> rootSignature;
        if (!gpu::CreateGemmTileRootSignature(ctx, rootSignature, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }

        // Deterministic 256-value fp16 test vector per scale: grid points,
        // halfway ties (exact for the power-of-two scale), saturators,
        // tiny/denormal fp16s, zeros.
        const float scales[4] = {1.0f, 2.0f, 0.0078125f, 0.014884727f};
        const char* scaleNames[4] = {"s1", "s2p", "spow2", "sfsr"};
        const char* csos[4] = {"quantize_s1", "quantize_s2p", "quantize_spow2", "quantize_sfsr"};

        uint16_t inputs[256];
        uint32_t idx = 0;
        for (int k = -64; k < 64; ++k) // grid points
            inputs[idx++] = core::F32ToF16(float(k) * scales[2]);
        for (int k = -32; k < 32; ++k) // halfway ties
            inputs[idx++] = core::F32ToF16((float(k) + 0.5f) * scales[2]);
        const float saturators[] = {127.49f, 127.5f, 127.51f, 128.49f, 200.0f, 1000.0f,
                                    300.0f, 65504.0f};
        for (float s : saturators) {
            inputs[idx++] = core::F32ToF16(s * 1.0f);
            inputs[idx++] = core::F32ToF16(-s * 1.0f);
            inputs[idx++] = core::F32ToF16(s * 0.0078125f);
            inputs[idx++] = core::F32ToF16(-s * 0.0078125f);
        }
        for (uint16_t pat = 0x0001; pat <= 0x0008; ++pat) { // fp16 denormals
            inputs[idx++] = pat;
            inputs[idx++] = uint16_t(pat | 0x8000);
        }
        inputs[idx++] = 0x0000;  // +0
        inputs[idx++] = 0x8000;  // -0
        while (idx < 256)
            inputs[idx++] = core::F32ToF16(float(idx % 251) - 125.0f); // filler

        for (uint32_t s = 0; s < 4 && ok; ++s) {
            std::vector<uint8_t> cso;
            if (!gpu::LoadShaderBytecode(csos[s], cso, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                ok = false;
                break;
            }
            D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
            psoDesc.pRootSignature = rootSignature.Get();
            psoDesc.CS = {cso.data(), cso.size()};
            gpu::ComPtr<ID3D12PipelineState> pso;
            if (FAILED(ctx.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)))) {
                std::fprintf(stderr, "error: CreateComputePipelineState (%s) failed\n", csos[s]);
                ok = false;
                break;
            }

            gpu::MatmulRunResult timing;
            std::vector<int32_t> outWarmup;
            std::vector<int32_t> outTimed;
            if (!gpu::RunGemmTile(ctx, rootSignature, pso, sizeof(inputs), 256,
                                  1, 1, 1, 1, inputs,
                                  outWarmup, outTimed, timing, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                ok = false;
                break;
            }

            // CPU contract on the same fp16 inputs.
            const uint8_t* hwBytes = reinterpret_cast<const uint8_t*>(outWarmup.data());
            size_t mismatches = 0;
            for (uint32_t c = 0; c < 256; ++c) {
                const float v = core::F16ToF32(inputs[c]);
                const int8_t qCpu = core::QuantizeSymmetricS8(
                    v, scales[s], core::QuantizeRounding::RoundNearestEven);
                if (int8_t(hwBytes[c]) != qCpu) {
                    if (mismatches < 8)
                        std::fprintf(stderr, "  [%s] c=%u v=%.9g: hw %d vs cpu %d\n",
                                     scaleNames[s], c, v, hwBytes[c], qCpu);
                    ++mismatches;
                }
            }
            std::printf("quantize_amd %s (scale %.9g): %zu/256 mismatches vs CPU contract\n",
                        scaleNames[s], scales[s], mismatches);
            if (mismatches)
                ok = false;

            // Idempotency: the timed snapshot must equal the warmup snapshot.
            if (memcmp(outTimed.data(), outWarmup.data(), 256) != 0) {
                std::fprintf(stderr, "  [%s] timed snapshot differs from warmup\n", scaleNames[s]);
                ok = false;
            }
        }

        // Special-value characterization (NaN/Inf through AMD's operator):
        // recorded to the report; pass = reproducible across two dispatches.
        if (ok) {
            std::vector<uint8_t> cso;
            if (!gpu::LoadShaderBytecode("quantize_s1", cso, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                ok = false;
            } else {
                D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
                psoDesc.pRootSignature = rootSignature.Get();
                psoDesc.CS = {cso.data(), cso.size()};
                gpu::ComPtr<ID3D12PipelineState> pso;
                if (FAILED(ctx.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)))) {
                    std::fprintf(stderr, "error: CreateComputePipelineState (special) failed\n");
                    ok = false;
                } else {
                    const uint16_t special[256] = {
                        0x7E00, 0x7C01, 0xFC01, 0x7C00, 0xFC00, 0x8000, 0x0001, 0x8001,
                    }; // qNaN, sNaN payloads, +inf, -inf, -0, min denormals (repeated)
                    uint16_t specialIn[256];
                    for (uint32_t i = 0; i < 256; ++i)
                        specialIn[i] = special[i % 8];

                    gpu::MatmulRunResult timing;
                    std::vector<int32_t> run1;
                    std::vector<int32_t> run2;
                    std::vector<int32_t> outTimed;
                    if (!gpu::RunGemmTile(ctx, rootSignature, pso, sizeof(specialIn), 256,
                                          1, 1, 1, 1, specialIn,
                                          run1, outTimed, timing, error) ||
                        !gpu::RunGemmTile(ctx, rootSignature, pso, sizeof(specialIn), 256,
                                          1, 1, 1, 1, specialIn,
                                          run2, outTimed, timing, error)) {
                        std::fprintf(stderr, "error: %s\n", error.c_str());
                        ok = false;
                    } else {
                        // Position-dependence probe: the same 8 values shifted
                        // by one channel — distinguishes value-based quirks
                        // from lane/position-based ones (matters for M09 NaN
                        // containment).
                        uint16_t rotated[256];
                        for (uint32_t i = 0; i < 256; ++i)
                            rotated[i] = special[(i + 1) % 8];
                        std::vector<int32_t> run3;
                        std::vector<int32_t> outTimed3;
                        if (!gpu::RunGemmTile(ctx, rootSignature, pso, sizeof(rotated), 256,
                                              1, 1, 1, 1, rotated,
                                              run3, outTimed3, timing, error)) {
                            std::fprintf(stderr, "error: %s\n", error.c_str());
                            ok = false;
                        }

                        const bool reproducible =
                            memcmp(run1.data(), run2.data(), 256) == 0;
                        // Byte address == channel index in the int8 tensor
                        // (z-stride = 1): byte c holds channel c's result.
                        std::string report =
                            "AMD QuantizeLinear special-value characterization (scale 1.0)\n"
                            "input_bits -> packed byte (= channel index) (cpu policy)\n";
                        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(run1.data());
                        const uint8_t* rotBytes = reinterpret_cast<const uint8_t*>(run3.data());
                        // Position-independence is ENFORCED: the result of a
                        // value must not depend on its channel index.
                        size_t posMismatches = 0;
                        for (uint32_t i = 0; i < 8; ++i) {
                            const float v = core::F16ToF32(special[i]);
                            // Rotated dispatch: value special[i] sits at the
                            // previous channel index.
                            const uint32_t rotIdx = (i + 7) % 8;
                            if (rotBytes[rotIdx] != bytes[i])
                                ++posMismatches;
                            char line[160];
                            std::snprintf(line, sizeof(line),
                                          "  0x%04X (v=%.9g) -> 0x%02X (cpu %d) | at channel %u -> 0x%02X\n",
                                          special[i], v, bytes[i],
                                          core::QuantizeSymmetricS8(
                                              v, 1.0f, core::QuantizeRounding::RoundNearestEven),
                                          rotIdx, rotBytes[rotIdx]);
                            report += line;
                        }
                        if (posMismatches) {
                            std::fprintf(stderr, "  position-dependence detected: %zu mismatches\n",
                                         posMismatches);
                            ok = false;
                        }
                        // CPU-policy equality for the special values is also
                        // enforced — F9's containment claim depends on it.
                        size_t policyMismatches = 0;
                        for (uint32_t i = 0; i < 8; ++i) {
                            const int8_t cpuQ = core::QuantizeSymmetricS8(
                                core::F16ToF32(special[i]), 1.0f,
                                core::QuantizeRounding::RoundNearestEven);
                            if (int8_t(bytes[i]) != cpuQ)
                                ++policyMismatches;
                        }
                        if (policyMismatches) {
                            std::fprintf(stderr, "  special-value CPU-policy mismatch: %zu\n",
                                         policyMismatches);
                            ok = false;
                        }
                        SaveReport(ReportsDirM08(), "special_values.txt", report);
                        std::printf("special values (NaN/Inf): %s; details -> "
                                    "build/reports/m08/special_values.txt\n",
                                    reproducible ? "reproducible across runs" : "NOT REPRODUCIBLE");
                        if (!reproducible)
                            ok = false;
                    }
                }
            }
        }

        std::string queueErrors;
        if (!gpu::InfoQueueClean(ctx, queueErrors)) {
            std::fprintf(stderr, "%s", queueErrors.c_str());
            ok = false;
        } else {
            std::printf("diagnostics: no error-severity messages\n");
        }
    }

    std::printf("%s\n", ok ? "quantization probe: PASS" : "quantization probe: FAIL");
    return ok ? 0 : 1;
}

int RunInt8Kernels() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Deterministic LCG (shared with the other experiments' data style).
    uint32_t seed = 0x9E3779B9u;
    const auto next01 = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return float(seed >> 8) / float(1 << 24);
    };
    const auto nextI8 = [&](int lo, int hi) {
        return int8_t(lo + int(next01() * float(hi - lo + 1)));
    };

    bool ok = true;

    // --- Device + extensions (analyzer needed for ISA evidence, C07). ---
    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    PrintAdapter(ctx.adapter);

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

    // Loads a CSO, creates the analyzer PSO (usable for dispatch), dumps ISA.
    auto LoadKernelPso = [&](const char* csoName, bool requireWmma) ->
                         gpu::ComPtr<ID3D12PipelineState> {
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
            SaveReport(ReportsDirM09(), std::string(csoName) + ".isa", isa);
            const size_t wmmaCount = CountIsaLines(isa, "v_wmma_i32_16x16x16_iu8");
            std::printf("kernel %-22s | wmma x%zu | %s\n", csoName, wmmaCount, statsLine.c_str());
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

    std::printf("kernels (analyzer PSOs, ISA evidence -> build/reports/m09/):\n");
    // Per-shape fixed-graph CSOs (K tiles + padded M/N + blob offsets baked
    // as -D defines — the ML2Code codegen model; see gemm_i8.hlsl).
    const gpu::ComPtr<ID3D12PipelineState> rawM16K64N16 = LoadKernelPso("gemm_raw_m16k64n16", true);
    const gpu::ComPtr<ID3D12PipelineState> rawM32K64N32 = LoadKernelPso("gemm_raw_m32k64n32", true);
    const gpu::ComPtr<ID3D12PipelineState> rawM16K256N16 = LoadKernelPso("gemm_raw_m16k256n16", true);
    const gpu::ComPtr<ID3D12PipelineState> rawM64K144N16 = LoadKernelPso("gemm_raw_m64k144n16", true);
    const gpu::ComPtr<ID3D12PipelineState> rawM64K144N32 = LoadKernelPso("gemm_raw_m64k144n32", true);
    const gpu::ComPtr<ID3D12PipelineState> rawM256K64N256 = LoadKernelPso("gemm_raw_m256k64n256", true);
    const gpu::ComPtr<ID3D12PipelineState> rawM256K256N256 = LoadKernelPso("gemm_raw_m256k256n256", true);
    const gpu::ComPtr<ID3D12PipelineState> requantM32N24 = LoadKernelPso("gemm_requant_m32n24", false);
    const gpu::ComPtr<ID3D12PipelineState> requantM64N16 = LoadKernelPso("gemm_requant_m64n16", false);
    const gpu::ComPtr<ID3D12PipelineState> requantM64N20 = LoadKernelPso("gemm_requant_m64n20", false);
    if (!ok) {
        // Analyzer rejects are fatal — dispatching a null PSO would crash.
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }

    // --- Part 1 (C01/C02): GEMM correctness, aligned + boundary shapes. ---
    struct GemmShape {
        uint32_t m, n, k;
        uint32_t mp, np; // padded (baked into the CSO)
        ID3D12PipelineState* pso;
        const char* label;
    };
    const GemmShape shapes[] = {
        {16, 16, 64, 16, 16, rawM16K64N16.Get(), "16x16x64"},
        {20, 24, 64, 32, 32, rawM32K64N32.Get(), "20x24x64 (boundary)"},
        {16, 16, 256, 16, 16, rawM16K256N16.Get(), "16x16x256"},
    };
    for (const GemmShape& shape : shapes) {
        if (!ok)
            break;
        // Packed operands: bounded range keeps the int32 accumulator safe.
        std::vector<int8_t> a(size_t(shape.mp) * shape.k);
        std::vector<int8_t> b(size_t(shape.k) * shape.np);
        for (uint32_t m = 0; m < shape.m; ++m)
            for (uint32_t k = 0; k < shape.k; ++k)
                a[m * shape.k + k] = nextI8(-40, 40);
        for (uint32_t k = 0; k < shape.k; ++k)
            for (uint32_t n = 0; n < shape.n; ++n)
                b[k * shape.np + n] = nextI8(-40, 40);

        // Overflow guard (gate: prevented): exact per-output bound.
        int64_t bound = 0;
        for (uint32_t m = 0; m < shape.m; ++m)
            for (uint32_t n = 0; n < shape.n; ++n) {
                int64_t boundMN = 0;
                for (uint32_t k = 0; k < shape.k; ++k)
                    boundMN += int64_t(std::abs(a[m * shape.k + k])) *
                               int64_t(std::abs(b[k * shape.np + n]));
                bound = std::max(bound, boundMN);
            }
        if (bound >= (1LL << 31)) {
            std::fprintf(stderr, "error: overflow guard should have passed for %s "
                                 "(bound %lld) - data/policy inconsistency\n",
                         shape.label, (long long)bound);
            ok = false;
            break;
        }

        // Blob: A | B | zeroTile (int32 16x16, accumulator init; the Fill
        // opcode is driver-rejected — loaded instead, M06 pattern). Blob
        // layout matches the CSO's baked offsets.
        std::vector<uint8_t> blob(size_t(shape.mp) * shape.k + size_t(shape.k) * shape.np + 1024);
        memcpy(blob.data(), a.data(), a.size());
        memcpy(blob.data() + a.size(), b.data(), b.size());
        memset(blob.data() + a.size() + b.size(), 0, 1024);

        gpu::MatmulRunResult timing;
        std::vector<int32_t> outWarmup;
        std::vector<int32_t> outTimed;
        if (!gpu::RunGemmTile(ctx, rootSignature, shape.pso, blob.size(),
                              shape.mp * shape.np * 4u, shape.np / 16, shape.mp / 16,
                              1, 1, blob.data(), outWarmup, outTimed, timing, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
            break;
        }

        // CPU reference over the PADDED operands (padding must not change
        // the real outputs), cropped to the real M/N.
        std::vector<int64_t> ref(size_t(shape.mp) * shape.np);
        reference::CpuInt8Gemm(a.data(), b.data(), ref.data(), shape.mp, shape.np, shape.k);
        size_t mismatches = 0;
        for (uint32_t m = 0; m < shape.m; ++m)
            for (uint32_t n = 0; n < shape.n; ++n)
                if (outWarmup[size_t(m) * shape.np + n] != ref[size_t(m) * shape.np + n]) {
                    if (mismatches < 4)
                        std::fprintf(stderr, "  [%s] (%u,%u) gpu %d vs ref %lld\n",
                                     shape.label, m, n, outWarmup[size_t(m) * shape.np + n],
                                     (long long)ref[size_t(m) * shape.np + n]);
                    ++mismatches;
                }
        // Idempotency: timed snapshot equals warmup snapshot.
        const bool idempotent = memcmp(outTimed.data(), outWarmup.data(),
                                       outWarmup.size() * sizeof(int32_t)) == 0;
        std::printf("gemm_raw %-18s | %zu mismatches | idempotent %s | min %.2f us\n",
                    shape.label, mismatches, idempotent ? "OK" : "FAIL",
                    timing.minMs * 1000.0);
        if (mismatches || !idempotent)
            ok = false;
    }

    // --- Part 2 (C05/C06): requant at store vs the AMD FNB_CT2D contract. ---
    {
        const uint32_t m = 20, n = 24, k = 64;
        const uint32_t mp = 32, np = 32;
        const float sIn = 0.014884727f;
        const float sW = 0.02f;
        const float sOut = 0.5f;
        const float combined = sIn * sW;   // fp32 product — same op as the shader
        const float rcpOut = 1.0f / sOut;

        std::vector<int8_t> a(size_t(mp) * k);
        std::vector<int8_t> b(size_t(k) * np);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t kk = 0; kk < k; ++kk)
                a[i * k + kk] = nextI8(-40, 40);
        for (uint32_t kk = 0; kk < k; ++kk)
            for (uint32_t j = 0; j < n; ++j)
                b[kk * np + j] = nextI8(-40, 40);

        std::vector<uint16_t> bias(np, 0); // padded bias (unused lanes)
        for (uint32_t j = 0; j < n; ++j)
            bias[j] = core::F32ToF16(float(int(next01() * 21) - 10) * 0.25f);

        std::vector<int64_t> ref64(size_t(mp) * np);
        reference::CpuInt8Gemm(a.data(), b.data(), ref64.data(), mp, np, k);
        std::vector<int8_t> refQ(size_t(m) * n);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j)
                refQ[i * n + j] = reference::CpuRequantElement(
                    ref64[size_t(i) * np + j], bias[j], combined, rcpOut);

        // Blob for the requant pass: raw accumulator (int32, padded) | bias.
        // Layout matches the CSO's baked offsets (acc at 0, bias after).
        std::vector<uint8_t> blob(size_t(mp) * np * 4 + np * 2);
        for (size_t i = 0; i < ref64.size(); ++i) {
            const int32_t v = int32_t(ref64[i]); // guard-verified: no wrap
            memcpy(blob.data() + i * 4, &v, 4);
        }
        memcpy(blob.data() + size_t(mp) * np * 4, bias.data(), np * 2);

        gpu::MatmulRunResult timing;
        std::vector<int32_t> outWarmup;
        std::vector<int32_t> outTimed;
        if (!gpu::RunGemmTile(ctx, rootSignature, requantM32N24.Get(), blob.size(), mp * np,
                              (mp * np / 4 + 63) / 64, 1, 1, 1, blob.data(),
                              outWarmup, outTimed, timing, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
        } else {
            const uint8_t* qHw = reinterpret_cast<const uint8_t*>(outWarmup.data());
            size_t mismatches = 0;
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < n; ++j)
                    if (int8_t(qHw[size_t(i) * np + j]) != refQ[i * n + j]) {
                        if (mismatches < 4)
                            std::fprintf(stderr, "  requant (%u,%u): gpu %d vs ref %d\n",
                                         i, j, int8_t(qHw[size_t(i) * np + j]), refQ[i * n + j]);
                        ++mismatches;
                    }
            std::printf("gemm_requant 20x24x64 | %zu mismatches vs AMD-contract reference "
                        "(sIn=%.9g sW=%.9g sOut=%.9g) | min %.2f us\n",
                        mismatches, sIn, sW, sOut, timing.minMs * 1000.0);
            if (mismatches)
                ok = false;
        }
    }

    // --- Part 3 (overflow guard): the pack-time bound vs INT32 accumulation.
    // Fact: int8 x int8 products accumulate in INT32 without overflow for any
    // K <= ~133k even at the full [-128, 127] operand range (127*127*K <
    // 2^31); the guard exists for pathological K / large accumulated inits.
    // Demonstrate both sides of the policy on synthetic bounds. ---
    {
        const int64_t boundK256 = int64_t(256) * 127 * 127;      // 4.13e6
        const int64_t boundK200k = int64_t(200000) * 127 * 127;  // 3.23e9 > 2^31
        const bool pass256 = boundK256 < (1LL << 31);
        const bool refuse200k = boundK200k >= (1LL << 31);
        std::printf("overflow guard: K=256 worst-case bound %lld -> allowed; "
                    "K=200000 bound %lld -> %s\n",
                    (long long)boundK256, (long long)boundK200k,
                    refuse200k ? "REFUSED (prevented)" : "NOT REFUSED");
        if (!pass256 || !refuse200k)
            ok = false;
    }

    // --- Part 4 (C03/C04): 3x3 stride-1 pad-1 convolution via
    // im2col+GEMM+requant vs the direct quantized CPU convolution. ---
    {
        const uint32_t h = 8, w = 8, cin = 16, cout = 16; // aligned conv
        const uint32_t h2 = 7, w2 = 7, cout2 = 20;        // boundary conv (M=49, N=20)
        const float sIn = 0.014884727f;
        const float sW = 0.02f;
        const float sOut = 0.5f;

        for (const bool boundary : {false, true}) {
            if (!ok)
                break;
            const uint32_t hh = boundary ? h2 : h;
            const uint32_t ww = boundary ? w2 : w;
            const uint32_t co = boundary ? cout2 : cout;
            const uint32_t mm = hh * ww; // real GEMM rows
            const uint32_t mp = (mm + 15) & ~15u;
            const uint32_t np = (co + 15) & ~15u;
            const uint32_t k = 9 * cin;

            // Deterministic int8 input/weights with negatives + bias. Two
            // input copies: unpadded (the CPU reference's layout, it does its
            // own bounds-checked zero padding) and padded (for im2col pack).
            std::vector<int8_t> inputUnpadded(size_t(hh) * ww * cin);
            std::vector<int8_t> inputPadded(size_t(hh + 2) * (ww + 2) * cin, 0);
            std::vector<int8_t> weights(size_t(co) * 9 * cin);
            std::vector<uint16_t> bias(np, 0);
            for (uint32_t y = 0; y < hh; ++y)
                for (uint32_t x = 0; x < ww; ++x)
                    for (uint32_t c = 0; c < cin; ++c) {
                        const int8_t v = nextI8(-40, 40);
                        inputUnpadded[(y * ww + x) * cin + c] = v;
                        inputPadded[((y + 1) * (ww + 2) + (x + 1)) * cin + c] = v;
                    }
            for (uint32_t i = 0; i < weights.size(); ++i)
                weights[i] = nextI8(-32, 32);
            for (uint32_t f = 0; f < co; ++f)
                bias[f] = core::F32ToF16(float(int(next01() * 17) - 8) * 0.25f);

            // CPU direct conv reference (the independent check).
            std::vector<int8_t> convRef(size_t(hh) * ww * co);
            reference::CpuConv3x3p1I8(inputUnpadded.data(), weights.data(), bias.data(),
                                      convRef.data(), hh, ww, cin, co, sIn, sW, sOut);

            // CPU im2col pack: A (Mp rows x K) zero-padded; B (K x Np).
            std::vector<int8_t> a(size_t(mp) * k, 0);
            std::vector<int8_t> b(size_t(k) * np, 0);
            for (uint32_t y = 0; y < hh; ++y)
                for (uint32_t x = 0; x < ww; ++x)
                    for (uint32_t ky = 0; ky < 3; ++ky)
                        for (uint32_t kx = 0; kx < 3; ++kx) {
                            const int32_t iy = int32_t(y + ky) - 1;
                            const int32_t ix = int32_t(x + kx) - 1;
                            if (iy < 0 || iy >= int32_t(hh) || ix < 0 || ix >= int32_t(ww))
                                continue; // zero pad
                            for (uint32_t c = 0; c < cin; ++c) {
                                const uint32_t kk = (ky * 3 + kx) * cin + c;
                                a[(y * ww + x) * k + kk] =
                                    inputPadded[(uint32_t(iy + 1) * (ww + 2) + uint32_t(ix + 1)) * cin + c];
                            }                        }
            for (uint32_t f = 0; f < co; ++f)
                for (uint32_t ky = 0; ky < 3; ++ky)
                    for (uint32_t kx = 0; kx < 3; ++kx)
                        for (uint32_t c = 0; c < cin; ++c) {
                            const uint32_t kk = (ky * 3 + kx) * cin + c;
                            b[size_t(kk) * np + f] = weights[(f * 9 + ky * 3 + kx) * cin + c];
                        }

            // Overflow guard on the actual packed data.
            int64_t bound = 0;
            for (uint32_t m = 0; m < mm; ++m)
                for (uint32_t n = 0; n < co; ++n) {
                    int64_t boundMN = 0;
                    for (uint32_t kk = 0; kk < k; ++kk)
                        boundMN += int64_t(std::abs(a[size_t(m) * k + kk])) *
                                   int64_t(std::abs(b[size_t(kk) * np + n]));
                    bound = std::max(bound, boundMN);
                }
            if (bound >= (1LL << 31)) {
                std::fprintf(stderr, "error: conv overflow guard refused valid data\n");
                ok = false;
                break;
            }

            // Blob: A | B | bias | zeroTile — layout matches the CSO's baked
            // offsets (A@0, B@Mp*K, bias@Mp*K+K*Np, zero after bias).
            std::vector<uint8_t> blob(a.size() + b.size() + np * 2 + 1024);
            memcpy(blob.data(), a.data(), a.size());
            memcpy(blob.data() + a.size(), b.data(), b.size());
            memcpy(blob.data() + a.size() + b.size(), bias.data(), np * 2);
            memset(blob.data() + a.size() + b.size() + np * 2, 0, 1024);

            gpu::MatmulRunResult timing;
            std::vector<int32_t> accWarmup;
            std::vector<int32_t> accTimed;
            ID3D12PipelineState* convGemmPso = boundary ? rawM64K144N32.Get() : rawM64K144N16.Get();
            if (!gpu::RunGemmTile(ctx, rootSignature, convGemmPso, blob.size(), mp * np * 4u,
                                  np / 16, mp / 16, 1, 1, blob.data(),
                                  accWarmup, accTimed, timing, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                ok = false;
                break;
            }

            // Requant input blob: raw accumulator | bias (layout matches the
            // requant CSO's baked offsets: acc@0, bias@Mp*Np*4).
            std::vector<uint8_t> rqBlob(size_t(mp) * np * 4 + np * 2);
            memcpy(rqBlob.data(), accWarmup.data(), size_t(mp) * np * 4);
            memcpy(rqBlob.data() + size_t(mp) * np * 4, bias.data(), np * 2);
            gpu::MatmulRunResult rqTiming;
            std::vector<int32_t> qWarmup;
            std::vector<int32_t> qTimed;
            ID3D12PipelineState* rqPso = boundary ? requantM64N20.Get() : requantM64N16.Get();
            if (!gpu::RunGemmTile(ctx, rootSignature, rqPso, rqBlob.size(), mp * np,
                                  (mp * np / 4 + 63) / 64, 1, 1, 1, rqBlob.data(),
                                  qWarmup, qTimed, rqTiming, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                ok = false;
                break;
            }

            const uint8_t* qHw = reinterpret_cast<const uint8_t*>(qWarmup.data());
            size_t mismatches = 0;
            for (uint32_t y = 0; y < hh; ++y)
                for (uint32_t x = 0; x < ww; ++x)
                    for (uint32_t f = 0; f < co; ++f)
                        if (int8_t(qHw[(size_t(y) * ww + x) * np + f]) !=
                            convRef[(size_t(y) * ww + x) * co + f]) {
                            if (mismatches < 4)
                                std::fprintf(stderr, "  conv %s (%u,%u,f%u): gpu %d vs ref %d\n",
                                             boundary ? "boundary" : "aligned", y, x, f,
                                             int8_t(qHw[(size_t(y) * ww + x) * np + f]),
                                             convRef[(size_t(y) * ww + x) * co + f]);
                            ++mismatches;
                        }
            std::printf("conv3x3p1 %dx%dx%d->%d %s | %zu mismatches vs direct quantized CPU conv\n",
                        hh, ww, cin, co, boundary ? "(boundary)" : "        ", mismatches);
            if (mismatches)
                ok = false;
        }
    }

    // --- Part 5 (C08): benchmarks, run twice for reproducibility. ---
    if (ok) {
        const auto benchGemm = [&](ID3D12PipelineState* pso, uint32_t mp, uint32_t np,
                                   uint32_t k, double& minUs) -> bool {
            std::vector<int8_t> a(size_t(mp) * k);
            std::vector<int8_t> b(size_t(k) * np);
            for (uint32_t i = 0; i < a.size(); ++i)
                a[i] = nextI8(-32, 32);
            for (uint32_t i = 0; i < b.size(); ++i)
                b[i] = nextI8(-32, 32);
            // Blob layout matches the CSO's baked offsets: A@0 | B@Mp*K | zero.
            std::vector<uint8_t> blob(a.size() + b.size() + 1024);
            memcpy(blob.data(), a.data(), a.size());
            memcpy(blob.data() + a.size(), b.data(), b.size());
            memset(blob.data() + a.size() + b.size(), 0, 1024);
            gpu::MatmulRunResult timing;
            std::vector<int32_t> outWarmup;
            std::vector<int32_t> outTimed;
            if (!gpu::RunGemmTile(ctx, rootSignature, pso, blob.size(), mp * np * 4u,
                                  np / 16, mp / 16, 10, 100, blob.data(),
                                  outWarmup, outTimed, timing, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                return false;
            }
            minUs = timing.minMs * 1000.0;
            return true;
        };

        double rawK64[2];
        double rawK256[2];
        // Discarded warm cycle: cross-run clock state dominates otherwise
        // (M07 finding: within-run min-of-100 is the reliable metric).
        {
            double discard = 0.0;
            benchGemm(rawM256K64N256.Get(), 256, 256, 64, discard);
            benchGemm(rawM256K256N256.Get(), 256, 256, 256, discard);
        }
        for (uint32_t r = 0; r < 2 && ok; ++r) {
            if (!benchGemm(rawM256K64N256.Get(), 256, 256, 64, rawK64[r]) ||
                !benchGemm(rawM256K256N256.Get(), 256, 256, 256, rawK256[r]))
                ok = false;
        }
        if (ok) {
            const double k64Spread = rawK64[0] > 0 ? std::fabs(rawK64[1] - rawK64[0]) / rawK64[0] : 1.0;
            const double k256Spread = rawK256[0] > 0 ? std::fabs(rawK256[1] - rawK256[0]) / rawK256[0] : 1.0;
            std::printf("benchmark gemm 256x256: K64 min %.2f/%.2f us (spread %.1f%%), "
                        "K256 min %.2f/%.2f us (spread %.1f%%), 10 warmup + 100 timed\n",
                        rawK64[0], rawK64[1], k64Spread * 100.0,
                        rawK256[0], rawK256[1], k256Spread * 100.0);
            if (k64Spread > 0.10 || k256Spread > 0.10) {
                std::fprintf(stderr, "error: benchmark reproducibility spread > 10%%\n");
                ok = false;
            }
        }
    }

    std::string queueErrors;
    size_t benignCount = 0;
    if (!gpu::InfoQueueCleanWithFilter(
            ctx,
            [](const D3D12_MESSAGE& msg) {
                if (!msg.pDescription)
                    return false;
                const char* desc = msg.pDescription;
                return strstr(desc, "GPU-BASED VALIDATION") != nullptr &&
                       strstr(desc, "Root descriptor access out of bounds") != nullptr &&
                       strstr(desc, "Root Parameter Index: [2]") != nullptr;
            },
            benignCount, queueErrors)) {
        std::fprintf(stderr, "%s", queueErrors.c_str());
        ok = false;
    } else {
        std::printf("diagnostics: no unexplained error-severity messages (%zu known-benign "
                    "GBAV intrinsics-dummy reports, see EXP-004)\n", benignCount);
    }

    gpu::ReleaseAmdExtensions(ext);
    std::printf("%s\n", ok ? "int8 kernels probe: PASS" : "int8 kernels probe: FAIL");
    return ok ? 0 : 1;
}

int RunComputeHarness() {
    // Deterministic input covering negative, positive and fractional values.
    const size_t count = 1u << 20;
    std::vector<float> a(count), b(count), expected(count);
    for (size_t i = 0; i < count; ++i) {
        a[i] = float(i % 1024) * 0.25f - 128.0f;      // [-128, 127.75]
        b[i] = float((i * 7u) % 511u) * 0.5f - 100.0f; // [-100, 154.5]
    }
    reference::CpuVectorAdd(a.data(), b.data(), expected.data(), count);

    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    PrintAdapter(ctx.adapter);

    gpu::VectorAddOutputs outputs;
    gpu::ComputeRunResult timing;
    if (!gpu::RunVectorAdd(ctx, count, /*warmup=*/10, /*timed=*/100,
                           a.data(), b.data(), outputs, timing, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    // Compare both snapshots against the independent CPU reference.
    const size_t mismatchWarmup =
        reference::CountMismatches(outputs.postWarmup.data(), expected.data(), count);
    const size_t mismatchTimed =
        reference::CountMismatches(outputs.postTimed.data(), expected.data(), count);

    std::printf("workload: float32 vector add, %zu elements (%zu KiB/field)\n",
                count, count * sizeof(float) >> 10);
    std::printf("dispatches: 10 warmup + %u timed (%u groups x 256 threads)\n",
                timing.iterations, (unsigned)((count + 255) / 256));
    std::printf("gpu time: min %.3f us, avg %.3f us, max %.3f us\n",
                timing.minMs * 1000.0, timing.avgMs * 1000.0, timing.maxMs * 1000.0);
    std::printf("correctness: post-warmup mismatches %zu, post-timed mismatches %zu (bit-exact expected)\n",
                mismatchWarmup, mismatchTimed);

    bool ok = (mismatchWarmup == 0) && (mismatchTimed == 0);

    // Debug diagnostics: no error-severity messages may remain (gate item).
    std::string queueErrors;
    if (!gpu::InfoQueueClean(ctx, queueErrors)) {
        ok = false;
        std::fprintf(stderr, "%s", queueErrors.c_str());
    } else {
        std::printf("diagnostics: debug layer + GPU-based validation enabled, no error-severity messages\n");
    }

    std::printf("%s\n", ok ? "compute harness: PASS" : "compute harness: FAIL");
    return ok ? 0 : 1;
}

} // namespace experiments
