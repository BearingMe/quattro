#include "experiments.h"

#include "../gpu/amdext.h"
#include "../gpu/compute.h"
#include "../gpu/device.h"
#include "../gpu/matmul.h"
#include "../reference/reference.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
                              /*groupCount=*/1, /*warmup=*/1, /*timed=*/100,
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
        fa[i] = reference::F32ToF16(av);
        fb[i] = reference::F32ToF16(bv);
    }
    float initF32[256];
    for (uint32_t i = 0; i < 256; ++i)
        initF32[i] = float(i % 9) - 4.0f;
    uint16_t initF16[256];
    for (uint32_t i = 0; i < 256; ++i)
        initF16[i] = reference::F32ToF16(initF32[i]);

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
                              1, 1, 100, probe.input,
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
                        aF[e] = reference::F32ToF16(float(e % 17) * 0.125f - 1.0f);
                        bF[e] = reference::F32ToF16(float((e * 7u) % 13u) * 0.125f - 0.75f);
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
                                      kGemms, 10, 100, input,
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
