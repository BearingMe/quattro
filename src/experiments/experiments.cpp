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
