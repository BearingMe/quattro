// M15 controlled baseline. Reuse assembled PSOs; retain all warmup/sample data.
#include "experiments.h"
#include "core/numerics.h"
#include "inference/models/fsr4_graph.h"
#include "inference/models/fsr4_weights.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>

namespace experiments {
namespace {

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int RunFsr4Profile() {
    using namespace inference::fsr4;
    using Clock = std::chrono::steady_clock;
    setvbuf(stdout, nullptr, _IONBF, 0);
    wchar_t exe[MAX_PATH]{};
    if (!GetModuleFileNameW(nullptr, exe, MAX_PATH)) return 1;
    const auto root = std::filesystem::path(exe).parent_path().parent_path().parent_path();
    const char* id = std::getenv("M15_RUN_ID");
    const std::string runId = id ? id : "profile-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64());
    if (runId.empty() || runId.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos)
        return 2;
    const auto directory = root / "build/reports/m15" / runId;
    std::error_code ec;
    std::filesystem::create_directories(directory.parent_path(), ec);
    if (ec || !std::filesystem::create_directory(directory, ec) || ec) return 1;
    const char* policyEnv = std::getenv("M15_INIT_POLICY");
    const std::string policy = policyEnv ? policyEnv : "baseline";
    const char* backendEnv = std::getenv("M15_MATRIX_BACKEND");
    const std::string backend = backendEnv ? backendEnv : "int8";
    if (backend != "int8" && backend != "fp16") {
        std::fprintf(stderr, "M15_MATRIX_BACKEND must be int8 or fp16\n"); return 2;
    }
    if (policy != "baseline" && policy != "poison" && policy != "skip" && policy != "gpu-zero") {
        std::fprintf(stderr, "M15_INIT_POLICY must be baseline, poison, skip or gpu-zero\n"); return 2;
    }
    std::printf("profile directory: %ls | GPU validation ENABLED | initialization policy %s | matrix backend %s\n",
                directory.c_str(), policy.c_str(), backend.c_str());
    std::string error;
    std::vector<uint8_t> weights;
    if (!LoadWeights(root / "build/reports/m12/weights.bin", weights, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    const auto goldenPath = root / "build/reports/m12/graph-captures/resident-release-final/fixture2-pass13-control.bin";
    std::ifstream goldenFile(goldenPath, std::ios::binary);
    std::vector<uint8_t> golden((std::istreambuf_iterator<char>(goldenFile)), std::istreambuf_iterator<char>());
    if (golden.size() != size_t(1920) * 1080 * 16) {
        std::fprintf(stderr, "retained EXP-016 control output missing/wrong size: %ls\n", goldenPath.c_str());
        return 1;
    }
    std::vector<uint8_t> input(size_t(1920) * 1080 * 16, 0);
    uint32_t seed = 0x5EED4F12u;
    for (size_t pixel = 0; pixel < size_t(1920) * 1080; ++pixel) {
        for (uint32_t c = 0; c < 7; ++c) {
            seed = seed * 1664525u + 1013904223u;
            const uint16_t bits = core::F32ToF16(float(int((seed >> 16) % 4095u) - 2047) / 1024.0f);
            memcpy(input.data() + pixel * 16 + c * 2, &bits, 2);
        }
    }
    gpu::DeviceContext ctx;
    if (!gpu::CreateComputeDevice(ctx, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    gpu::AmdExtensions ext;
    if (!gpu::LoadAmdExtensions(ctx, ext, error) || !ext.waveMatrixSupported) {
        gpu::ReleaseAmdExtensions(ext); std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    struct ExtensionLifetime { gpu::AmdExtensions& ext; ~ExtensionLifetime() { gpu::ReleaseAmdExtensions(ext); } } lifetime{ext};
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(ctx, plan.signature, error, true)) return 1;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()}};
    std::vector<size_t> boundaries;
    size_t destination = 0;
    const auto assemblyStart = Clock::now();
    if (!AppendModelGraph(ctx, ext, plan, 0, 1, 13, destination, error, &boundaries)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    const auto int8Passes = plan.passes;
    std::vector<std::string> fp16Stats;
    if (backend == "fp16") {
        if (!UseFp16MatrixComparison(ctx, ext, plan, boundaries, fp16Stats, error)) {
            std::fprintf(stderr, "FP16 matrix comparison: %s\n", error.c_str()); return 1;
        }
        std::printf("11 native FP16/F32 WMMA products; quantized weights, DP4A and numeric epilogues unchanged\n");
    }
    const double assemblyMs = std::chrono::duration<double, std::milli>(Clock::now() - assemblyStart).count();
    std::printf("adapter: %ls | assembly %.3f ms | resources %zu | dispatches %zu | PSOs reused across samples\n",
                ctx.adapter.name.c_str(), assemblyMs, plan.buffers.size(), plan.passes.size());
    std::vector<std::vector<uint8_t>> poison(plan.buffers.size());
    for (size_t i = 0; i < plan.buffers.size(); ++i) {
        if (plan.buffers[i].data)
            continue;
        if (policy == "skip" || policy == "gpu-zero")
            plan.buffers[i].initialize = false;
        else if (policy == "poison") {
            poison[i].assign(plan.buffers[i].bytes, 0xA5);
            plan.buffers[i].data = poison[i].data();
        }
    }
    if (policy == "gpu-zero") {
        size_t maximum = 0;
        for (const auto& buffer : plan.buffers)
            if (!buffer.initialize)
                maximum = std::max(maximum, buffer.bytes);
        const size_t zero = plan.buffers.size();
        plan.buffers.push_back({maximum, nullptr});
        for (size_t i = 0; i < zero; ++i)
            if (!plan.buffers[i].initialize)
                plan.passes.front().copiesBefore.push_back({zero, i, 0, 0, plan.buffers[i].bytes});
        std::printf("GPU-zero candidate: additional %zu-byte zero pool and %zu pre-dispatch copies\n",
                    maximum, plan.passes.front().copiesBefore.size());
    }
    {
        auto invalidBuffers = plan.buffers;
        invalidBuffers[0].initialize = false; // input still has data: contradictory declaration.
        std::vector<uint8_t> output;
        std::vector<std::vector<uint8_t>> snapshots;
        std::string diagnostic;
        if (gpu::RunBufferPasses(ctx, plan.signature, invalidBuffers, plan.passes, destination,
                std::vector<bool>(plan.passes.size(), false), snapshots, destination, 4,
                output, diagnostic, true) || diagnostic.find("cannot supply data") == std::string::npos) {
            std::fprintf(stderr, "initialization policy validation did not fail closed\n"); return 1;
        }
    }
    std::ofstream stats(directory / "shader-stats.txt");
    stats << "Initial graph compiled stats (eleven raw products are replaced when backend=fp16):\n";
    for (size_t i = 0; i < plan.convolutionStats.size(); ++i)
        stats << i << ": " << plan.convolutionStats[i] << '\n';
    for (const auto& line : fp16Stats)
        stats << "Executed FP16 " << line << '\n';
    stats.close();
    if (!stats) return 1;
    // Preserve the original input allocation: BufferInit stores its address.
    const auto savedInput = input;
    for (uint32_t fixture = 0; fixture < 3; ++fixture) {
        if (fixture == 2)
            std::copy(savedInput.begin(), savedInput.end(), input.begin());
        else
            std::fill(input.begin(), input.end(), uint8_t(0));
        if (fixture == 1) {
            const size_t center = size_t(540) * 1920 + 960;
            for (uint32_t c = 0; c < 7; ++c) {
                const uint16_t bits = core::F32ToF16(c & 1 ? -1.0f : 1.0f);
                memcpy(input.data() + center * 16 + c * 2, &bits, 2);
            }
        }
        const auto path = goldenPath.parent_path() / ("fixture" + std::to_string(fixture) + "-pass13-control.bin");
        std::ifstream file(path, std::ios::binary);
        std::vector<uint8_t> expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::vector<uint8_t> output;
        std::vector<std::vector<uint8_t>> snapshots;
        std::vector<bool> capture(plan.passes.size(), false);
        for (size_t boundary : boundaries)
            capture[boundary] = true;
        bool correct = expected.size() == golden.size() &&
            gpu::RunBufferPasses(ctx, plan.signature, plan.buffers, plan.passes, destination,
                capture, snapshots, destination, golden.size(), output, error, true) &&
            output == expected && snapshots.size() == 14;
        for (size_t pass = 0; pass < snapshots.size(); ++pass) {
            const auto controlPath = goldenPath.parent_path() / ("fixture" + std::to_string(fixture) +
                "-pass" + std::to_string(pass) + "-control.bin");
            std::ifstream control(controlPath, std::ios::binary);
            std::vector<uint8_t> reference((std::istreambuf_iterator<char>(control)), std::istreambuf_iterator<char>());
            correct = correct && !reference.empty() && snapshots[pass] == reference;
        }
        if (!correct) {
            std::fprintf(stderr, "policy %s fixture%u exact-control regression failed: %s\n", policy.c_str(), fixture, error.c_str());
            std::string diagnostics;
            size_t benign = 0;
            gpu::InfoQueueCleanWithFilter(ctx, gpu::IsBufferPassWmmaDiagnostic, benign, diagnostics);
            std::fprintf(stderr, "failure validation: benign %zu | %s\n", benign,
                         diagnostics.empty() ? "no other validation errors" : diagnostics.c_str());
            if (!output.empty()) {
                std::ofstream failed(directory / ("failed-fixture" + std::to_string(fixture) + ".bin"), std::ios::binary);
                failed.write(reinterpret_cast<const char*>(output.data()), std::streamsize(output.size()));
            }
            for (size_t pass = 0; pass < snapshots.size(); ++pass) {
                const auto controlPath = goldenPath.parent_path() / ("fixture" + std::to_string(fixture) +
                    "-pass" + std::to_string(pass) + "-control.bin");
                std::ifstream control(controlPath, std::ios::binary);
                std::vector<uint8_t> reference((std::istreambuf_iterator<char>(control)), std::istreambuf_iterator<char>());
                std::fprintf(stderr, "diagnostic pass%zu: %s\n", pass,
                             snapshots[pass] == reference ? "exact" : "DIFFERENT");
            }
            if (backend == "fp16") {
                std::vector<std::vector<uint8_t>> rawValues;
                for (uint32_t variant = 0; variant < 2; ++variant) {
                    auto passes = variant == 0 ? int8Passes : plan.passes;
                    const size_t rawStage = variant == 0 ? 2 : 3;
                    passes[rawStage].snapshotBuffer = passes[rawStage].uav0;
                    std::vector<bool> flags(passes.size(), false);
                    flags[rawStage] = true;
                    std::vector<std::vector<uint8_t>> rawSnapshots;
                    std::vector<uint8_t> ignored;
                    if (!gpu::RunBufferPasses(ctx, plan.signature, plan.buffers, passes, destination,
                            flags, rawSnapshots, destination, 4, ignored, error, true)) {
                        std::fprintf(stderr, "raw comparison diagnostic failed: %s\n", error.c_str()); return 1;
                    }
                    rawValues.push_back(std::move(rawSnapshots.front()));
                }
                size_t different = 0, shown = 0;
                int64_t maximum = 0;
                for (size_t i = 0; i < size_t(518400) * 16; ++i) {
                    int32_t a = 0, b = 0;
                    memcpy(&a, rawValues[0].data() + i * 4, 4);
                    memcpy(&b, rawValues[1].data() + i * 4, 4);
                    if (a != b) {
                        ++different;
                        maximum = std::max(maximum, std::abs(int64_t(a) - int64_t(b)));
                        if (shown++ < 8)
                            std::fprintf(stderr, "raw cnb0 element%zu: I8 %d F16 %d\n", i, a, b);
                    }
                }
                std::fprintf(stderr, "first matrix product: %zu differing I32 elements, max delta %lld\n",
                             different, static_cast<long long>(maximum));
            }
            return 1;
        }
        std::printf("policy %s fixture%u: all 14 boundaries exact retained control\n", policy.c_str(), fixture);
    }
    std::copy(savedInput.begin(), savedInput.end(), input.begin());
    std::ofstream samples(directory / "samples.csv"), dispatches(directory / "dispatches.csv");
    samples << "iteration,warmup,wall_ms,upload_gpu_ms,dispatch_sum_gpu_ms,readback_gpu_ms,total_gpu_ms,default_heap_bytes,upload_bytes,output_bytes\n";
    dispatches << "iteration,warmup,dispatch,model_pass,phase,gpu_ms\n";
    std::vector<std::vector<double>> layerTimes(14);
    std::vector<double> gpuTotals, walls, uploadTimes;
    for (uint32_t iteration = 0; iteration < 9; ++iteration) {
        const bool warmup = iteration < 2;
        std::vector<std::vector<uint8_t>> snapshots;
        std::vector<uint8_t> output;
        std::vector<double> times;
        gpu::BufferRunMetrics metrics;
        const auto start = Clock::now();
        if (!gpu::RunBufferPasses(ctx, plan.signature, plan.buffers, plan.passes, destination,
                std::vector<bool>(plan.passes.size(), false), snapshots, destination, golden.size(),
                output, error, true, &times, &metrics)) {
            std::fprintf(stderr, "%s\n", error.c_str()); return 1;
        }
        const double wallMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (output != golden || times.size() != plan.passes.size()) {
            std::fprintf(stderr, "profile sample violates exact retained control output or timestamp count\n"); return 1;
        }
        const double dispatchSum = std::accumulate(times.begin(), times.end(), 0.0);
        samples << iteration << ',' << warmup << ',' << wallMs << ',' << metrics.uploadGpuMs << ',' << dispatchSum
                << ',' << metrics.outputReadbackGpuMs << ',' << metrics.totalGpuMs << ',' << metrics.defaultHeapBytes
                << ',' << metrics.uploadBytes << ',' << metrics.outputBytes << '\n';
        size_t first = 0;
        for (uint32_t pass = 0; pass < 14; ++pass) {
            const size_t last = boundaries[pass];
            double layer = 0;
            for (size_t stage = first; stage <= last; ++stage) {
                const bool matrix = pass == 1 || pass == 2 || pass == 3 || pass == 6 || pass == 12;
                const size_t offset = (stage - first) % (backend == "fp16" ? 4 : 3);
                const char* phase = !matrix ? "fused" : offset == 0 ? "pack" :
                    backend == "fp16" && offset == 1 ? "i8-to-f16" :
                    offset == (backend == "fp16" ? 2 : 1) ? "wmma" : "post";
                dispatches << iteration << ',' << warmup << ',' << stage << ',' << pass << ',' << phase << ',' << times[stage] << '\n';
                layer += times[stage];
            }
            if (!warmup) layerTimes[pass].push_back(layer);
            first = last + 1;
        }
        if (!warmup) {
            gpuTotals.push_back(metrics.totalGpuMs); walls.push_back(wallMs); uploadTimes.push_back(metrics.uploadGpuMs);
        }
        std::printf("sample %u %s: total GPU %.4f ms | upload %.4f | dispatch sum %.4f | readback %.4f | wall %.3f ms | exact yes\n",
                    iteration, warmup ? "warmup" : "measured", metrics.totalGpuMs, metrics.uploadGpuMs,
                    dispatchSum, metrics.outputReadbackGpuMs, wallMs);
        if (iteration == 0)
            std::printf("memory: default %zu B | upload %zu B | output readback %zu B\n",
                        metrics.defaultHeapBytes, metrics.uploadBytes, metrics.outputBytes);
    }
    if (!samples || !dispatches) return 1;
    std::printf("measured GPU span min/median/max %.4f / %.4f / %.4f ms | median wall %.3f ms | upload median %.4f ms\n",
                *std::min_element(gpuTotals.begin(), gpuTotals.end()), Median(gpuTotals),
                *std::max_element(gpuTotals.begin(), gpuTotals.end()), Median(walls), Median(uploadTimes));
    for (uint32_t pass = 0; pass < 14; ++pass)
        std::printf("pass%u median %.6f ms\n", pass, Median(layerTimes[pass]));
    std::string validation;
    size_t benign = 0;
    if (!gpu::InfoQueueCleanWithFilter(ctx, gpu::IsBufferPassWmmaDiagnostic, benign, validation)) {
        std::fprintf(stderr, "%s\n", validation.c_str()); return 1;
    }
    std::printf("M15 %s/%s collected: 2 warmups + 7 measured, all three fixtures exact | intrinsic messages %zu\n",
                backend.c_str(), policy.c_str(), benign);
    return 0;
}

} // namespace experiments
