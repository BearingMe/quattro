// M12 reference oracle: executes the pinned FSR 4 v07 INT8 Quality (1080p)
// model passes exactly as shipped (AMD-generated HLSL, compiled locally from
// the read-only research tree) and captures the tensors the reproduction must
// match.
//
// Scope: this is the *reference*, not the reproduction. The M12 gate requires
// our own executor; this experiment exists so every stage has a comparable
// numeric truth and so the extracted graph/metadata (scripts/m12_extract_graph.py)
// can be validated against real execution.

#include "experiments.h"

#include "../gpu/compute.h"
#include "../gpu/device.h"
#include "../gpu/graph.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace experiments {
namespace {

constexpr uint32_t kWidth = 1920;
constexpr uint32_t kHeight = 1080;
constexpr uint32_t kInputChannels = 7;       // current/history Y, chroma distance, RFM 4
constexpr uint32_t kInputStorageChannels = 8; // padded to a whole dword
// Pass0's rounded grid can read through input row 1083; pass13 can write
// through output row 1087 (68*16=1088). These are harness safety allocations,
// not a claim about the signed runtime's resource sizes.
constexpr uint32_t kPaddedRows = 1088;
constexpr size_t kInputBytes = size_t(kWidth) * kPaddedRows * kInputStorageChannels * 2;
constexpr size_t kOutputValidBytes = size_t(kWidth) * kHeight * 8 * 2;
constexpr size_t kOutputBytes = size_t(kWidth) * kPaddedRows * 8 * 2;
// The pass11 64-column overhang reaches exactly 1024 bytes beyond the
// documented 20,736,000-byte logical scratch extent. Retain this tail in dumps
// so the out-of-logical-extent stores remain observable without physical OOB.
constexpr size_t kScratchBytes = 20736000 + 1024;

// Non-WMMA (v07-i8) dispatch grids from the provider, 1920x1080.
struct PassGrid { uint32_t x, y, z; };
constexpr PassGrid kGrid[14] = {
    {120, 68, 1}, {15, 540, 1}, {15, 540, 1}, {8, 270, 1}, {8, 270, 1},
    {8, 270, 1},  {4, 135, 1},  {4, 135, 1},  {4, 135, 1}, {30, 17, 1},
    {8, 270, 1},  {8, 270, 1},  {15, 540, 1}, {120, 68, 1},
};

std::filesystem::path RepoRoot() {
    wchar_t exePathW[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exePathW, MAX_PATH) == 0)
        return {};
    return std::filesystem::path(exePathW).parent_path() / ".." / "..";
}

std::filesystem::path ModelDir() {
    return RepoRoot() / ".references/sources/FSR-4.0.2-reference/Kits/FidelityFX/"
                       "upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality";
}

uint64_t Fnv1a(const uint8_t* data, size_t bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

} // namespace

int RunFsr4Oracle() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string error;

    // --- Reference artifacts (read-only research tree; hashes pin identity). ---
    const std::filesystem::path modelDir = ModelDir();
    const std::filesystem::path initializersPath = modelDir / "initializers.bin";
    std::ifstream initializersFile(initializersPath, std::ios::binary);
    if (!initializersFile) {
        std::fprintf(stderr, "error: reference artifacts not found at %ls\n",
                     initializersPath.c_str());
        return 1;
    }
    std::vector<uint8_t> initializers((std::istreambuf_iterator<char>(initializersFile)),
                                      std::istreambuf_iterator<char>());
    if (initializers.size() != 89216 || Fnv1a(initializers.data(), initializers.size()) != 0x1f2b7afc32bdb020ull) {
        std::fprintf(stderr, "error: pinned reference initializer identity mismatch\n");
        return 1;
    }
    std::printf("reference: fsr4_model_v07_i8_quality/initializers.bin %zu B fnv1a=%016llx\n",
                initializers.size(), (unsigned long long)Fnv1a(initializers.data(), initializers.size()));

    gpu::DeviceContext ctx;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::printf("adapter: %ls (vid=0x%04X did=0x%04X)\n", ctx.adapter.name.c_str(),
                ctx.adapter.vendorId, ctx.adapter.deviceId);

    gpu::ComPtr<ID3D12RootSignature> rootSignature;
    if (!gpu::CreateBufferPassRootSignature(ctx, rootSignature, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    char* controlEnv = nullptr;
    size_t controlEnvLen = 0;
    _dupenv_s(&controlEnv, &controlEnvLen, "M12_PREDICATED");
    const bool predicatedControl = controlEnv != nullptr && std::strcmp(controlEnv, "1") == 0;
    free(controlEnv);
    std::printf("mode: %s; scratch is zero-initialized for each invocation\n",
                predicatedControl ? "DERIVATIVE pass11 predication control (NOT shipped reference)"
                                  : "unmodified shipped reference");

    // --- PSOs: the shipped per-pass CSOs, compiled by CMake from the pinned tree. ---
    std::vector<gpu::ComPtr<ID3D12PipelineState>> psos;
    for (uint32_t i = 0; i < 14; ++i) {
        std::vector<uint8_t> cso;
        const std::string name = predicatedControl && i == 11
                                     ? "m12control/pass11" : "m12ref/pass" + std::to_string(i);
        if (!gpu::LoadShaderBytecode(name.c_str(), cso, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = rootSignature.Get();
        desc.CS = {cso.data(), cso.size()};
        gpu::ComPtr<ID3D12PipelineState> pso;
        if (FAILED(ctx.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
            std::fprintf(stderr, "error: CreateComputePipelineState failed for pass%u\n", i);
            return 1;
        }
        psos.push_back(pso);
    }

    // --- Deterministic input features: FP16 NHWC, 7 valid channels. ---
    // Values are exact multiples of 2^-10 with |value| <= 2 (i/1024, |i| <= 2047)
    // so the FP16 bit pattern is produced exactly by construction — the oracle
    // pins the input bits without relying on a float→half conversion path.
    std::vector<uint8_t> input(kInputBytes, 0);
    uint32_t seed = 0x5EED4F12u;
    const auto exactHalf = [](int value) -> uint16_t {
        if (value == 0)
            return 0;
        const uint16_t sign = value < 0 ? 0x8000u : 0u;
        const int magnitude = value < 0 ? -value : value; // i/1024, |i| <= 2047
        int e = 0;
        while ((magnitude >> (e + 1)) != 0)
            ++e;
        const uint16_t exponent = uint16_t(e + 5); // bias 15, value = m * 2^(e-10)
        const uint16_t mantissa = uint16_t((magnitude << (10 - e)) & 0x3FF);
        return sign | (exponent << 10) | mantissa;
    };
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const size_t pixel = (size_t(y) * kWidth + x) * kInputStorageChannels * 2;
            for (uint32_t c = 0; c < kInputChannels; ++c) {
                seed = seed * 1664525u + 1013904223u;
                const int value = int((seed >> 16) % 4095u) - 2047;
                const uint16_t bits = exactHalf(value);
                memcpy(input.data() + pixel + c * 2, &bits, 2);
            }
        }
    }

    std::printf("input features: %zu B fnv1a=%016llx (includes zero padding)\n",
                input.size(), (unsigned long long)Fnv1a(input.data(), input.size()));

    std::vector<gpu::BufferPass> passes;
    for (uint32_t i = 0; i < 14; ++i) {
        gpu::BufferPass p{};
        p.pso = psos[i].Get();
        p.gridX = kGrid[i].x;
        p.gridY = kGrid[i].y;
        p.gridZ = kGrid[i].z;
        p.srv0 = 0; // input features (read by pass 0)
        p.srv1 = 1; // initializers
        p.uav0 = 2; // network output (written by pass 13)
        p.uav1 = 3; // scratch
        passes.push_back(p);
    }

    const std::vector<gpu::BufferInit> buffers = {
        {kInputBytes, input.data()},
        {initializers.size(), initializers.data()},
        {kOutputBytes, nullptr},
        {kScratchBytes, nullptr},
    };

    char* snapshotsEnv = nullptr;
    size_t snapshotsEnvLen = 0;
    _dupenv_s(&snapshotsEnv, &snapshotsEnvLen, "M12_SNAPSHOTS");
    const bool denseSnapshots = snapshotsEnv != nullptr && std::strcmp(snapshotsEnv, "1") == 0;
    free(snapshotsEnv);
    std::vector<bool> snapshotAfter(14, denseSnapshots);

    std::vector<std::vector<uint8_t>> snapshots;
    std::vector<uint8_t> output;
    if (!gpu::RunBufferPasses(ctx, rootSignature, buffers, passes, 3, snapshotAfter,
                              snapshots, 2, kOutputValidBytes, output, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    char* dumpEnv = nullptr;
    size_t dumpEnvLen = 0;
    _dupenv_s(&dumpEnv, &dumpEnvLen, "M12_DUMP");
    const bool dumpFiles = dumpEnv != nullptr && std::strcmp(dumpEnv, "1") == 0;
    free(dumpEnv);
    char* runEnv = nullptr;
    size_t runEnvLen = 0;
    _dupenv_s(&runEnv, &runEnvLen, "M12_RUN_ID");
    const std::string runId = runEnv ? runEnv : "manual";
    free(runEnv);
    if (runId.empty() || runId.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos) {
        std::fprintf(stderr, "error: M12_RUN_ID must contain only letters, digits, '-' or '_'\n");
        return 1;
    }
    const std::filesystem::path dumpDir = RepoRoot() / "build/reports/m12/ref" / runId;
    if (dumpFiles) {
        std::error_code ec;
        // Exclusive directory creation: repeated manual run IDs cannot
        // silently truncate retained evidence. Parent creation is harmless.
        std::filesystem::create_directories(dumpDir.parent_path(), ec);
        if (ec || !std::filesystem::create_directory(dumpDir, ec) || ec) {
            std::fprintf(stderr, "error: reference dump directory exists or cannot be created\n");
            return 1;
        }
    }

    // --- Report: per-pass scratch hashes (dense mode) + final output analysis. ---
    if (denseSnapshots) {
        for (size_t i = 0; i < snapshots.size(); ++i) {
            std::printf("scratch after pass %2zu: fnv1a=%016llx\n", i,
                        (unsigned long long)Fnv1a(snapshots[i].data(), snapshots[i].size()));
            if (dumpFiles) {
                std::ofstream f(dumpDir / ("scratch-pass" + std::to_string(i) + ".bin"),
                                std::ios::binary);
                f.write(reinterpret_cast<const char*>(snapshots[i].data()),
                        std::streamsize(snapshots[i].size()));
                f.close();
                if (!f) {
                    std::fprintf(stderr, "error: cannot write scratch dump %zu\n", i);
                    return 1;
                }
            }
        }
    }

    size_t nanCount = 0, infCount = 0, nonzero = 0;
    float minValue = 1e30f, maxValue = -1e30f;
    for (size_t i = 0; i + 1 < output.size(); i += 2) {
        uint16_t bits;
        memcpy(&bits, output.data() + i, 2);
        const uint32_t sign = (bits >> 15) & 1u;
        const uint32_t exp = (bits >> 10) & 0x1Fu;
        const uint32_t mant = bits & 0x3FFu;
        if (exp == 31) {
            if (mant)
                ++nanCount;
            else
                ++infCount;
            continue;
        }
        float value;
        if (exp == 0) {
            value = std::ldexp(float(mant), -24);
        } else {
            value = std::ldexp(float(mant | 0x400u), int(exp) - 25);
        }
        if (sign)
            value = -value;
        if (value != 0.0f)
            ++nonzero;
        minValue = std::fmin(minValue, value);
        maxValue = std::fmax(maxValue, value);
    }
    std::printf("network output: %zu halfwords | nan %zu | inf %zu | nonzero %zu | "
                "min %.6f | max %.6f | fnv1a=%016llx\n",
                output.size() / 2, nanCount, infCount, nonzero, minValue, maxValue,
                (unsigned long long)Fnv1a(output.data(), output.size()));

    if (dumpFiles) {
        const std::filesystem::path outPath = dumpDir / "output.bin";
        std::ofstream outFile(outPath, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(output.data()), std::streamsize(output.size()));
        outFile.close();
        if (!outFile) {
            std::fprintf(stderr, "error: could not write %ls\n", outPath.c_str());
            return 1;
        }
        std::printf("wrote %ls (%zu B)\n", outPath.c_str(), output.size());
    }

    std::string queueErrors;
    if (!gpu::InfoQueueClean(ctx, queueErrors)) {
        std::fprintf(stderr, "%s", queueErrors.c_str());
        return 1;
    }

    const bool ok = nanCount == 0 && infCount == 0 && nonzero > 0;
    std::printf("reference smoke: %s (finiteness/nonzero only; not determinism or M12 gate)\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace experiments
