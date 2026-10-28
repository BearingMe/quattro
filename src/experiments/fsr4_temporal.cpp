#include "experiments.h"
#include "core/numerics.h"
#include "inference/models/fsr4_temporal.h"
#include "inference/models/fsr4_weights.h"
#include "gpu/graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace experiments {
namespace {

std::filesystem::path RepositoryRoot() {
    wchar_t path[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, path, MAX_PATH) == 0)
        return {};
    return std::filesystem::path(path).parent_path().parent_path().parent_path();
}

bool WritePpm(const std::filesystem::path& path, const std::vector<uint8_t>& result) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n1920 1080\n255\n";
    for (size_t pixel = 0; pixel < size_t(1920) * 1080; ++pixel) {
        float color[3]{};
        memcpy(color, result.data() + pixel * 32 + 16, sizeof(color));
        for (float value : color) {
            // Display-only Reinhard + gamma; raw linear results are also retained.
            const float mapped = std::pow(std::max(value, 0.0f) / (1 + std::max(value, 0.0f)), 1.0f / 2.2f);
            file.put(char(uint8_t(std::clamp(int(std::round(mapped * 255)), 0, 255))));
        }
    }
    return bool(file);
}

uint16_t HalfAt(const std::vector<uint8_t>& data, size_t offset) {
    uint16_t bits = 0;
    memcpy(&bits, data.data() + offset, 2);
    return bits;
}

} // namespace

int RunFsr4Temporal() {
    using namespace inference::fsr4;
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string error;
    const char* sequenceEnv = std::getenv("M13_SEQUENCE");
    const std::string sequence = sequenceEnv ? sequenceEnv : "moving-reset";
    if (sequence != "moving-reset" && sequence != "static") {
        std::fprintf(stderr, "M13_SEQUENCE must be moving-reset or static\n");
        return 2;
    }
    const bool staticSequence = sequence == "static";
    std::printf("sequence: %s\n", sequence.c_str());
    const char* requestedId = std::getenv("M13_RUN_ID");
    const std::string runId = requestedId ? requestedId :
        "run-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64());
    if (runId.empty() || runId.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos) {
        std::fprintf(stderr, "M13_RUN_ID must contain only letters, digits, '-' or '_'\n");
        return 2;
    }
    const auto directory = RepositoryRoot() / "build/reports/m13" / runId;
    std::error_code ec;
    std::filesystem::create_directories(directory.parent_path(), ec);
    if (ec || !std::filesystem::create_directory(directory, ec) || ec) {
        std::fprintf(stderr, "temporal capture directory exists or cannot be created\n");
        return 1;
    }
    std::printf("temporal captures: %ls\n", directory.c_str());
    std::vector<TemporalFrame> frames(3);
    for (uint32_t i = 0; i < 3; ++i) {
        auto& frame = frames[i];
        auto& c = frame.constants;
        c.frame = i;
        c.reset = i == 0;
        c.jitter[0] = i % 2 == 0 ? 0.25f : -0.25f;
        c.jitter[1] = -c.jitter[0];
        frame.pixels.resize(size_t(1280) * 720);
        for (uint32_t y = 0; y < 720; ++y) {
            for (uint32_t x = 0; x < 1280; ++x) {
                auto& p = frame.pixels[size_t(y) * 1280 + x];
                const float px = float(x) - c.jitter[0] - (staticSequence ? 0.0f : float(i));
                const float py = float(y) - c.jitter[1];
                const bool foreground = px >= 480 && px < 800 && py >= 240 && py < 480;
                const bool checker = (int(std::floor(px / 16)) + int(std::floor(py / 16))) & 1;
                p.color[0] = foreground ? 2.0f : checker ? 0.25f : 0.05f;
                p.color[1] = foreground ? 0.3f : 0.1f + 0.4f * float(y) / 720;
                p.color[2] = foreground ? 0.05f : 0.1f + 0.4f * float(x) / 1280;
                p.depth = foreground ? 0.8f : 0.2f;
                p.motion[0] = staticSequence || i == 0 ? 0.0f : -1.0f;
                p.motion[1] = 0;
                if (!staticSequence && i == 1 && x >= 624 && x < 656 && y >= 344 && y < 376)
                    p.motion[0] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    // A camera cut back to the initial scene with a dirty previous history.
    // Identical input/settings must reproduce frame0 exactly despite frame1.
    if (!staticSequence) {
        frames[2] = frames[0];
        frames[2].constants.frame = 2;
    }
    bool rejected = true;
    std::vector<TemporalFrame> empty;
    rejected &= !ValidateTemporalFrames(empty, error);
    frames[0].constants.reset = 0;
    rejected &= !ValidateTemporalFrames(frames, error);
    frames[0].constants.reset = 1;
    frames[0].constants.width = 1920;
    rejected &= !ValidateTemporalFrames(frames, error);
    frames[0].constants.width = 1280;
    frames[0].constants.flags = 0;
    rejected &= !ValidateTemporalFrames(frames, error);
    frames[0].constants.flags = 3;
    frames[0].constants.preExposure = 0;
    rejected &= !ValidateTemporalFrames(frames, error);
    frames[0].constants.preExposure = 1;
    if (!rejected || !ValidateTemporalFrames(frames, error)) {
        std::fprintf(stderr, "temporal contract rejection failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("temporal contract: invalid empty/reset/dimensions/permutation/exposure rejected; NaN motion accepted for GPU invalidation\n");
    std::vector<uint8_t> weights;
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
        std::fprintf(stderr, "WMMA unavailable: %s\n", error.c_str());
        gpu::ReleaseAmdExtensions(ext);
        return 1;
    }
    struct ExtensionLifetime {
        gpu::AmdExtensions& ext;
        ~ExtensionLifetime() { gpu::ReleaseAmdExtensions(ext); }
    } lifetime{ext};
    std::printf("adapter: %ls | 1280x720 -> 1920x1080 | linear/manual exposure/LR MV/inverted depth/no RCAS\n",
                ctx.adapter.name.c_str());
    TemporalResult result;
    if (!RunTemporalSequence(ctx, ext, frames, weights, result, error)) {
        std::fprintf(stderr, "temporal sequence: %s\n", error.c_str());
        return 1;
    }
    bool ok = result.frames.size() == 3 && result.features.size() == 3 && result.times.size() == 3;
    if (!ok)
        return 1;
    constexpr size_t pixels = size_t(1920) * 1080;
    for (size_t i = 0; i < 3; ++i) {
        if (result.frames[i].size() != pixels * 32 || result.features[i].size() != pixels * 16)
            return 1;
        size_t nonfinite = 0, nonzero = 0, invalidRecurrent = 0, resetFeatureErrors = 0;
        size_t persistedRecurrence = 0;
        float minimum = INFINITY, maximum = -INFINITY;
        for (size_t p = 0; p < pixels; ++p) {
            float color[3]{};
            memcpy(color, result.frames[i].data() + p * 32 + 16, sizeof(color));
            for (float value : color) {
                nonfinite += !std::isfinite(value);
                nonzero += value != 0;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            for (uint32_t c = 0; c < 4; ++c) {
                const float recurrence = core::F16ToF32(HalfAt(result.frames[i], p * 32 + 8 + c * 2));
                invalidRecurrent += !std::isfinite(recurrence) || recurrence < 0 || recurrence > 1;
                if (frames[i].constants.reset)
                    resetFeatureErrors += HalfAt(result.features[i], p * 16 + 6 + c * 2) != 0xBC00;
                else
                    persistedRecurrence += HalfAt(result.features[i], p * 16 + 6 + c * 2) != 0xBC00;
            }
            if (frames[i].constants.reset)
                resetFeatureErrors += HalfAt(result.features[i], p * 16) != HalfAt(result.features[i], p * 16 + 2);
            resetFeatureErrors += HalfAt(result.features[i], p * 16 + 14) != 0;
        }
        const auto& t = result.times[i];
        std::printf("frame%zu: finite errors %zu | nonzero RGB components %zu | range [%g,%g] | invalid recurrence %zu | reset/pad errors %zu\n",
                    i, nonfinite, nonzero, minimum, maximum, invalidRecurrent, resetFeatureErrors);
        std::printf("frame%zu GPU timestamps (validation enabled): pre %.6f ms | inference %.6f ms | post %.6f ms\n",
                    i, t.preMs, t.inferenceMs, t.postMs);
        ok = ok && nonfinite == 0 && nonzero > 0 && minimum >= 0 && maximum <= 64000 &&
             invalidRecurrent == 0 && resetFeatureErrors == 0 &&
             std::isfinite(t.preMs) && t.preMs > 0 && std::isfinite(t.inferenceMs) &&
             t.inferenceMs > 0 && std::isfinite(t.postMs) && t.postMs > 0;
        if (!frames[i].constants.reset) {
            std::printf("frame%zu persisted recurrent feature components: %zu\n", i, persistedRecurrence);
            ok = ok && persistedRecurrence > 0;
        }
        if (!WritePpm(directory / ("frame" + std::to_string(i) + ".ppm"), result.frames[i]))
            return 1;
        std::ofstream raw(directory / ("frame" + std::to_string(i) + ".bin"), std::ios::binary);
        raw.write(reinterpret_cast<const char*>(result.frames[i].data()), std::streamsize(result.frames[i].size()));
        if (!raw)
            return 1;
        std::ofstream input(directory / ("input-frame" + std::to_string(i) + ".bin"), std::ios::binary);
        input.write(reinterpret_cast<const char*>(&frames[i].constants), sizeof(FrameConstants));
        input.write(reinterpret_cast<const char*>(frames[i].pixels.data()),
                    std::streamsize(frames[i].pixels.size() * sizeof(FramePixel)));
        if (!input)
            return 1;
    }
    const size_t center = size_t(540) * 1920 + 960;
    bool invalidMotionReset = HalfAt(result.features[1], center * 16 + 2) == 0xBC00;
    for (uint32_t c = 0; c < 4; ++c)
        invalidMotionReset &= HalfAt(result.features[1], center * 16 + 6 + c * 2) == 0xBC00;
    const bool resetIdentical = result.frames[0] == result.frames[2] && result.features[0] == result.features[2];
    const bool historyUsed = result.features[1] != result.features[0];
    if (staticSequence)
        std::printf("static scene: continued features changed %s | camera-cut/NaN-motion cases belong to moving-reset test\n",
                    historyUsed ? "yes" : "NO");
    else
        std::printf("camera-cut reset equals clean initial frame: %s | continued features changed: %s | NaN-motion local invalidation: %s\n",
                    resetIdentical ? "yes" : "NO", historyUsed ? "yes" : "NO", invalidMotionReset ? "yes" : "NO");
    ok = ok && historyUsed && (staticSequence || (resetIdentical && invalidMotionReset));
    std::string queueErrors;
    size_t intrinsicMessages = 0;
    if (!gpu::InfoQueueCleanWithFilter(ctx, gpu::IsBufferPassWmmaDiagnostic, intrinsicMessages, queueErrors)) {
        std::fprintf(stderr, "%s\n", queueErrors.c_str());
        return 1;
    }
    std::printf("experimental temporal sequence: %s | exact intrinsic messages %zu | official pre/post sampling equivalence UNVERIFIED\n",
                ok ? "PASS" : "FAIL", intrinsicMessages);
    return ok ? 0 : 1;
}

} // namespace experiments
