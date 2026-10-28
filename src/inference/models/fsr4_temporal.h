#pragma once

#include "gpu/amdext.h"
#include <cstdint>
#include <string>
#include <vector>

namespace inference::fsr4 {

// M13's explicit supported permutation: quality 1280x720 -> 1920x1080,
// linear color, manual exposure, low-resolution motion, inverted depth, no RCAS.
// Raw MV units are LR pixels; scale converts them to normalized UV offsets.
struct FrameConstants {
    uint32_t reset = 0, frame = 0, width = 1280, height = 720;
    float jitter[2] = {}, exposure = 1, preExposure = 1;
    float mvScale[2] = {1.0f / 1280, 1.0f / 720};
    float jitterCancellation[2] = {};
    uint32_t flags = 3; // bit0 LR motion, bit1 inverted depth, bit2 jittered MV
    uint32_t reserved[3] = {};
};
static_assert(sizeof(FrameConstants) == 64);

struct FramePixel {
    float color[3];
    float depth;
    float motion[2];
};
static_assert(sizeof(FramePixel) == 24);

struct TemporalFrame {
    FrameConstants constants;
    std::vector<FramePixel> pixels;
};

struct TemporalResult {
    // Per output pixel: half4 model-space history, half4 activated recurrence,
    // float4 linear-color output. 32 bytes, 1920x1080 pixels per frame.
    std::vector<std::vector<uint8_t>> frames;
    // Seven meaningful FP16 features plus one zero pad, 16 bytes per pixel.
    std::vector<std::vector<uint8_t>> features;
    struct StageTimes { double preMs, inferenceMs, postMs; };
    std::vector<StageTimes> times;
};

bool ValidateTemporalFrames(const std::vector<TemporalFrame>& frames, std::string& error);
bool RunTemporalSequence(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                          const std::vector<TemporalFrame>& frames,
                          const std::vector<uint8_t>& weights, TemporalResult& result,
                          std::string& error);

} // namespace inference::fsr4
