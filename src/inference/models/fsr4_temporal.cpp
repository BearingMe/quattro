#include "fsr4_temporal.h"
#include "fsr4_graph.h"
#include "fsr4_weights.h"
#include "gpu/compute.h"

#include <cmath>
#include <cstring>
#include <numeric>

namespace inference::fsr4 {

bool ValidateTemporalFrames(const std::vector<TemporalFrame>& frames, std::string& error) {
    if (frames.empty() || frames.size() > 3 || frames.front().constants.reset != 1) {
        error = "temporal sequence requires 1..3 frames and an initial explicit reset";
        return false;
    }
    for (const auto& frame : frames) {
        const auto& c = frame.constants;
        if (c.width != 1280 || c.height != 720 || c.reset > 1 ||
            (c.flags != 3 && c.flags != 7) || frame.pixels.size() != size_t(1280) * 720 ||
            !std::isfinite(c.exposure) || c.exposure < 0 ||
            !std::isfinite(c.preExposure) || c.preExposure <= 0) {
            error = "unsupported temporal input dimensions/permutation/exposure";
            return false;
        }
        for (uint32_t i = 0; i < 2; ++i) {
            if (!std::isfinite(c.jitter[i]) || std::abs(c.jitter[i]) > 1 ||
                !std::isfinite(c.mvScale[i]) || !std::isfinite(c.jitterCancellation[i])) {
                error = "invalid temporal jitter/motion scale";
                return false;
            }
        }
        for (const auto& pixel : frame.pixels) {
            for (float color : pixel.color) {
                if (!std::isfinite(color) || color < 0 || color > 64000) {
                    error = "temporal color must be finite nonnegative linear RGB <= 64000";
                    return false;
                }
            }
            if (!std::isfinite(pixel.depth) || pixel.depth < 0 || pixel.depth > 1) {
                error = "temporal inverted depth must be finite in [0,1]";
                return false;
            }
            // NaN/Inf motion is intentionally allowed: the reference explicitly
            // invalidates history for it. Do not erase that branch on the CPU.
        }
    }
    return true;
}

bool RunTemporalSequence(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                          const std::vector<TemporalFrame>& frames,
                          const std::vector<uint8_t>& weights, TemporalResult& result,
                          std::string& error) {
    result = {};
    if (!ValidateTemporalFrames(frames, error) || !ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error, true))
        return false;
    constexpr size_t pixels = size_t(1920) * 1080;
    plan.buffers = {{weights.size(), weights.data()}, {pixels * 32, nullptr}};
    size_t previous = 1;
    std::vector<std::vector<uint8_t>> scenes(frames.size());
    // All backing allocations are complete before descriptors store pointers.
    for (size_t i = 0; i < frames.size(); ++i) {
        scenes[i].resize(sizeof(FrameConstants) + frames[i].pixels.size() * sizeof(FramePixel));
        memcpy(scenes[i].data(), &frames[i].constants, sizeof(FrameConstants));
        memcpy(scenes[i].data() + sizeof(FrameConstants), frames[i].pixels.data(),
               frames[i].pixels.size() * sizeof(FramePixel));
    }
    gpu::ComPtr<ID3D12PipelineState> pre, post;
    for (uint32_t stage = 0; stage < 2; ++stage) {
        const char* name = stage == 0 ? "m13/pre" : "m13/post";
        auto& target = stage == 0 ? pre : post;
        std::vector<uint8_t> cso;
        if (!gpu::LoadShaderBytecode(name, cso, error))
            return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = plan.signature.Get();
        desc.CS = {cso.data(), cso.size()};
        if (FAILED(context.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&target)))) {
            error = std::string("cannot create temporal PSO: ") + name;
            return false;
        }
    }
    struct FrameStages { size_t pre, firstNetwork, post; };
    std::vector<FrameStages> stages;
    std::vector<bool> snapshotAfter;
    for (size_t i = 0; i < frames.size(); ++i) {
        const size_t scene = plan.buffers.size();
        plan.buffers.push_back({scenes[i].size(), scenes[i].data()});
        const size_t features = plan.buffers.size();
        plan.buffers.push_back({pixels * 16, nullptr});
        const size_t reprojected = plan.buffers.size();
        plan.buffers.push_back({pixels * 8, nullptr});
        gpu::BufferPass p{};
        p.pso = pre.Get();
        p.gridX = uint32_t((pixels + 63) / 64);
        p.srv0 = scene;
        p.srv1 = previous;
        p.uav0 = features;
        p.uav1 = reprojected;
        p.snapshotBuffer = features;
        const size_t preStage = plan.passes.size();
        plan.passes.push_back(p);
        size_t networkOutput = 0;
        const size_t firstNetwork = plan.passes.size();
        if (!AppendModelGraph(context, extensions, plan, features, 0, 13, networkOutput, error))
            return false;
        const size_t destination = plan.buffers.size();
        plan.buffers.push_back({pixels * 32, nullptr});
        p.pso = post.Get();
        p.srv0 = networkOutput;
        p.srv1 = scene;
        p.uav0 = destination;
        p.snapshotBuffer = destination;
        const size_t postStage = plan.passes.size();
        plan.passes.push_back(p);
        snapshotAfter.resize(plan.passes.size(), false);
        snapshotAfter[preStage] = true;
        snapshotAfter[postStage] = true;
        stages.push_back({preStage, firstNetwork, postStage});
        previous = destination;
    }
    std::vector<std::vector<uint8_t>> captured;
    std::vector<uint8_t> finalOutput;
    std::vector<double> times;
    if (!gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes, previous,
                              snapshotAfter, captured, previous, pixels * 32, finalOutput,
                              error, true, &times))
        return false;
    if (captured.size() != frames.size() * 2 || finalOutput != captured.back()) {
        error = "temporal snapshot/final output count mismatch";
        return false;
    }
    for (size_t i = 0; i < frames.size(); ++i) {
        result.features.push_back(std::move(captured[i * 2]));
        result.frames.push_back(std::move(captured[i * 2 + 1]));
        const auto& s = stages[i];
        result.times.push_back({times[s.pre],
            std::accumulate(times.begin() + s.firstNetwork, times.begin() + s.post, 0.0), times[s.post]});
    }
    return true;
}

} // namespace inference::fsr4
