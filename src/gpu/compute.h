// gpu — minimal compute harness: shader loading, buffers, dispatch, sync,
// GPU timestamps. M05 scope (docs/MILESTONES.md); no neural-network code.
// Dependency rule: gpu never compares against a CPU reference — the caller
// (experiments) owns the reference comparison via `reference`.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "device.h"

namespace gpu {

struct ComputeRunResult {
    double minMs = 0.0;
    double avgMs = 0.0;
    double maxMs = 0.0;
    uint32_t iterations = 0;
};

struct VectorAddOutputs {
    std::vector<float> postWarmup; // snapshot after warmup dispatches
    std::vector<float> postTimed;  // snapshot after the timed loop
};

// Loads a compiled shader object (.cso) relative to the running executable
// (shaders/<name>.cso next to the binary). Returns false with `error`.
bool LoadShaderBytecode(const char* shaderName, std::vector<uint8_t>& out, std::string& error);

// Runs the vector-add workload: upload -> warmup dispatches -> post-warmup
// snapshot -> timed dispatch loop (GPU timestamp pair per dispatch) -> final
// snapshot. `a`/`b` are the inputs; both snapshots are returned for the
// caller to compare against an independent CPU reference.
bool RunVectorAdd(DeviceContext& ctx, size_t elementCount, uint32_t warmup,
                  uint32_t timed, const float* a, const float* b,
                  VectorAddOutputs& outputs, ComputeRunResult& timing, std::string& error);

} // namespace gpu
