// gpu — M06 GEMM-tile workload runner: root signature shared by the INT8
// probes, dispatch with GPU timestamp pairs, readback snapshots.
// Scope note: probe-stage code (M06); a generalized kernel runner is justified
// by M09's real shape requirements, not before.

#pragma once

#include "amdext.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpu {

struct MatmulRunResult {
    double minMs = 0.0;
    double avgMs = 0.0;
    double maxMs = 0.0;
    uint32_t iterations = 0;
};

// Root signature shared by the M06/M07 probe shaders:
//   t0 space0        SRV  — byte-addressed input tile data
//   u0 space0        UAV  — byte-addressed output tile
//   u0 space 0x7FFF0ADE  UAV — AMD intrinsics dummy UAV (driver-intercepted)
bool CreateGemmTileRootSignature(DeviceContext& ctx,
                                 ComPtr<ID3D12RootSignature>& rootSignature,
                                 std::string& error);

// Runs the tile workload: `warmup` dispatches, correctness snapshot, then a
// timed loop of `timed` dispatches (timestamp pair per dispatch) and a second
// snapshot. `input`/`inputBytes` are staged verbatim into t0; `outputBytes`
// are read back as int32 words (the fp16 accumulator tile is bit-cast). The
// PSO must have been created against this root signature.
bool RunGemmTile(DeviceContext& ctx, ComPtr<ID3D12RootSignature> rootSignature,
                 ComPtr<ID3D12PipelineState> pso, size_t inputBytes, size_t outputBytes,
                 uint32_t groupCount, uint32_t warmup, uint32_t timed, const void* input,
                 std::vector<int32_t>& outPostWarmup, std::vector<int32_t>& outPostTimed,
                 MatmulRunResult& timing, std::string& error);

} // namespace gpu
