// gpu — dispatch-sequence runner: several PSOs sharing default-heap buffers,
// executed back-to-back in one command list. M11's real requirement (GPU-
// resident multi-layer inference, no CPU roundtrip between stages) justifies
// this generalization of RunGemmTile; the caller owns stage/graph semantics.
// Dependency rule: gpu never learns network semantics — stages/buffers are
// opaque descriptors supplied by the caller.

#pragma once

#include "matmul.h"

#include <string>
#include <vector>

namespace gpu {

struct DispatchStage {
    ID3D12PipelineState* pso = nullptr;
    uint32_t gridX = 1;
    uint32_t gridY = 1;
    size_t srvBuffer = 0; // bound to t0 space0 for this stage
    size_t uavBuffer = 0; // bound to u0 space0 for this stage
};

struct SequenceBufferDesc {
    size_t bytes = 0;
    const void* initialData = nullptr; // copied verbatim; null -> zero-filled
};

struct SequenceSnapshotDesc {
    size_t buffer = 0;
    size_t offset = 0;
    size_t bytes = 0;
};

// Executes `stages` in order, `warmupNetworks` full sequences, then takes the
// requested snapshots plus one full `outputBuffer` readback; then runs
// `timedNetworks` sequences with a GPU timestamp pair around each sequence
// (dispatches + inter-stage barriers only — upload/readback are outside) and
// takes a second `outputBuffer` readback. The caller verifies
// outputWarmup == outputTimed (per-sequence idempotency) and compares
// snapshots against its own references.
bool RunDispatchSequence(
    DeviceContext& ctx, ComPtr<ID3D12RootSignature> rootSignature,
    const std::vector<SequenceBufferDesc>& buffers,
    const std::vector<DispatchStage>& stages,
    const std::vector<SequenceSnapshotDesc>& snapshots,
    size_t outputBuffer, size_t outputBytes,
    uint32_t warmupNetworks, uint32_t timedNetworks,
    std::vector<std::vector<uint8_t>>& snapshotData,
    std::vector<uint8_t>& outputWarmup, std::vector<uint8_t>& outputTimed,
    MatmulRunResult& timing, std::string& error);

} // namespace gpu
