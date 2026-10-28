// gpu — generic byte-addressed multi-pass runner.
//
// `RunDispatchSequence` (sequence.h) models a small fixed pipeline where each
// stage binds one SRV and one UAV against the WMMA root signature. External
// reference shaders (FSR 4 v07 model, M12) instead bind a fixed table:
//   t0 space0 SRV, t1 space0 SRV, u0 space0 UAV, u1 space0 UAV
// with every tensor descriptor baked into the shader and no root constants.
// This runner executes such a pass list over default-heap buffers in one
// command list and can snapshot a buffer between passes. It stays free of
// model semantics: the caller supplies PSOs, grids and bindings.

#pragma once

#include "matmul.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpu {

// Root signature for the table above (two SRVs, two UAVs, no constants).
// nativeWmma adds the reserved AMD intrinsic dummy UAV at root parameter 4.
bool CreateBufferPassRootSignature(DeviceContext& ctx,
                                   ComPtr<ID3D12RootSignature>& rootSignature,
                                   std::string& error, bool nativeWmma = false);

// Recognize only the observed driver-magic access against THIS signature's
// named 16-byte dummy UAV at root4. Never exempts tensor/weight accesses.
bool IsBufferPassWmmaDiagnostic(const D3D12_MESSAGE& message);

struct BufferCopy {
    size_t source = 0;
    size_t destination = 0;
    size_t sourceOffset = 0;
    size_t destinationOffset = 0;
    size_t bytes = 0;
};

struct BufferPass {
    ID3D12PipelineState* pso = nullptr;
    uint32_t gridX = 1;
    uint32_t gridY = 1;
    uint32_t gridZ = 1;
    size_t srv0 = 0; // t0 (bound but may be unused by the shader)
    size_t srv1 = 0; // t1
    size_t uav0 = 0; // u0
    size_t uav1 = 0; // u1
    // GPU-only copies before this dispatch (e.g. assembling a packed input).
    // Same-resource copies are deliberately unsupported. Bounds checked first.
    std::vector<BufferCopy> copiesBefore;
    // SIZE_MAX uses RunBufferPasses' default; otherwise snapshot this resource
    // when snapshotAfter is set. No readback ever becomes a dispatch input.
    size_t snapshotBuffer = SIZE_MAX;
};

struct BufferInit {
    size_t bytes = 0;
    const void* data = nullptr; // null -> zero-filled
};

// Runs `passes` in order. After pass `i`, if `snapshotAfter[i]` is true, the
// whole of `snapshotBuffer` (or the pass override) is copied to a readback and appended to
// `snapshots`. Finally the first `outputBytes` of `outputBuffer` are read back.
// SRVs may alias each other, but no SRV may alias a UAV and the UAVs must be
// distinct. Unsupported aliasing is rejected before recording any dispatch.
// nativeWmma must match the signature supplied by the caller.
bool RunBufferPasses(DeviceContext& ctx, ComPtr<ID3D12RootSignature> rootSignature,
                     const std::vector<BufferInit>& buffers,
                     const std::vector<BufferPass>& passes,
                     size_t snapshotBuffer, const std::vector<bool>& snapshotAfter,
                     std::vector<std::vector<uint8_t>>& snapshots,
                     size_t outputBuffer, size_t outputBytes,
                      std::vector<uint8_t>& output, std::string& error,
                      bool nativeWmma = false,
                      std::vector<double>* gpuMilliseconds = nullptr);

} // namespace gpu
