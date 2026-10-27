// gpu — AMD D3D12 driver extension runtime (amdxc64.dll): shader intrinsics
// (WMMA) and shader analyzer (ISA disassembly). M06.
// Dependency rule: gpu never learns FSR 4 semantics; this module exposes raw
// driver capability, nothing model-specific.

#pragma once

#include "device.h"

#include <string>
#include <vector>

#include "amd/AmdExtD3D/AmdExtD3D.h"
#include "amd/AmdExtD3D/AmdExtD3DShaderAnalyzerApi.h"
#include "amd/AmdExtD3D/AmdExtD3DShaderIntrinsicsApi.h"

namespace gpu {

struct AmdExtensions {
    HMODULE library = nullptr;
    ComPtr<IAmdExtD3DFactory> factory;
    ComPtr<IAmdExtD3DShaderIntrinsics> intrinsics;
    ComPtr<IAmdExtD3DShaderAnalyzer> analyzer;
    ComPtr<IAmdExtD3DShaderAnalyzer1> analyzer1; // optional v1: per-stage stats
    AmdExtD3DShaderIntrinsicsInfo intrinsicsInfo{};
    bool waveMatrixSupported = false;
};

// Loads amdxc64.dll, creates the extension factory on the device, checks and
// enables the WaveMatrix shader intrinsics, and queries the shader analyzer.
// Intrinsics/analyzer interfaces may be null on failure — the callers decide
// whether that is fatal for their experiment; `error` explains why.
bool LoadAmdExtensions(DeviceContext& ctx, AmdExtensions& out, std::string& error);

// Creates a compute pipeline through the driver's shader analyzer extension —
// the same compiler the live dispatch path uses — returning the usable PSO
// plus the pipeline handle for ISA queries and register/LDS usage stats.
bool CreateAnalyzedComputePipeline(AmdExtensions& ext, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
                                   ComPtr<ID3D12PipelineState>& pso,
                                   AmdExtD3DPipelineHandle& handle,
                                   AmdExtD3DComputeShaderStats& stats, std::string& error);

// Returns the ISA disassembly text for a pipeline created by
// CreateAnalyzedComputePipeline. Writes a single-line summary into `statsLine`.
bool GetPipelineIsa(AmdExtensions& ext, AmdExtD3DPipelineHandle handle,
                    std::string& isa, std::string& statsLine, std::string& error);

void ReleaseAmdExtensions(AmdExtensions& ext);

} // namespace gpu
