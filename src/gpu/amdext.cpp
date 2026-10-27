#include "amdext.h"

#include <cstdio>

namespace gpu {

bool LoadAmdExtensions(DeviceContext& ctx, AmdExtensions& out, std::string& error) {
    // amdxc64.dll ships with the Adrenalin driver in System32.
    out.library = LoadLibraryW(L"amdxc64.dll");
    if (!out.library) {
        error = "amdxc64.dll not loadable (AMD driver extensions missing)";
        return false;
    }
    const auto createInterface =
        reinterpret_cast<PFNAmdExtD3DCreateInterface>(
            reinterpret_cast<void*>(GetProcAddress(out.library, "AmdExtD3DCreateInterface")));
    if (!createInterface) {
        error = "amdxc64.dll does not export AmdExtD3DCreateInterface";
        ReleaseAmdExtensions(out);
        return false;
    }

    // The factory is created on the device (pOuter = device); passing null
    // returns E_POINTER.
    HRESULT hr = createInterface(ctx.device.Get(), IID_PPV_ARGS(&out.factory));
    if (FAILED(hr)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "AmdExtD3DCreateInterface(IAmdExtD3DFactory) failed, HRESULT=0x%08lX",
                      (unsigned long)hr);
        error = buf;
        ReleaseAmdExtensions(out);
        return false;
    }

    hr = out.factory->CreateInterface(ctx.device.Get(), IID_PPV_ARGS(&out.intrinsics));
    if (SUCCEEDED(hr)) {
        // Report support state, then enable. Failure here is not fatal for the
        // analyzer path — the caller decides.
        if (FAILED(out.intrinsics->GetInfo(&out.intrinsicsInfo)))
            error = "IAmdExtD3DShaderIntrinsics::GetInfo failed";
        out.waveMatrixSupported =
            SUCCEEDED(out.intrinsics->CheckSupport(AmdExtD3DShaderIntrinsicsSupport_WaveMatrix));
        if (out.waveMatrixSupported &&
            FAILED(out.intrinsics->Enable()))
            error = "IAmdExtD3DShaderIntrinsics::Enable failed";
    } else {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "IAmdExtD3DShaderIntrinsics unavailable, HRESULT=0x%08lX",
                      (unsigned long)hr);
        error = buf;
        out.intrinsics = nullptr;
    }

    hr = out.factory->CreateInterface(ctx.device.Get(), IID_PPV_ARGS(&out.analyzer));
    if (FAILED(hr)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "IAmdExtD3DShaderAnalyzer unavailable, HRESULT=0x%08lX",
                      (unsigned long)hr);
        error = buf;
        out.analyzer = nullptr;
    } else {
        out.analyzer->QueryInterface(IID_PPV_ARGS(&out.analyzer1));
    }
    return true;
}

bool CreateAnalyzedComputePipeline(AmdExtensions& ext, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
                                   ComPtr<ID3D12PipelineState>& pso,
                                   AmdExtD3DPipelineHandle& handle,
                                   AmdExtD3DComputeShaderStats& stats, std::string& error) {
    if (!ext.analyzer) {
        error = "shader analyzer extension unavailable";
        return false;
    }
    // The returned PSO is a real ID3D12PipelineState usable for dispatch; the
    // handle is the driver-side pipeline for ISA/stats queries.
    const HRESULT hr = ext.analyzer->CreateComputePipelineState(
        &desc, IID_PPV_ARGS(&pso), &stats, &handle);
    if (FAILED(hr)) {
        error = "IAmdExtD3DShaderAnalyzer::CreateComputePipelineState failed";
        return false;
    }
    return true;
}

bool GetPipelineIsa(AmdExtensions& ext, AmdExtD3DPipelineHandle handle,
                    std::string& isa, std::string& statsLine, std::string& error) {
    if (!ext.analyzer) {
        error = "shader analyzer extension unavailable";
        return false;
    }
    if (!handle) {
        error = "no analyzer pipeline handle";
        return false;
    }

    // First call: query size. Second call: fill the buffer. The stage
    // disassembly pointers are only valid until the next call — copy out.
    size_t isaSize = 0;
    AmdExtD3DPipelineDisassembly sizeQueryDisasm{};
    HRESULT hr = ext.analyzer->GetShaderIsaCode(handle, nullptr, &isaSize, &sizeQueryDisasm);
    if (FAILED(hr) || isaSize == 0) {
        error = "GetShaderIsaCode size query failed";
        return false;
    }
    std::vector<char> buffer(isaSize + 1, '\0');
    AmdExtD3DPipelineDisassembly disasm{};
    hr = ext.analyzer->GetShaderIsaCode(handle, buffer.data(), &isaSize, &disasm);
    if (FAILED(hr)) {
        error = "GetShaderIsaCode failed";
        return false;
    }
    isa.assign(buffer.data(), isaSize);
    if (disasm.pComputeDisassembly)
        isa.assign(disasm.pComputeDisassembly); // prefer the stage text if present

    AmdExtD3DComputeShaderStats stats{};
    const bool haveStats = ext.analyzer1 &&
                           SUCCEEDED(ext.analyzer1->GetComputeShaderStats(handle, &stats));
    if (haveStats) {
        char line[256];
        std::snprintf(line, sizeof(line),
                      "vgpr %u/%u, sgpr %u/%u, lds %zu B/group",
                      stats.usageStats.numUsedVgprs, stats.numPhysicalVgprs,
                      stats.usageStats.numUsedSgprs, stats.numPhysicalSgprs,
                      stats.usageStats.ldsUsageSizeInBytes);
        statsLine = line;
    } else {
        statsLine = "stats unavailable";
    }
    return true;
}

void ReleaseAmdExtensions(AmdExtensions& ext) {
    ext.analyzer.Reset();
    ext.intrinsics.Reset();
    ext.factory.Reset();
    if (ext.library) {
        FreeLibrary(ext.library);
        ext.library = nullptr;
    }
}

} // namespace gpu
