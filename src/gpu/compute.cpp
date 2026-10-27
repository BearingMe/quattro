#include "compute.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#pragma comment(lib, "d3d12.lib")

namespace gpu {
namespace {

using Microsoft::WRL::ComPtr;

// Minimal replacements for the d3dx12.h helpers (that header is not part of
// the Windows SDK; the D3D12 structs are trivial to fill directly).

D3D12_HEAP_PROPERTIES HeapProps(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = type;
    return props;
}

D3D12_RESOURCE_DESC BufferDesc(uint64_t bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_BARRIER TransitionBarrier(ID3D12Resource* resource,
                                         D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

std::string HrMsg(const char* what, HRESULT hr) {
    std::ostringstream out;
    out << what << " failed, HRESULT=0x" << std::hex << (uint32_t)hr;
    return out.str();
}

uint32_t DispatchGroups(size_t elementCount) {
    return uint32_t((elementCount + 255) / 256);
}

} // namespace

bool LoadShaderBytecode(const char* shaderName, std::vector<uint8_t>& out, std::string& error) {
    wchar_t exePathW[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exePathW, MAX_PATH) == 0) {
        error = "GetModuleFileNameW failed";
        return false;
    }
    const std::filesystem::path exeDir = std::filesystem::path(exePathW).parent_path();
    const std::filesystem::path cso = exeDir / "shaders" / (std::string(shaderName) + ".cso");

    std::ifstream file(cso, std::ios::binary);
    if (!file) {
        error = "cannot open shader bytecode: " + cso.string();
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (out.empty()) {
        error = "shader bytecode empty: " + cso.string();
        return false;
    }
    return true;
}

bool RunVectorAdd(DeviceContext& ctx, size_t elementCount, uint32_t warmup,
                  uint32_t timed, const float* a, const float* b,
                  VectorAddOutputs& outputs, ComputeRunResult& result, std::string& error) {
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             ctx.allocator.Get(), nullptr,
                                             IID_PPV_ARGS(&list)))) {
        error = "CreateCommandList failed";
        return false;
    }

    // --- Root signature: two root SRVs, one root UAV, one 32-bit constant. ---
    D3D12_ROOT_PARAMETER params[4] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 1;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[3].Constants.ShaderRegister = 0;
    params[3].Constants.Num32BitValues = 1;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 4;
    rsDesc.pParameters = params;
    ComPtr<ID3DBlob> rsBlob;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                           &rsBlob, nullptr))) {
        error = "D3D12SerializeRootSignature failed";
        return false;
    }
    ComPtr<ID3D12RootSignature> rootSignature;
    if (FAILED(ctx.device->CreateRootSignature(0, rsBlob->GetBufferPointer(),
                                               rsBlob->GetBufferSize(),
                                               IID_PPV_ARGS(&rootSignature)))) {
        error = "CreateRootSignature failed";
        return false;
    }

    // --- PSO from the offline-compiled CSO. ---
    std::vector<uint8_t> cso;
    if (!LoadShaderBytecode("vector_add", cso, error))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSignature.Get();
    psoDesc.CS = {cso.data(), cso.size()};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(ctx.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)))) {
        error = "CreateComputePipelineState failed";
        return false;
    }

    // --- Resources. ---
    // Sanity: the fence mechanism itself must work before any GPU work.
    if (!GpuSync(ctx, 1000, error)) {
        error = "fence sanity check before any GPU work failed: " + error;
        return false;
    }
    const uint64_t bytes = uint64_t(elementCount) * sizeof(float);
    const D3D12_HEAP_PROPERTIES defaultHeap = HeapProps(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_HEAP_PROPERTIES uploadHeap = HeapProps(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_HEAP_PROPERTIES readbackHeap = HeapProps(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC bufferDesc = BufferDesc(bytes, D3D12_RESOURCE_FLAG_NONE);

    ComPtr<ID3D12Resource> bufA, bufB, bufOut, uploadA, uploadB, readback1, readback2, tsReadback;
    HRESULT hr = S_OK;
    // Buffers must be created in COMMON; explicit barriers move them into place.
    hr = ctx.device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
                                             &bufferDesc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&bufA));
    if (FAILED(hr)) {
        error = HrMsg("CreateCommittedResource (gpu input a)", hr);
        return false;
    }
    hr = ctx.device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
                                             &bufferDesc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&bufB));
    if (FAILED(hr)) {
        error = HrMsg("CreateCommittedResource (gpu input b)", hr);
        return false;
    }
    const D3D12_RESOURCE_DESC outDesc = BufferDesc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    hr = ctx.device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
                                             &outDesc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&bufOut));
    if (FAILED(hr)) {
        error = HrMsg("CreateCommittedResource (gpu output)", hr);
        return false;
    }
    if (FAILED(ctx.device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
                                                   &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr, IID_PPV_ARGS(&uploadA))) ||
        FAILED(ctx.device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
                                                   &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr, IID_PPV_ARGS(&uploadB)))) {
        error = "CreateCommittedResource (uploads) failed";
        return false;
    }
    if (FAILED(ctx.device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
                                                   &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&readback1))) ||
        FAILED(ctx.device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
                                                   &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&readback2)))) {
        error = "CreateCommittedResource (readbacks) failed";
        return false;
    }
    const D3D12_RESOURCE_DESC tsDesc = BufferDesc(uint64_t(timed) * 2 * sizeof(uint64_t),
                                                  D3D12_RESOURCE_FLAG_NONE);
    if (FAILED(ctx.device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
                                                   &tsDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&tsReadback)))) {
        error = "CreateCommittedResource (timestamp readback) failed";
        return false;
    }

    // Stage input data into upload heaps (persistent maps are legal for UPLOAD).
    float* uploadAData = nullptr;
    float* uploadBData = nullptr;
    if (FAILED(uploadA->Map(0, nullptr, reinterpret_cast<void**>(&uploadAData))) ||
        FAILED(uploadB->Map(0, nullptr, reinterpret_cast<void**>(&uploadBData)))) {
        error = "upload Map failed";
        return false;
    }
    std::copy_n(a, elementCount, uploadAData);
    std::copy_n(b, elementCount, uploadBData);
    uploadA->Unmap(0, nullptr);
    uploadB->Unmap(0, nullptr);

    // --- Timestamp query heap (timestamp queries use EndQuery only). ---
    ComPtr<ID3D12QueryHeap> queryHeap;
    D3D12_QUERY_HEAP_DESC queryDesc{};
    queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    queryDesc.Count = uint32_t(timed) * 2;
    if (FAILED(ctx.device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queryHeap)))) {
        error = "CreateQueryHeap failed";
        return false;
    }

    // --- Record: upload copy -> warmup -> snapshot -> timed loop -> readback. ---
    D3D12_RESOURCE_BARRIER barrierAToCopy = TransitionBarrier(
        bufA.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_RESOURCE_BARRIER barrierBToCopy = TransitionBarrier(
        bufB.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &barrierAToCopy);
    list->ResourceBarrier(1, &barrierBToCopy);
    list->CopyBufferRegion(bufA.Get(), 0, uploadA.Get(), 0, bytes);
    list->CopyBufferRegion(bufB.Get(), 0, uploadB.Get(), 0, bytes);

    D3D12_RESOURCE_BARRIER barrierA = TransitionBarrier(
        bufA.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    D3D12_RESOURCE_BARRIER barrierB = TransitionBarrier(
        bufB.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    D3D12_RESOURCE_BARRIER barrierOutCommonUav = TransitionBarrier(
        bufOut.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &barrierA);
    list->ResourceBarrier(1, &barrierB);
    list->ResourceBarrier(1, &barrierOutCommonUav);

    list->SetComputeRootSignature(rootSignature.Get());
    list->SetPipelineState(pso.Get());
    list->SetComputeRootShaderResourceView(0, bufA->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1, bufB->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(2, bufOut->GetGPUVirtualAddress());
    list->SetComputeRoot32BitConstant(3, uint32_t(elementCount), 0);

    const uint32_t groups = DispatchGroups(elementCount);
    for (uint32_t i = 0; i < warmup; ++i)
        list->Dispatch(groups, 1, 1);

    // Validation snapshot after warmup: copy the result out mid-list.
    D3D12_RESOURCE_BARRIER barrierOutToSrc1 = TransitionBarrier(
        bufOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrierOutToSrc1);
    list->CopyBufferRegion(readback1.Get(), 0, bufOut.Get(), 0, bytes);
    D3D12_RESOURCE_BARRIER barrierOutToUav = TransitionBarrier(
        bufOut.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &barrierOutToUav);

    // Timed loop: a timestamp stamp before and after each dispatch.
    for (uint32_t i = 0; i < timed; ++i) {
        list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2);
        list->Dispatch(groups, 1, 1);
        list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2 + 1);
    }

    // Final readback (validates the timed loop's results too).
    D3D12_RESOURCE_BARRIER barrierOutToSrc2 = TransitionBarrier(
        bufOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrierOutToSrc2);
    list->CopyBufferRegion(readback2.Get(), 0, bufOut.Get(), 0, bytes);

    // Timestamp resolve straight into the readback heap (created COPY_DEST).
    list->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                           uint32_t(timed) * 2, tsReadback.Get(), 0);

    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    ctx.queue->ExecuteCommandLists(1, lists);
    if (!GpuSync(ctx, 10000, error)) {
        // Surface why the GPU never finished: removal reason + info queue.
        const HRESULT removed = ctx.device->GetDeviceRemovedReason();
        std::string queueErrors;
        gpu::InfoQueueClean(ctx, queueErrors);
        error = "gpu sync after dispatch failed: " + error + "; " +
                HrMsg("GetDeviceRemovedReason", removed) +
                (queueErrors.empty() ? std::string() : "; " + queueErrors);
        return false;
    }

    // --- Read back both snapshots. ---
    void* raw1 = nullptr;
    void* raw2 = nullptr;
    if (FAILED(readback1->Map(0, nullptr, &raw1)) ||
        FAILED(readback2->Map(0, nullptr, &raw2))) {
        error = "readback Map failed";
        return false;
    }
    const float* result1 = static_cast<const float*>(raw1);
    const float* result2 = static_cast<const float*>(raw2);
    outputs.postWarmup.assign(result1, result1 + elementCount);
    outputs.postTimed.assign(result2, result2 + elementCount);
    readback1->Unmap(0, nullptr);
    readback2->Unmap(0, nullptr);

    // --- Timing. ---
    uint64_t* stamps = nullptr;
    if (FAILED(tsReadback->Map(0, nullptr, reinterpret_cast<void**>(&stamps)))) {
        error = "timestamp Map failed";
        return false;
    }
    result.iterations = timed;
    result.minMs = 1e30;
    result.maxMs = 0.0;
    double total = 0.0;
    for (uint32_t i = 0; i < timed; ++i) {
        const uint64_t start = stamps[i * 2];
        const uint64_t end = stamps[i * 2 + 1];
        if (end < start) {
            error = "timestamp pair went backwards (query issue)";
            tsReadback->Unmap(0, nullptr);
            return false;
        }
        const double ms = double(end - start) * 1000.0 / double(ctx.timestampFrequency);
        result.minMs = std::min(result.minMs, ms);
        result.maxMs = std::max(result.maxMs, ms);
        total += ms;
    }
    result.avgMs = total / double(timed);
    tsReadback->Unmap(0, nullptr);
    return true;
}

} // namespace gpu
