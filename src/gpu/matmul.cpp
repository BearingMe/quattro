#include "matmul.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#pragma comment(lib, "d3d12.lib")

namespace gpu {

bool CreateGemmTileRootSignature(DeviceContext& ctx,
                                 ComPtr<ID3D12RootSignature>& rootSignature,
                                 std::string& error) {
    D3D12_ROOT_PARAMETER params[3] = {};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[0].Descriptor.ShaderRegister = 0; // u0 space0 (output)
    params[0].Descriptor.RegisterSpace = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0; // t0 space0 (input)
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0; // intrinsics dummy UAV
    params[2].Descriptor.RegisterSpace = 0x7FFF0ADE;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 3;
    rsDesc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, nullptr))) {
        error = "D3D12SerializeRootSignature failed";
        return false;
    }
    if (FAILED(ctx.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                               IID_PPV_ARGS(&rootSignature)))) {
        error = "CreateRootSignature failed";
        return false;
    }
    return true;
}

namespace {

D3D12_HEAP_PROPERTIES HeapProps(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = type;
    return props;
}

D3D12_RESOURCE_DESC BufferDesc(uint64_t bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
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

} // namespace

bool RunGemmTile(DeviceContext& ctx, ComPtr<ID3D12RootSignature> rootSignature,
                 ComPtr<ID3D12PipelineState> pso, size_t inputBytes, size_t outputBytes,
                 uint32_t groupCount, uint32_t warmup, uint32_t timed, const void* input,
                 std::vector<int32_t>& outPostWarmup, std::vector<int32_t>& outPostTimed,
                 MatmulRunResult& timing, std::string& error) {
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             ctx.allocator.Get(), nullptr,
                                             IID_PPV_ARGS(&list)))) {
        error = "CreateCommandList failed";
        return false;
    }

    // --- Resources: gpu input + upload, gpu output, intrinsics dummy, readbacks, timestamp buffer. ---
    const D3D12_HEAP_PROPERTIES defaultHeap = HeapProps(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_HEAP_PROPERTIES uploadHeap = HeapProps(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_HEAP_PROPERTIES readbackHeap = HeapProps(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC inDesc = BufferDesc(inputBytes, D3D12_RESOURCE_FLAG_NONE);
    const D3D12_RESOURCE_DESC outDesc = BufferDesc(outputBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const D3D12_RESOURCE_DESC dummyDesc = BufferDesc(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const D3D12_RESOURCE_DESC rbDesc = BufferDesc(outputBytes, D3D12_RESOURCE_FLAG_NONE);
    const D3D12_RESOURCE_DESC tsDesc = BufferDesc(uint64_t(timed) * 2 * sizeof(uint64_t), D3D12_RESOURCE_FLAG_NONE);

    ComPtr<ID3D12Resource> bufIn, bufOut, bufDummy, upload, readbackWarmup, readbackTimed, tsReadback;
    HRESULT hr = ctx.device->CreateCommittedResource(
        &defaultHeap, D3D12_HEAP_FLAG_NONE, &inDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bufIn));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (gpu input) failed";
        return false;
    }
    hr = ctx.device->CreateCommittedResource(
        &defaultHeap, D3D12_HEAP_FLAG_NONE, &outDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bufOut));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (gpu output) failed";
        return false;
    }
    hr = ctx.device->CreateCommittedResource(
        &defaultHeap, D3D12_HEAP_FLAG_NONE, &dummyDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bufDummy));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (intrinsics dummy) failed";
        return false;
    }
    hr = ctx.device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &inDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (upload) failed";
        return false;
    }
    hr = ctx.device->CreateCommittedResource(
        &readbackHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackWarmup));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (readback warmup) failed";
        return false;
    }
    hr = ctx.device->CreateCommittedResource(
        &readbackHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackTimed));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (readback timed) failed";
        return false;
    }
    hr = ctx.device->CreateCommittedResource(
        &readbackHeap, D3D12_HEAP_FLAG_NONE, &tsDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tsReadback));
    if (FAILED(hr)) {
        error = "CreateCommittedResource (timestamp readback) failed";
        return false;
    }

    void* uploadData = nullptr;
    hr = upload->Map(0, nullptr, &uploadData);
    if (FAILED(hr)) {
        error = "upload Map failed";
        return false;
    }
    memcpy(uploadData, input, inputBytes);
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12QueryHeap> queryHeap;
    D3D12_QUERY_HEAP_DESC queryDesc{};
    queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    queryDesc.Count = timed * 2;
    if (FAILED(ctx.device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queryHeap)))) {
        error = "CreateQueryHeap failed";
        return false;
    }

    // --- Record. ---
    D3D12_RESOURCE_BARRIER toCopyIn = TransitionBarrier(bufIn.Get(), D3D12_RESOURCE_STATE_COMMON,
                                                        D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &toCopyIn);
    list->CopyBufferRegion(bufIn.Get(), 0, upload.Get(), 0, inputBytes);

    D3D12_RESOURCE_BARRIER inToSrv = TransitionBarrier(bufIn.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    D3D12_RESOURCE_BARRIER outToUav = TransitionBarrier(bufOut.Get(), D3D12_RESOURCE_STATE_COMMON,
                                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &inToSrv);
    list->ResourceBarrier(1, &outToUav);

    list->SetComputeRootSignature(rootSignature.Get());
    list->SetPipelineState(pso.Get());
    list->SetComputeRootUnorderedAccessView(0, bufOut->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1, bufIn->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(2, bufDummy->GetGPUVirtualAddress());

    for (uint32_t i = 0; i < warmup; ++i)
        list->Dispatch(groupCount, 1, 1);

    // Correctness snapshot: exactly the state after `warmup` dispatches.
    D3D12_RESOURCE_BARRIER outToCopy1 = TransitionBarrier(bufOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                          D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &outToCopy1);
    list->CopyBufferRegion(readbackWarmup.Get(), 0, bufOut.Get(), 0, outputBytes);
    D3D12_RESOURCE_BARRIER outToUav2 = TransitionBarrier(bufOut.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &outToUav2);

    // Timed loop: each dispatch re-loads its inputs (initC comes from the
    // input SRV, not from the output tile), so the workload is idempotent per
    // dispatch — the timed snapshot must equal the post-warmup snapshot.
    for (uint32_t i = 0; i < timed; ++i) {
        list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2);
        list->Dispatch(groupCount, 1, 1);
        list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2 + 1);
    }

    D3D12_RESOURCE_BARRIER outToCopy2 = TransitionBarrier(bufOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                          D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &outToCopy2);
    list->CopyBufferRegion(readbackTimed.Get(), 0, bufOut.Get(), 0, outputBytes);

    list->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, timed * 2, tsReadback.Get(), 0);
    list->Close();

    ID3D12CommandList* lists[] = {list.Get()};
    ctx.queue->ExecuteCommandLists(1, lists);
    if (!GpuSync(ctx, 10000, error)) {
        const HRESULT removed = ctx.device->GetDeviceRemovedReason();
        std::string queueErrors;
        InfoQueueClean(ctx, queueErrors);
        std::ostringstream out;
        out << "gpu sync after dispatch failed: " << error << "; GetDeviceRemovedReason=0x"
            << std::hex << (uint32_t)removed
            << (queueErrors.empty() ? std::string() : "; " + queueErrors);
        error = out.str();
        return false;
    }

    // --- Readback. ---
    void* rawWarmup = nullptr;
    void* rawTimed = nullptr;
    if (FAILED(readbackWarmup->Map(0, nullptr, &rawWarmup)) ||
        FAILED(readbackTimed->Map(0, nullptr, &rawTimed))) {
        error = "readback Map failed";
        return false;
    }
    const int32_t* dataWarmup = static_cast<const int32_t*>(rawWarmup);
    const int32_t* dataTimed = static_cast<const int32_t*>(rawTimed);
    outPostWarmup.assign(dataWarmup, dataWarmup + outputBytes / sizeof(int32_t));
    outPostTimed.assign(dataTimed, dataTimed + outputBytes / sizeof(int32_t));
    readbackWarmup->Unmap(0, nullptr);
    readbackTimed->Unmap(0, nullptr);

    uint64_t* stamps = nullptr;
    if (FAILED(tsReadback->Map(0, nullptr, reinterpret_cast<void**>(&stamps)))) {
        error = "timestamp Map failed";
        return false;
    }
    timing.iterations = timed;
    timing.minMs = 1e30;
    timing.maxMs = 0.0;
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
        timing.minMs = std::min(timing.minMs, ms);
        timing.maxMs = std::max(timing.maxMs, ms);
        total += ms;
    }
    timing.avgMs = total / double(timed);
    tsReadback->Unmap(0, nullptr);
    return true;
}

} // namespace gpu
