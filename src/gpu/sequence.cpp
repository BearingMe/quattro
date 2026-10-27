#include "sequence.h"
#include "d3d12util.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace gpu {
using d3d12::BufferDesc;
using d3d12::HeapProps;
using d3d12::TransitionBarrier;

bool RunDispatchSequence(
    DeviceContext& ctx, ComPtr<ID3D12RootSignature> rootSignature,
    const std::vector<SequenceBufferDesc>& buffers,
    const std::vector<DispatchStage>& stages,
    const std::vector<SequenceSnapshotDesc>& snapshots,
    size_t outputBuffer, size_t outputBytes,
    uint32_t warmupNetworks, uint32_t timedNetworks,
    std::vector<std::vector<uint8_t>>& snapshotData,
    std::vector<uint8_t>& outputWarmup, std::vector<uint8_t>& outputTimed,
    MatmulRunResult& timing, std::string& error) {
    if (buffers.empty() || stages.empty() || outputBuffer >= buffers.size() ||
        buffers[outputBuffer].bytes < outputBytes || timedNetworks == 0) {
        error = "invalid sequence descriptor (empty buffers/stages, bad output, "
                "or zero timed networks)";
        return false;
    }
    for (size_t i = 0; i < stages.size(); ++i) {
        const DispatchStage& stage = stages[i];
        if (!stage.pso || stage.srvBuffer >= buffers.size() ||
            stage.uavBuffer >= buffers.size() || stage.srvBuffer == stage.uavBuffer ||
            stage.gridX == 0 || stage.gridY == 0) {
            error = "invalid dispatch stage " + std::to_string(i);
            return false;
        }
    }
    for (const SequenceSnapshotDesc& snap : snapshots)
        if (snap.buffer >= buffers.size() ||
            snap.offset + snap.bytes > buffers[snap.buffer].bytes) {
            error = "snapshot outside its buffer";
            return false;
        }

    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             ctx.allocator.Get(), nullptr,
                                             IID_PPV_ARGS(&list)))) {
        error = "CreateCommandList failed";
        return false;
    }

    // --- Resources: default-heap buffers, one upload heap, readbacks, timestamps. ---
    const D3D12_HEAP_PROPERTIES defaultHeap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES uploadHeap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readbackHeap{D3D12_HEAP_TYPE_READBACK};

    // Upload staging: one committed region PER buffer (sum of sizes) — the
    // recorded GPU copies each read their own offset, so a single reused
    // region would race them.
    size_t maxBytes = 0;
    for (const SequenceBufferDesc& desc : buffers)
        maxBytes += desc.bytes;

    std::vector<ComPtr<ID3D12Resource>> buf(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        const D3D12_RESOURCE_DESC desc =
            BufferDesc(buffers[i].bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (FAILED(ctx.device->CreateCommittedResource(
                &defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf[i])))) {
            error = "CreateCommittedResource (sequence buffer) failed";
            return false;
        }
    }
    // Every buffer is bound as UAV at least once in the intended use; the
    // ALLOW_UNORDERED_ACCESS flag is required for that and harmless otherwise.

    ComPtr<ID3D12Resource> upload;
    {
        const D3D12_RESOURCE_DESC desc =
            BufferDesc(maxBytes, D3D12_RESOURCE_FLAG_NONE);
        if (FAILED(ctx.device->CreateCommittedResource(
                &uploadHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)))) {
            error = "CreateCommittedResource (sequence upload) failed";
            return false;
        }
    }

    // Readback: one per snapshot descriptor + warmup/timed output copies.
    std::vector<ComPtr<ID3D12Resource>> snapReadback(snapshots.size());
    for (size_t i = 0; i < snapshots.size(); ++i) {
        const D3D12_RESOURCE_DESC desc =
            BufferDesc(snapshots[i].bytes, D3D12_RESOURCE_FLAG_NONE);
        if (FAILED(ctx.device->CreateCommittedResource(
                &readbackHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&snapReadback[i])))) {
            error = "CreateCommittedResource (snapshot readback) failed";
            return false;
        }
    }
    const D3D12_RESOURCE_DESC outRbDesc = BufferDesc(outputBytes, D3D12_RESOURCE_FLAG_NONE);
    ComPtr<ID3D12Resource> outReadbackWarmup, outReadbackTimed;
    if (FAILED(ctx.device->CreateCommittedResource(
            &readbackHeap, D3D12_HEAP_FLAG_NONE, &outRbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outReadbackWarmup))) ||
        FAILED(ctx.device->CreateCommittedResource(
            &readbackHeap, D3D12_HEAP_FLAG_NONE, &outRbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outReadbackTimed)))) {
        error = "CreateCommittedResource (output readback) failed";
        return false;
    }

    // AMD intrinsics dummy UAV (driver-intercepted, M06 pattern) — required
    // by the shared WMMA root signature even for intrinsics-free stages.
    const D3D12_RESOURCE_DESC dummyDesc = BufferDesc(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComPtr<ID3D12Resource> bufDummy;
    if (FAILED(ctx.device->CreateCommittedResource(
            &defaultHeap, D3D12_HEAP_FLAG_NONE, &dummyDesc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&bufDummy)))) {
        error = "CreateCommittedResource (intrinsics dummy) failed";
        return false;
    }

    const D3D12_RESOURCE_DESC tsDesc =
        BufferDesc(uint64_t(timedNetworks) * 2 * sizeof(uint64_t), D3D12_RESOURCE_FLAG_NONE);
    ComPtr<ID3D12Resource> tsReadback;
    if (FAILED(ctx.device->CreateCommittedResource(
            &readbackHeap, D3D12_HEAP_FLAG_NONE, &tsDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tsReadback)))) {
        error = "CreateCommittedResource (timestamp readback) failed";
        return false;
    }

    ComPtr<ID3D12QueryHeap> queryHeap;
    D3D12_QUERY_HEAP_DESC queryDesc{};
    queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    queryDesc.Count = timedNetworks * 2;
    if (FAILED(ctx.device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queryHeap)))) {
        error = "CreateQueryHeap failed";
        return false;
    }

    // --- Initial data: each buffer gets its own upload-heap region — a single
    // mapped region reused per buffer would race the recorded GPU copies. ---
    {
        void* uploadData = nullptr;
        if (FAILED(upload->Map(0, nullptr, &uploadData))) {
            error = "upload Map failed";
            return false;
        }
        size_t uploadOffset = 0;
        for (size_t i = 0; i < buffers.size(); ++i) {
            uint8_t* region = static_cast<uint8_t*>(uploadData) + uploadOffset;
            if (buffers[i].initialData)
                memcpy(region, buffers[i].initialData, buffers[i].bytes);
            else
                memset(region, 0, buffers[i].bytes);
            const D3D12_RESOURCE_BARRIER toCopy =
                TransitionBarrier(buf[i].Get(), D3D12_RESOURCE_STATE_COMMON,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
            list->ResourceBarrier(1, &toCopy);
            list->CopyBufferRegion(buf[i].Get(), 0, upload.Get(), uploadOffset, buffers[i].bytes);
            const D3D12_RESOURCE_BARRIER toCommon =
                TransitionBarrier(buf[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                  D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(1, &toCommon);
            uploadOffset += buffers[i].bytes;
        }
        upload->Unmap(0, nullptr);
    }

    // --- Network execution: all stages in order, explicit state transitions
    // around each stage (buffers rest at COMMON; COMMON promotes to SRV/UAV). ---
    const auto runNetwork = [&]() {
        for (const DispatchStage& stage : stages) {
            D3D12_RESOURCE_BARRIER toSrv = TransitionBarrier(
                buf[stage.srvBuffer].Get(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            D3D12_RESOURCE_BARRIER toUav = TransitionBarrier(
                buf[stage.uavBuffer].Get(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            D3D12_RESOURCE_BARRIER both[] = {toSrv, toUav};
            list->ResourceBarrier(2, both);
            list->SetComputeRootSignature(rootSignature.Get());
            list->SetPipelineState(stage.pso);
            list->SetComputeRootUnorderedAccessView(0, buf[stage.uavBuffer]->GetGPUVirtualAddress());
            list->SetComputeRootShaderResourceView(1, buf[stage.srvBuffer]->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(2, bufDummy->GetGPUVirtualAddress());
            list->Dispatch(stage.gridX, stage.gridY, 1);
            D3D12_RESOURCE_BARRIER uavOrder{};
            uavOrder.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; // null = order all UAV writes
            list->ResourceBarrier(1, &uavOrder);
            D3D12_RESOURCE_BARRIER srvBack = TransitionBarrier(
                buf[stage.srvBuffer].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
            D3D12_RESOURCE_BARRIER uavBack = TransitionBarrier(
                buf[stage.uavBuffer].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON);
            D3D12_RESOURCE_BARRIER back[] = {srvBack, uavBack};
            list->ResourceBarrier(2, back);
        }
    };

    for (uint32_t i = 0; i < warmupNetworks; ++i)
        runNetwork();

    // Correctness snapshots: exactly the state after `warmupNetworks` sequences.
    std::vector<size_t> copySources;
    copySources.reserve(snapshots.size() + 1);
    for (size_t i = 0; i < snapshots.size(); ++i)
        copySources.push_back(snapshots[i].buffer);
    copySources.push_back(outputBuffer);
    std::sort(copySources.begin(), copySources.end());
    copySources.erase(std::unique(copySources.begin(), copySources.end()), copySources.end());
    for (size_t b : copySources) {
        const D3D12_RESOURCE_BARRIER toCopy = TransitionBarrier(
            buf[b].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &toCopy);
    }
    for (size_t i = 0; i < snapshots.size(); ++i)
        list->CopyBufferRegion(snapReadback[i].Get(), 0, buf[snapshots[i].buffer].Get(),
                               snapshots[i].offset, snapshots[i].bytes);
    list->CopyBufferRegion(outReadbackWarmup.Get(), 0, buf[outputBuffer].Get(), 0, outputBytes);
    for (size_t b : copySources) {
        const D3D12_RESOURCE_BARRIER toCommon = TransitionBarrier(
            buf[b].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        list->ResourceBarrier(1, &toCommon);
    }

    // Timed loop: each sequence must be idempotent (outputs fully rewritten
    // from unmodified inputs); the caller verifies warmup == timed snapshot.
    for (uint32_t i = 0; i < timedNetworks; ++i) {
        list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2);
        runNetwork();
        list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2 + 1);
    }

    {
        const D3D12_RESOURCE_BARRIER toCopy = TransitionBarrier(
            buf[outputBuffer].Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &toCopy);
        list->CopyBufferRegion(outReadbackTimed.Get(), 0, buf[outputBuffer].Get(), 0, outputBytes);
        const D3D12_RESOURCE_BARRIER toCommon = TransitionBarrier(
            buf[outputBuffer].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON);
        list->ResourceBarrier(1, &toCommon);
    }

    list->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                           timedNetworks * 2, tsReadback.Get(), 0);
    list->Close();

    ID3D12CommandList* lists[] = {list.Get()};
    ctx.queue->ExecuteCommandLists(1, lists);
    if (!GpuSync(ctx, 10000, error)) {
        const HRESULT removed = ctx.device->GetDeviceRemovedReason();
        std::string queueErrors;
        InfoQueueClean(ctx, queueErrors);
        std::ostringstream out;
        out << "gpu sync after sequence failed: " << error << "; GetDeviceRemovedReason=0x"
            << std::hex << (uint32_t)removed
            << (queueErrors.empty() ? std::string() : "; " + queueErrors);
        error = out.str();
        return false;
    }
    // A fault can surface AFTER the fence signals (all subsequent work and
    // readbacks then see zeros / DXGI_ERROR_DEVICE_REMOVED). Fail loudly here.
    if (const HRESULT removed = ctx.device->GetDeviceRemovedReason(); FAILED(removed)) {
        std::ostringstream out;
        out << "device removed during sequence execution (GetDeviceRemovedReason=0x"
            << std::hex << (uint32_t)removed << ")";
        error = out.str();
        return false;
    }

    // --- Readback. ---
    snapshotData.assign(snapshots.size(), {});
    for (size_t i = 0; i < snapshots.size(); ++i) {
        void* raw = nullptr;
        const HRESULT snapHr = snapReadback[i]->Map(0, nullptr, &raw);
        if (FAILED(snapHr)) {
            error = "snapshot readback Map failed (hr=0x" +
                    std::to_string(static_cast<uint32_t>(snapHr)) + ", snapshot " +
                    std::to_string(i) + ")";
            return false;
        }
        snapshotData[i].assign(static_cast<const uint8_t*>(raw),
                               static_cast<const uint8_t*>(raw) + snapshots[i].bytes);
        snapReadback[i]->Unmap(0, nullptr);
    }
    void* rawWarmup = nullptr;
    void* rawTimed = nullptr;
    if (FAILED(outReadbackWarmup->Map(0, nullptr, &rawWarmup)) ||
        FAILED(outReadbackTimed->Map(0, nullptr, &rawTimed))) {
        error = "output readback Map failed";
        return false;
    }
    outputWarmup.assign(static_cast<const uint8_t*>(rawWarmup),
                        static_cast<const uint8_t*>(rawWarmup) + outputBytes);
    outputTimed.assign(static_cast<const uint8_t*>(rawTimed),
                       static_cast<const uint8_t*>(rawTimed) + outputBytes);
    outReadbackWarmup->Unmap(0, nullptr);
    outReadbackTimed->Unmap(0, nullptr);

    uint64_t* stamps = nullptr;
    if (FAILED(tsReadback->Map(0, nullptr, reinterpret_cast<void**>(&stamps)))) {
        error = "timestamp Map failed";
        return false;
    }
    timing.iterations = timedNetworks;
    timing.minMs = 1e30;
    timing.maxMs = 0.0;
    double total = 0.0;
    for (uint32_t i = 0; i < timedNetworks; ++i) {
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
    timing.avgMs = total / double(timedNetworks);
    tsReadback->Unmap(0, nullptr);
    return true;
}

} // namespace gpu
