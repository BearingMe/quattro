#include "graph.h"
#include "d3d12util.h"

#include <cstring>
#include <sstream>

#pragma comment(lib, "d3d12.lib")

namespace gpu {
using d3d12::BufferDesc;
using d3d12::HeapProps;
using d3d12::TransitionBarrier;

bool IsBufferPassWmmaDiagnostic(const D3D12_MESSAGE& message) {
    if (!message.pDescription || message.ID != D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_ROOT_DESCRIPTOR_ACCESS_OUT_OF_BOUNDS)
        return false;
    const char* text = message.pDescription;
    return strstr(text, "GPU-BASED VALIDATION") &&
           strstr(text, "Root descriptor access out of bounds") &&
           strstr(text, "QUATTRO_WMMA_INTRINSIC_DUMMY") &&
           strstr(text, "Root Descriptor Type: UAV") &&
           strstr(text, "Root Parameter Index: [4]") &&
           strstr(text, "Highest byte offset from view start accessed: [1342177324]") &&
           strstr(text, "Bytes available from view start based on remaining resource size: 16.");
}

bool CreateBufferPassRootSignature(DeviceContext& ctx,
                                   ComPtr<ID3D12RootSignature>& rootSignature,
                                   std::string& error, bool nativeWmma) {
    D3D12_ROOT_PARAMETER params[5]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[0].Descriptor.ShaderRegister = 0; // u0
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0; // t0
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 1; // t1
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 1; // u1
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 0;
    params[4].Descriptor.RegisterSpace = 0x7FFF0ADE;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = nativeWmma ? 5 : 4;
    rsDesc.pParameters = params;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob, rsError;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob,
                                           &rsError))) {
        error = "D3D12SerializeRootSignature failed";
        if (rsError)
            error += std::string(": ") +
                     static_cast<const char*>(rsError->GetBufferPointer());
        return false;
    }
    if (FAILED(ctx.device->CreateRootSignature(0, blob->GetBufferPointer(),
                                               blob->GetBufferSize(),
                                               IID_PPV_ARGS(&rootSignature)))) {
        error = "CreateRootSignature (buffer passes) failed";
        return false;
    }
    return true;
}

bool RunBufferPasses(DeviceContext& ctx, ComPtr<ID3D12RootSignature> rootSignature,
                     const std::vector<BufferInit>& buffers,
                     const std::vector<BufferPass>& passes,
                     size_t snapshotBuffer, const std::vector<bool>& snapshotAfter,
                     std::vector<std::vector<uint8_t>>& snapshots,
                     size_t outputBuffer, size_t outputBytes,
                     std::vector<uint8_t>& output, std::string& error, bool nativeWmma) {
    if (!rootSignature || buffers.empty() || passes.empty() ||
        snapshotBuffer >= buffers.size() || outputBuffer >= buffers.size() ||
        buffers[outputBuffer].bytes < outputBytes ||
        snapshotAfter.size() != passes.size()) {
        error = "invalid buffer-pass descriptor";
        return false;
    }
    for (size_t i = 0; i < passes.size(); ++i) {
        const BufferPass& p = passes[i];
        if (!p.pso || p.gridX == 0 || p.gridY == 0 || p.gridZ == 0 ||
            p.srv0 >= buffers.size() || p.srv1 >= buffers.size() ||
            p.uav0 >= buffers.size() || p.uav1 >= buffers.size() || p.uav0 == p.uav1 ||
            p.srv0 == p.uav0 || p.srv0 == p.uav1 ||
            p.srv1 == p.uav0 || p.srv1 == p.uav1) {
            error = "invalid buffer pass " + std::to_string(i);
            return false;
        }
    }

    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             ctx.allocator.Get(), nullptr,
                                             IID_PPV_ARGS(&list)))) {
        error = "CreateCommandList failed";
        return false;
    }

    size_t totalBytes = 0;
    for (const BufferInit& b : buffers) {
        if (b.bytes == 0 || b.bytes > SIZE_MAX - totalBytes) {
            error = "invalid buffer size or total size overflow";
            return false;
        }
        totalBytes += b.bytes;
    }

    const D3D12_HEAP_PROPERTIES defaultHeap = HeapProps(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_HEAP_PROPERTIES uploadHeap = HeapProps(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_HEAP_PROPERTIES readbackHeap = HeapProps(D3D12_HEAP_TYPE_READBACK);

    ComPtr<ID3D12Resource> intrinsicDummy;
    if (nativeWmma) {
        const auto desc = BufferDesc(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (FAILED(ctx.device->CreateCommittedResource(
                &defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&intrinsicDummy)))) {
            error = "cannot create WMMA intrinsic dummy";
            return false;
        }
        intrinsicDummy->SetName(L"QUATTRO_WMMA_INTRINSIC_DUMMY");
    }

    std::vector<ComPtr<ID3D12Resource>> buf(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        const D3D12_RESOURCE_DESC desc =
            BufferDesc(buffers[i].bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (FAILED(ctx.device->CreateCommittedResource(
                &defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf[i])))) {
            error = "CreateCommittedResource (pass buffer) failed";
            return false;
        }
    }

    ComPtr<ID3D12Resource> upload;
    {
        const D3D12_RESOURCE_DESC desc = BufferDesc(totalBytes, D3D12_RESOURCE_FLAG_NONE);
        if (FAILED(ctx.device->CreateCommittedResource(
                &uploadHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)))) {
            error = "CreateCommittedResource (pass upload) failed";
            return false;
        }
    }

    size_t snapshotCount = 0;
    for (bool s : snapshotAfter)
        snapshotCount += s ? 1 : 0;
    std::vector<ComPtr<ID3D12Resource>> snapshotRb(snapshotCount);
    for (size_t i = 0; i < snapshotCount; ++i) {
        const D3D12_RESOURCE_DESC desc =
            BufferDesc(buffers[snapshotBuffer].bytes, D3D12_RESOURCE_FLAG_NONE);
        if (FAILED(ctx.device->CreateCommittedResource(
                &readbackHeap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&snapshotRb[i])))) {
            error = "CreateCommittedResource (pass snapshot readback) failed";
            return false;
        }
    }
    const D3D12_RESOURCE_DESC outRbDesc = BufferDesc(outputBytes, D3D12_RESOURCE_FLAG_NONE);
    ComPtr<ID3D12Resource> outRb;
    if (FAILED(ctx.device->CreateCommittedResource(
            &readbackHeap, D3D12_HEAP_FLAG_NONE, &outRbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outRb)))) {
        error = "CreateCommittedResource (pass output readback) failed";
        return false;
    }

    // --- Initial data: one upload region per buffer. ---
    {
        void* mapped = nullptr;
        if (FAILED(upload->Map(0, nullptr, &mapped))) {
            error = "upload Map failed";
            return false;
        }
        size_t offset = 0;
        for (size_t i = 0; i < buffers.size(); ++i) {
            uint8_t* region = static_cast<uint8_t*>(mapped) + offset;
            if (buffers[i].data)
                memcpy(region, buffers[i].data, buffers[i].bytes);
            else
                memset(region, 0, buffers[i].bytes);
            const D3D12_RESOURCE_BARRIER toCopy = TransitionBarrier(
                buf[i].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            list->ResourceBarrier(1, &toCopy);
            list->CopyBufferRegion(buf[i].Get(), 0, upload.Get(), offset, buffers[i].bytes);
            const D3D12_RESOURCE_BARRIER toCommon = TransitionBarrier(
                buf[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(1, &toCommon);
            offset += buffers[i].bytes;
        }
        upload->Unmap(0, nullptr);
    }

    list->SetComputeRootSignature(rootSignature.Get());
    if (nativeWmma)
        list->SetComputeRootUnorderedAccessView(4, intrinsicDummy->GetGPUVirtualAddress());
    size_t snapIndex = 0;
    for (size_t i = 0; i < passes.size(); ++i) {
        const BufferPass& p = passes[i];
        D3D12_RESOURCE_BARRIER transitions[4];
        uint32_t count = 0;
        transitions[count++] = TransitionBarrier(buf[p.srv0].Get(), D3D12_RESOURCE_STATE_COMMON,
                                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (p.srv1 != p.srv0)
            transitions[count++] = TransitionBarrier(
                buf[p.srv1].Get(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transitions[count++] = TransitionBarrier(buf[p.uav0].Get(), D3D12_RESOURCE_STATE_COMMON,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (p.uav1 != p.uav0)
            transitions[count++] = TransitionBarrier(
                buf[p.uav1].Get(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(count, transitions);

        list->SetPipelineState(p.pso);
        list->SetComputeRootUnorderedAccessView(0, buf[p.uav0]->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(1, buf[p.srv0]->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2, buf[p.srv1]->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(3, buf[p.uav1]->GetGPUVirtualAddress());
        list->Dispatch(p.gridX, p.gridY, p.gridZ);

        D3D12_RESOURCE_BARRIER uavOrder{};
        uavOrder.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        list->ResourceBarrier(1, &uavOrder);

        count = 0;
        transitions[count++] = TransitionBarrier(
            buf[p.srv0].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
        if (p.srv1 != p.srv0)
            transitions[count++] = TransitionBarrier(
                buf[p.srv1].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        transitions[count++] = TransitionBarrier(
            buf[p.uav0].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON);
        if (p.uav1 != p.uav0)
            transitions[count++] = TransitionBarrier(
                buf[p.uav1].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON);
        list->ResourceBarrier(count, transitions);

        if (snapshotAfter[i]) {
            const D3D12_RESOURCE_BARRIER toCopy = TransitionBarrier(
                buf[snapshotBuffer].Get(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &toCopy);
            list->CopyBufferRegion(snapshotRb[snapIndex].Get(), 0,
                                   buf[snapshotBuffer].Get(), 0,
                                   buffers[snapshotBuffer].bytes);
            const D3D12_RESOURCE_BARRIER toCommon = TransitionBarrier(
                buf[snapshotBuffer].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(1, &toCommon);
            ++snapIndex;
        }
    }

    {
        const D3D12_RESOURCE_BARRIER toCopy = TransitionBarrier(
            buf[outputBuffer].Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &toCopy);
        list->CopyBufferRegion(outRb.Get(), 0, buf[outputBuffer].Get(), 0, outputBytes);
        const D3D12_RESOURCE_BARRIER toCommon = TransitionBarrier(
            buf[outputBuffer].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON);
        list->ResourceBarrier(1, &toCommon);
    }

    if (FAILED(list->Close())) {
        error = "Close buffer-pass command list failed";
        return false;
    }
    ID3D12CommandList* lists[] = {list.Get()};
    ctx.queue->ExecuteCommandLists(1, lists);
    if (!GpuSync(ctx, 30000, error)) {
        const HRESULT removed = ctx.device->GetDeviceRemovedReason();
        std::string queueErrors;
        InfoQueueClean(ctx, queueErrors);
        std::ostringstream out;
        out << "gpu sync after buffer passes failed: " << error
            << "; GetDeviceRemovedReason=0x" << std::hex << uint32_t(removed)
            << (queueErrors.empty() ? std::string() : "; " + queueErrors);
        error = out.str();
        return false;
    }
    if (const HRESULT removed = ctx.device->GetDeviceRemovedReason(); FAILED(removed)) {
        std::string queueErrors;
        InfoQueueClean(ctx, queueErrors);
        std::ostringstream out;
        out << "device removed during buffer passes (GetDeviceRemovedReason=0x"
            << std::hex << uint32_t(removed) << ")"
            << (queueErrors.empty() ? std::string() : "; " + queueErrors);
        error = out.str();
        return false;
    }

    snapshots.assign(snapshotCount, {});
    for (size_t i = 0; i < snapshotCount; ++i) {
        void* raw = nullptr;
        if (FAILED(snapshotRb[i]->Map(0, nullptr, &raw))) {
            error = "snapshot Map failed (entry " + std::to_string(i) + ")";
            return false;
        }
        snapshots[i].assign(static_cast<const uint8_t*>(raw),
                            static_cast<const uint8_t*>(raw) + buffers[snapshotBuffer].bytes);
        snapshotRb[i]->Unmap(0, nullptr);
    }
    void* rawOut = nullptr;
    if (FAILED(outRb->Map(0, nullptr, &rawOut))) {
        error = "output Map failed";
        return false;
    }
    output.assign(static_cast<const uint8_t*>(rawOut),
                  static_cast<const uint8_t*>(rawOut) + outputBytes);
    outRb->Unmap(0, nullptr);
    return true;
}

} // namespace gpu
