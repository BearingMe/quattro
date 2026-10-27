#include "device.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstring>
#include <sstream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// windows.h (via winuser.h) defines GetMessage as a macro; the
// ID3D12InfoQueue method has the same name.
#undef GetMessage

#include <vector>

namespace gpu {
namespace {

AdapterInfo ToAdapterInfo(const DXGI_ADAPTER_DESC1& d) {
    AdapterInfo info;
    info.name = d.Description;
    info.vendorId = d.VendorId;
    info.deviceId = d.DeviceId;
    info.dedicatedVideoMemory = d.DedicatedVideoMemory;
    info.luidHighLow = (uint64_t(d.AdapterLuid.HighPart) << 32) | d.AdapterLuid.LowPart;
    return info;
}

bool ContainsRadeon(const std::wstring& name) {
    return name.find(L"Radeon") != std::wstring::npos;
}

std::string Narrow(const std::wstring& wide) {
    std::string narrow;
    narrow.reserve(wide.size());
    for (wchar_t c : wide)
        narrow.push_back(c >= 0 && c < 128 ? char(c) : '?');
    return narrow;
}

} // namespace

bool CreateComputeDevice(DeviceContext& out, std::string& error) {
    // Diagnostics first: debug layer + GPU-based validation must be enabled
    // before device creation (M05 C07).
    ComPtr<ID3D12Debug> debug;
    ComPtr<ID3D12Debug1> debug1;
    const bool debugLayerAvailable =
        SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))) &&
        SUCCEEDED(debug.As(&debug1));
    if (!debugLayerAvailable) {
        error = "D3D12 debug layer unavailable (install the Windows \"Graphics Tools\" "
                "optional feature); validation diagnostics required by M05 are missing";
        return false;
    }
    debug->EnableDebugLayer();
    debug1->SetEnableGPUBasedValidation(TRUE);

    const UINT factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory)))) {
        error = "CreateDXGIFactory2 failed";
        return false;
    }

    // Explicit adapter selection: enumerate everything, dump it, then pick the
    // physical AMD card. The environment contains a "SudoMaker Virtual Display
    // Adapter" reporting the same PCI IDs and VRAM as the RX 7700 XT, so
    // "the first adapter" or PCI-ID matching alone is not safe. Hardware
    // adapters with "Radeon" in the description are candidates; highest
    // dedicated VRAM wins.
    DXGI_ADAPTER_DESC1 chosen{};
    ComPtr<IDXGIAdapter1> chosenAdapter;
    bool haveChosen = false;
    std::wostringstream dump;
    dump << L"adapter enumeration:";
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)))
            continue;
        const bool software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        dump << L"\n  [" << i << "] " << desc.Description
             << (software ? L" (software)" : L" (hardware)")
             << L", vid=0x" << std::hex << desc.VendorId
             << L" did=0x" << desc.DeviceId << std::dec
             << L", vram=" << (desc.DedicatedVideoMemory >> 20) << L" MB";
        if (!software && ContainsRadeon(desc.Description) &&
            (!haveChosen || desc.DedicatedVideoMemory > chosen.DedicatedVideoMemory)) {
            chosen = desc;
            chosenAdapter = adapter;
            haveChosen = true;
        }
    }

    if (!haveChosen) {
        error = "no hardware AMD Radeon adapter found; " + Narrow(dump.str());
        return false;
    }
    out.adapter = ToAdapterInfo(chosen);

    if (FAILED(D3D12CreateDevice(chosenAdapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                 IID_PPV_ARGS(&out.device)))) {
        error = "D3D12CreateDevice failed on the selected adapter";
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(out.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&out.queue)))) {
        error = "CreateCommandQueue failed";
        return false;
    }
    if (FAILED(out.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&out.allocator)))) {
        error = "CreateCommandAllocator failed";
        return false;
    }
    if (FAILED(out.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&out.fence)))) {
        error = "CreateFence failed";
        return false;
    }
    out.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!out.fenceEvent) {
        error = "CreateEventW failed";
        return false;
    }
    out.queue->GetTimestampFrequency(&out.timestampFrequency);

    out.device->QueryInterface(IID_PPV_ARGS(&out.infoQueue));
    if (out.infoQueue)
        out.infoQueue->SetMessageCountLimit(1024);
    return true;
}

bool GpuSync(DeviceContext& ctx, uint32_t timeoutMs, std::string& error) {
    static uint64_t value = 0;
    ++value;
    if (FAILED(ctx.queue->Signal(ctx.fence.Get(), value))) {
        error = "fence Signal failed";
        return false;
    }
    // Only register + wait when completion has not already happened.
    if (ctx.fence->GetCompletedValue() < value) {
        if (FAILED(ctx.fence->SetEventOnCompletion(value, ctx.fenceEvent))) {
            error = "SetEventOnCompletion failed";
            return false;
        }
        const DWORD wait = WaitForSingleObject(ctx.fenceEvent, timeoutMs);
        if (wait != WAIT_OBJECT_0) {
            std::ostringstream out;
            out << "fence wait timed out or failed (wait result " << wait
                << ", waiting for value " << value
                << ", completed value " << ctx.fence->GetCompletedValue() << ")";
            error = out.str();
            return false;
        }
    }
    return true;
}

bool InfoQueueCleanWithFilter(DeviceContext& ctx,
                              const std::function<bool(const D3D12_MESSAGE&)>& benign,
                              size_t& benignCount, std::string& errors) {
    benignCount = 0;
    if (!ctx.infoQueue) {
        errors.clear();
        return true; // diagnostics unavailable — surfaced at creation
    }
    const UINT64 count = ctx.infoQueue->GetNumStoredMessages();
    bool clean = true;
    std::ostringstream out;
    for (UINT64 i = 0; i < count; ++i) {
        // First call with nullptr retrieves the message byte length.
        SIZE_T byteLength = 0;
        if (FAILED(ctx.infoQueue->GetMessage(i, nullptr, &byteLength)) || byteLength == 0)
            continue;
        std::vector<char> storage(byteLength);
        D3D12_MESSAGE* msg = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(ctx.infoQueue->GetMessage(i, msg, &byteLength)))
            continue;
        if (msg->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
            msg->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
            if (benign && benign(*msg)) {
                ++benignCount;
                continue;
            }
            clean = false;
            out << "info queue error: "
                << (msg->pDescription ? msg->pDescription : "(no description)") << "\n";
        }
    }
    ctx.infoQueue->ClearStoredMessages();
    errors = out.str();
    return clean;
}

bool InfoQueueClean(DeviceContext& ctx, std::string& errors) {
    size_t ignored = 0;
    return InfoQueueCleanWithFilter(ctx, {}, ignored, errors);
}

bool InfoQueueCleanGbavBenign(DeviceContext& ctx, size_t& benignCount, std::string& errors) {
    return InfoQueueCleanWithFilter(
        ctx,
        [](const D3D12_MESSAGE& msg) {
            if (!msg.pDescription)
                return false;
            const char* desc = msg.pDescription;
            return strstr(desc, "GPU-BASED VALIDATION") != nullptr &&
                   strstr(desc, "Root descriptor access out of bounds") != nullptr &&
                   strstr(desc, "Root Parameter Index: [2]") != nullptr;
        },
        benignCount, errors);
}

} // namespace gpu
