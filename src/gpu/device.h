// gpu — D3D12 device creation, adapter selection, sync, diagnostics.
// Dependency rule (docs/architecture.md §3): gpu depends on core only;
// it must never know FSR 4 model semantics.

#pragma once

// windows.h must come first and its GetMessage macro must die before d3d12.h
// parses, or ID3D12InfoQueue::GetMessage is declared as GetMessageA.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef GetMessage

#include <d3d12.h>
#include <functional>
#include <string>
#include <wrl/client.h>

namespace gpu {

using Microsoft::WRL::ComPtr;

struct AdapterInfo {
    std::wstring name;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    size_t dedicatedVideoMemory = 0;
    uint64_t luidHighLow = 0; // packed for logging
};

struct DeviceContext {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12Fence> fence;
    void* fenceEvent = nullptr;
    ComPtr<ID3D12InfoQueue> infoQueue;
    AdapterInfo adapter;
    uint64_t timestampFrequency = 0;
};

// Selects the physical AMD adapter explicitly (hardware adapters only,
// "Radeon" in description, highest dedicated VRAM wins) and creates a
// compute device with the debug layer + GPU-based validation enabled.
// Returns false and fills `error` when unavailable.
bool CreateComputeDevice(DeviceContext& out, std::string& error);

// Signals the fence after everything queued so far and waits for it.
bool GpuSync(DeviceContext& ctx, uint32_t timeoutMs, std::string& error);

// Drains the D3D12 info queue. Returns true when no error-severity messages
// were stored; otherwise `errors` lists them (experiment must fail).
bool InfoQueueClean(DeviceContext& ctx, std::string& errors);

// Same verdict, but error-severity messages for which `benign` returns true
// are counted in `benignCount` instead of failing the experiment — for
// known, evidence-backed false positives (e.g. GPU-based validation shadow
// passes over AMD intrinsics dummy UAVs). Everything else still fails.
bool InfoQueueCleanWithFilter(DeviceContext& ctx,
                              const std::function<bool(const D3D12_MESSAGE&)>& benign,
                              size_t& benignCount, std::string& errors);

} // namespace gpu
