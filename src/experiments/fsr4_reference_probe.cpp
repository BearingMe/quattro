// M14 public API preflight. No binary patching or model-version substitution.
#include "experiments.h"
#include "gpu/device.h"
#include "gpu/d3d12util.h"
#include "inference/models/fsr4_temporal.h"
#include "../../.references/sources/FidelityFX-SDK/Kits/FidelityFX/upscalers/include/ffx_upscale.h"
#include "../../.references/sources/FidelityFX-SDK/Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

namespace experiments {
namespace {

void ApiMessage(uint32_t type, const wchar_t* message) {
    std::fprintf(stderr, "reference API message %u: %ls\n", type, message);
}

bool UploadTexture(gpu::DeviceContext& ctx, ID3D12GraphicsCommandList* list,
                   uint32_t width, uint32_t height, DXGI_FORMAT format, uint32_t bytesPerPixel,
                   const void* source, gpu::ComPtr<ID3D12Resource>& texture,
                   std::vector<gpu::ComPtr<ID3D12Resource>>& uploads, std::string& error) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    const auto heap = gpu::d3d12::HeapProps(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(ctx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)))) {
        error = "cannot create official-reference input texture";
        return false;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint64_t total = 0;
    ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    const auto uploadHeap = gpu::d3d12::HeapProps(D3D12_HEAP_TYPE_UPLOAD);
    const auto buffer = gpu::d3d12::BufferDesc(total, D3D12_RESOURCE_FLAG_NONE);
    gpu::ComPtr<ID3D12Resource> upload;
    if (FAILED(ctx.device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &buffer,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)))) {
        error = "cannot create official-reference upload";
        return false;
    }
    void* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, &mapped))) {
        error = "official-reference upload Map failed";
        return false;
    }
    memset(mapped, 0, size_t(total));
    for (uint32_t y = 0; y < height; ++y)
        memcpy(static_cast<uint8_t*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
               static_cast<const uint8_t*>(source) + size_t(y) * width * bytesPerPixel,
               size_t(width) * bytesPerPixel);
    upload->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION destination{}, origin{};
    destination.pResource = texture.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    origin.pResource = upload.Get();
    origin.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    origin.PlacedFootprint = footprint;
    list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
    const auto barrier = gpu::d3d12::TransitionBarrier(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &barrier);
    uploads.push_back(upload);
    return true;
}

} // namespace

int RunFsr4ReferenceProbe() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    wchar_t exe[MAX_PATH]{};
    if (!GetModuleFileNameW(nullptr, exe, MAX_PATH))
        return 1;
    const auto root = std::filesystem::path(exe).parent_path().parent_path().parent_path();
    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("M14 official-reference API preflight | adapter: %ls\n", ctx.adapter.name.c_str());
    bool available = false;
    for (const char* tree : {"FidelityFX-SDK", "FSR-4.0.2-reference"}) {
        const auto path = root / ".references/sources" / tree /
            "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll";
        HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module) {
            std::printf("%s: DLL load failed, Win32 error %lu\n", tree, GetLastError());
            continue;
        }
        struct ModuleLifetime { HMODULE module; ~ModuleLifetime() { FreeLibrary(module); } } lifetime{module};
        const auto query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
        const auto create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
        const auto destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
        if (!query || !create || !destroy) {
            std::printf("%s: public API entry points unavailable\n", tree);
            continue;
        }
        uint64_t count = 0;
        ffxQueryDescGetVersions versions{};
        versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        versions.device = ctx.device.Get();
        versions.outputCount = &count;
        auto code = query(nullptr, &versions.header);
        std::printf("%s: version query return %u, count %llu\n", tree, code,
                    static_cast<unsigned long long>(count));
        if (code != FFX_API_RETURN_OK || count == 0 || count > 32)
            continue;
        std::vector<uint64_t> ids(size_t(count), 0);
        std::vector<const char*> names(size_t(count), nullptr);
        versions.versionIds = ids.data();
        versions.versionNames = names.data();
        code = query(nullptr, &versions.header);
        if (code != FFX_API_RETURN_OK || count > ids.size()) {
            std::printf("%s: version list query failed (%u)\n", tree, code);
            continue;
        }
        for (size_t i = 0; i < size_t(count); ++i) {
            std::printf("%s: version %s id=0x%016llx\n", tree, names[i] ? names[i] : "(unnamed)",
                        static_cast<unsigned long long>(ids[i]));
        }
        ffxCreateBackendDX12Desc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device = ctx.device.Get();
        ffxCreateContextDescUpscale desc{};
        desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        desc.header.pNext = &backend.header;
        desc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                     FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
        desc.maxRenderSize = {1280, 720};
        desc.maxUpscaleSize = {1920, 1080};
        desc.fpMessage = ApiMessage;
        ffxContext context = nullptr;
        code = create(&context, &desc.header, nullptr);
        std::printf("%s: create default context return %u | context %s\n", tree, code,
                    context ? "non-null" : "null");
        if (context) {
            ffxQueryGetProviderVersion version{};
            version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
            const auto versionCode = query(&context, &version.header);
            std::printf("%s: selected version query %u | %s | id=0x%016llx\n", tree,
                        versionCode, version.versionName ? version.versionName : "(unavailable)",
                        static_cast<unsigned long long>(version.versionId));
            available |= code == FFX_API_RETURN_OK && versionCode == FFX_API_RETURN_OK;
            const auto destroyed = destroy(&context, nullptr);
            if (destroyed != FFX_API_RETURN_OK) {
                std::fprintf(stderr, "reference context destroy failed (%u)\n", destroyed);
                return 1;
            }
        }
    }
    std::string errors;
    if (!gpu::InfoQueueClean(ctx, errors)) {
        std::fprintf(stderr, "%s\n", errors.c_str());
        return 1;
    }
    std::printf("reference API preflight: %s | no frame dispatch/capture yet; M14 gate UNVERIFIED\n",
                available ? "available" : "unavailable");
    return available ? 0 : 1;
}

int RunFsr4ReferenceCapture() {
    using namespace inference::fsr4;
    setvbuf(stdout, nullptr, _IONBF, 0);
    wchar_t exe[MAX_PATH]{};
    if (!GetModuleFileNameW(nullptr, exe, MAX_PATH)) return 1;
    const auto root = std::filesystem::path(exe).parent_path().parent_path().parent_path();
    const char* inputEnv = std::getenv("M14_INPUT_DIR");
    const auto inputDirectory = inputEnv ? std::filesystem::path(inputEnv) : root / "build/reports/m13/release-final";
    const char* id = std::getenv("M14_RUN_ID");
    const std::string runId = id ? id : "official-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64());
    if (runId.empty() || runId.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos)
        return 2;
    const auto directory = root / "build/reports/m14" / runId;
    std::error_code ec;
    std::filesystem::create_directories(directory.parent_path(), ec);
    if (ec || !std::filesystem::create_directory(directory, ec) || ec) {
        std::fprintf(stderr, "official capture directory exists or cannot be created\n");
        return 1;
    }
    std::vector<TemporalFrame> frames(3);
    for (size_t i = 0; i < 3; ++i) {
        std::ifstream file(inputDirectory / ("input-frame" + std::to_string(i) + ".bin"), std::ios::binary | std::ios::ate);
        constexpr size_t bytes = sizeof(FrameConstants) + size_t(1280) * 720 * sizeof(FramePixel);
        if (!file || file.tellg() != std::streamoff(bytes)) {
            std::fprintf(stderr, "missing or incorrectly sized M13 serialized input frame %zu\n", i);
            return 1;
        }
        file.seekg(0);
        file.read(reinterpret_cast<char*>(&frames[i].constants), sizeof(FrameConstants));
        frames[i].pixels.resize(size_t(1280) * 720);
        file.read(reinterpret_cast<char*>(frames[i].pixels.data()), std::streamsize(bytes - sizeof(FrameConstants)));
        if (!file) return 1;
    }
    std::string error;
    if (!ValidateTemporalFrames(frames, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    gpu::DeviceContext ctx;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    const auto modulePath = root / ".references/sources/FidelityFX-SDK/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll";
    HMODULE module = LoadLibraryExW(modulePath.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) {
        std::fprintf(stderr, "official runtime load failed: %lu\n", GetLastError()); return 1;
    }
    struct ModuleLifetime { HMODULE m; ~ModuleLifetime() { FreeLibrary(m); } } moduleLifetime{module};
    const auto query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
    const auto create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
    const auto destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
    const auto dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
    if (!query || !create || !destroy || !dispatch) return 1;
    uint64_t count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = ctx.device.Get();
    versions.outputCount = &count;
    if (query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0 || count > 32) return 1;
    std::vector<uint64_t> ids(size_t(count), 0);
    std::vector<const char*> names(size_t(count), nullptr);
    versions.versionIds = ids.data(); versions.versionNames = names.data();
    if (query(nullptr, &versions.header) != FFX_API_RETURN_OK || count > ids.size()) return 1;
    uint64_t selected = 0;
    for (size_t i = 0; i < size_t(count); ++i)
        if (names[i] && std::strncmp(names[i], "4.1.2", 5) == 0) selected = ids[i];
    if (!selected) {
        std::fprintf(stderr, "expected explicitly versioned 4.1.2 provider unavailable; no fallback\n"); return 1;
    }
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = ctx.device.Get();
    ffxOverrideVersion overrideVersion{};
    overrideVersion.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    overrideVersion.header.pNext = &backend.header;
    overrideVersion.versionId = selected;
    ffxCreateContextDescUpscale contextDesc{};
    contextDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    contextDesc.header.pNext = &overrideVersion.header;
    contextDesc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                        FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    contextDesc.maxRenderSize = {1280, 720}; contextDesc.maxUpscaleSize = {1920, 1080};
    contextDesc.fpMessage = ApiMessage;
    ffxContext context = nullptr;
    if (create(&context, &contextDesc.header, nullptr) != FFX_API_RETURN_OK || !context) {
        std::fprintf(stderr, "official context creation failed\n"); return 1;
    }
    struct ContextLifetime {
        ffxContext& ctx; PfnFfxDestroyContext destroy;
        ~ContextLifetime() { if (ctx) destroy(&ctx, nullptr); }
    } contextLifetime{context, destroy};
    ffxQueryGetProviderVersion actual{};
    actual.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (query(&context, &actual.header) != FFX_API_RETURN_OK || actual.versionId != selected) return 1;
    std::printf("official capture: provider %s id=0x%016llx | experimental model is 4.0.2 (CROSS-VERSION)\n",
                actual.versionName, static_cast<unsigned long long>(actual.versionId));
    std::printf("output: %ls | inputs: %ls | adapter: %ls\n", directory.c_str(), inputDirectory.c_str(), ctx.adapter.name.c_str());
    D3D12_RESOURCE_DESC outDesc{};
    outDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    outDesc.Width = 1920; outDesc.Height = 1080; outDesc.DepthOrArraySize = 1;
    outDesc.MipLevels = 1; outDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    outDesc.SampleDesc.Count = 1; outDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto heap = gpu::d3d12::HeapProps(D3D12_HEAP_TYPE_DEFAULT);
    gpu::ComPtr<ID3D12Resource> output;
    if (FAILED(ctx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &outDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&output)))) return 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT outFootprint{};
    uint64_t outputBytes = 0;
    ctx.device->GetCopyableFootprints(&outDesc, 0, 1, 0, &outFootprint, nullptr, nullptr, &outputBytes);
    const auto readHeap = gpu::d3d12::HeapProps(D3D12_HEAP_TYPE_READBACK);
    const auto readDesc = gpu::d3d12::BufferDesc(outputBytes, D3D12_RESOURCE_FLAG_NONE);
    gpu::ComPtr<ID3D12Resource> readback;
    if (FAILED(ctx.device->CreateCommittedResource(&readHeap, D3D12_HEAP_FLAG_NONE, &readDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) return 1;
    for (size_t i = 0; i < 3; ++i) {
        const auto& frame = frames[i];
        gpu::ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                ctx.allocator.Get(), nullptr, IID_PPV_ARGS(&list)))) return 1;
        std::vector<float> colors(frame.pixels.size() * 4), depths(frame.pixels.size()), velocities(frame.pixels.size() * 2);
        for (size_t p = 0; p < frame.pixels.size(); ++p) {
            memcpy(colors.data() + p * 4, frame.pixels[p].color, 12);
            colors[p * 4 + 3] = 1;
            depths[p] = frame.pixels[p].depth;
            memcpy(velocities.data() + p * 2, frame.pixels[p].motion, 8);
        }
        std::vector<gpu::ComPtr<ID3D12Resource>> uploads;
        gpu::ComPtr<ID3D12Resource> color, depth, velocity, exposure;
        const float exposureValue = frame.constants.exposure;
        if (!UploadTexture(ctx, list.Get(), 1280, 720, DXGI_FORMAT_R32G32B32A32_FLOAT, 16,
                colors.data(), color, uploads, error) ||
            !UploadTexture(ctx, list.Get(), 1280, 720, DXGI_FORMAT_R32_FLOAT, 4,
                depths.data(), depth, uploads, error) ||
            !UploadTexture(ctx, list.Get(), 1280, 720, DXGI_FORMAT_R32G32_FLOAT, 8,
                velocities.data(), velocity, uploads, error) ||
            !UploadTexture(ctx, list.Get(), 1, 1, DXGI_FORMAT_R32_FLOAT, 4,
                &exposureValue, exposure, uploads, error)) {
            std::fprintf(stderr, "%s\n", error.c_str()); return 1;
        }
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = list.Get();
        d.color = ffxApiGetResourceDX12(color.Get());
        d.depth = ffxApiGetResourceDX12(depth.Get());
        d.motionVectors = ffxApiGetResourceDX12(velocity.Get());
        d.exposure = ffxApiGetResourceDX12(exposure.Get());
        d.output = ffxApiGetResourceDX12(output.Get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {frame.constants.jitter[0], frame.constants.jitter[1]};
        d.motionVectorScale = {frame.constants.mvScale[0] * 1280, frame.constants.mvScale[1] * 720};
        d.renderSize = {1280, 720}; d.upscaleSize = {1920, 1080};
        d.frameTimeDelta = 1000.0f / 60;
        d.preExposure = frame.constants.preExposure; d.reset = frame.constants.reset != 0;
        d.cameraNear = 0.1f; d.cameraFar = 1000.0f;
        d.cameraFovAngleVertical = 1.0471975512f; d.viewSpaceToMetersFactor = 1;
        const auto code = dispatch(&context, &d.header);
        if (code != FFX_API_RETURN_OK) {
            std::fprintf(stderr, "official dispatch frame%zu failed: %u\n", i, code); return 1;
        }
        auto barrier = gpu::d3d12::TransitionBarrier(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                     D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        D3D12_TEXTURE_COPY_LOCATION destination{}, origin{};
        destination.pResource = readback.Get(); destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = outFootprint;
        origin.pResource = output.Get(); origin.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        list->ResourceBarrier(1, &barrier);
        if (FAILED(list->Close())) return 1;
        ID3D12CommandList* lists[] = {list.Get()};
        ctx.queue->ExecuteCommandLists(1, lists);
        if (!gpu::GpuSync(ctx, 30000, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
        void* mapped = nullptr;
        if (FAILED(readback->Map(0, nullptr, &mapped))) return 1;
        std::vector<uint8_t> pixels(size_t(1920) * 1080 * 16);
        for (uint32_t y = 0; y < 1080; ++y)
            memcpy(pixels.data() + size_t(y) * 1920 * 16,
                   static_cast<uint8_t*>(mapped) + outFootprint.Offset + size_t(y) * outFootprint.Footprint.RowPitch,
                   size_t(1920) * 16);
        readback->Unmap(0, nullptr);
        size_t nonfinite = 0;
        float minimum = INFINITY, maximum = -INFINITY;
        for (size_t p = 0; p < size_t(1920) * 1080; ++p) {
            float rgb[3]{}; memcpy(rgb, pixels.data() + p * 16, 12);
            for (float value : rgb) {
                nonfinite += !std::isfinite(value);
                minimum = std::min(minimum, value); maximum = std::max(maximum, value);
            }
        }
        std::printf("official frame%zu: nonfinite RGB %zu | range [%g,%g]\n", i, nonfinite, minimum, maximum);
        std::ofstream file(directory / ("frame" + std::to_string(i) + ".rgba32f"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(pixels.data()), std::streamsize(pixels.size()));
        std::ofstream input(directory / ("input-frame" + std::to_string(i) + ".bin"), std::ios::binary);
        input.write(reinterpret_cast<const char*>(&frame.constants), sizeof(FrameConstants));
        input.write(reinterpret_cast<const char*>(frame.pixels.data()), std::streamsize(frame.pixels.size() * sizeof(FramePixel)));
        if (!file || !input || nonfinite != 0) return 1;
        std::string validation;
        if (!gpu::InfoQueueClean(ctx, validation)) { std::fprintf(stderr, "%s\n", validation.c_str()); return 1; }
        if (FAILED(ctx.allocator->Reset())) return 1;
    }
    std::printf("official 4.1.2 capture complete | cross-version comparison data only; M14 gate UNVERIFIED\n");
    return 0;
}

} // namespace experiments
