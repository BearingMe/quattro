#include "experiments.h"

#include "../gpu/compute.h"
#include "../gpu/device.h"
#include "../reference/reference.h"

#include <cstdio>
#include <string>
#include <vector>

namespace experiments {
namespace {

void PrintAdapter(const gpu::AdapterInfo& adapter) {
    std::printf("adapter: %ls (vid=0x%04X did=0x%04X, %zu MiB, luid 0x%llX)\n",
                adapter.name.c_str(), adapter.vendorId, adapter.deviceId,
                adapter.dedicatedVideoMemory >> 20,
                (unsigned long long)adapter.luidHighLow);
}

} // namespace

int RunComputeHarness() {
    // Deterministic input covering negative, positive and fractional values.
    const size_t count = 1u << 20;
    std::vector<float> a(count), b(count), expected(count);
    for (size_t i = 0; i < count; ++i) {
        a[i] = float(i % 1024) * 0.25f - 128.0f;      // [-128, 127.75]
        b[i] = float((i * 7u) % 511u) * 0.5f - 100.0f; // [-100, 154.5]
    }
    reference::CpuVectorAdd(a.data(), b.data(), expected.data(), count);

    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    PrintAdapter(ctx.adapter);

    gpu::VectorAddOutputs outputs;
    gpu::ComputeRunResult timing;
    if (!gpu::RunVectorAdd(ctx, count, /*warmup=*/10, /*timed=*/100,
                           a.data(), b.data(), outputs, timing, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    // Compare both snapshots against the independent CPU reference.
    const size_t mismatchWarmup =
        reference::CountMismatches(outputs.postWarmup.data(), expected.data(), count);
    const size_t mismatchTimed =
        reference::CountMismatches(outputs.postTimed.data(), expected.data(), count);

    std::printf("workload: float32 vector add, %zu elements (%zu KiB/field)\n",
                count, count * sizeof(float) >> 10);
    std::printf("dispatches: 10 warmup + %u timed (%u groups x 256 threads)\n",
                timing.iterations, (unsigned)((count + 255) / 256));
    std::printf("gpu time: min %.3f us, avg %.3f us, max %.3f us\n",
                timing.minMs * 1000.0, timing.avgMs * 1000.0, timing.maxMs * 1000.0);
    std::printf("correctness: post-warmup mismatches %zu, post-timed mismatches %zu (bit-exact expected)\n",
                mismatchWarmup, mismatchTimed);

    bool ok = (mismatchWarmup == 0) && (mismatchTimed == 0);

    // Debug diagnostics: no error-severity messages may remain (gate item).
    std::string queueErrors;
    if (!gpu::InfoQueueClean(ctx, queueErrors)) {
        ok = false;
        std::fprintf(stderr, "%s", queueErrors.c_str());
    } else {
        std::printf("diagnostics: debug layer + GPU-based validation enabled, no error-severity messages\n");
    }

    std::printf("%s\n", ok ? "compute harness: PASS" : "compute harness: FAIL");
    return ok ? 0 : 1;
}

} // namespace experiments
