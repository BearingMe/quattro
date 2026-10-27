// reference — CPU implementations for GPU validation. Must never touch GPU
// APIs (docs/architecture.md §3).

#pragma once

#include <cstddef>

namespace reference {

// Independent CPU reference for the M05 harness workload. Bit-exactness is
// expected: both sides perform the same fp32 addition per element in the same
// order — no reassociation anywhere.
void CpuVectorAdd(const float* a, const float* b, float* out, size_t count);

// Returns the number of elements where `gpu` and `cpu` differ bit-wise.
size_t CountMismatches(const float* gpu, const float* cpu, size_t count);

} // namespace reference
