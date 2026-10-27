#pragma once

#include <array>
#include <cstdint>

namespace reference {

// M11's fixed batch of 16 vectors and two 16x16 signed weight matrices.
// Weight layout is K-major [input channel][output channel], matching the WMMA
// GEMM B layout. Quantization is symmetric, zero point zero, with a 1/16
// accumulator-to-output factor at each layer.
constexpr unsigned FixedNetworkWidth = 16;
constexpr unsigned FixedNetworkElements = 16 * 16;
using FixedNetworkBytes = std::array<int8_t, FixedNetworkElements>;
using FixedNetworkAccumulators = std::array<int64_t, FixedNetworkElements>;

struct FixedNetworkReference {
    FixedNetworkAccumulators firstAcc{};
    FixedNetworkBytes hidden{};
    FixedNetworkAccumulators secondAcc{};
    FixedNetworkBytes output{};
};

// Independent INT64 dot-product oracle. Requantization is exact integer
// nearest-even division by 16, followed by INT8 saturation. Hidden output
// additionally clamps negative values to zero. No floating-point dependence.
FixedNetworkReference CpuFixedNetwork(const FixedNetworkBytes& input,
                                      const FixedNetworkBytes& firstWeights,
                                      const FixedNetworkBytes& secondWeights);

} // namespace reference
