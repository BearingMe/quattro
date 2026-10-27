#include "fixed_network.h"

#include <algorithm>

namespace reference {
namespace {
int8_t Quantize(int64_t acc, bool relu) {
    // The fixed dot products have only 16 terms, so negating acc is safe.
    const bool negative = acc < 0;
    const uint64_t magnitude = uint64_t(negative ? -acc : acc);
    uint64_t q = magnitude / 16;
    const uint64_t remainder = magnitude % 16;
    if (remainder > 8 || (remainder == 8 && (q & 1))) ++q;
    const int64_t signedResult = negative ? -int64_t(q) : int64_t(q);
    return int8_t(std::clamp<int64_t>(signedResult, relu ? 0 : -128, 127));
}

void Layer(const FixedNetworkBytes& input, const FixedNetworkBytes& weights,
           FixedNetworkAccumulators& acc, FixedNetworkBytes& output, bool relu) {
    for (unsigned row = 0; row < FixedNetworkWidth; ++row) {
        for (unsigned column = 0; column < FixedNetworkWidth; ++column) {
            int64_t sum = 0;
            for (unsigned k = 0; k < FixedNetworkWidth; ++k)
                sum += int64_t(input[row * FixedNetworkWidth + k]) *
                       int64_t(weights[k * FixedNetworkWidth + column]);
            const unsigned index = row * FixedNetworkWidth + column;
            acc[index] = sum;
            output[index] = Quantize(sum, relu);
        }
    }
}
}

FixedNetworkReference CpuFixedNetwork(const FixedNetworkBytes& input,
                                      const FixedNetworkBytes& firstWeights,
                                      const FixedNetworkBytes& secondWeights) {
    FixedNetworkReference result;
    Layer(input, firstWeights, result.firstAcc, result.hidden, true);
    Layer(result.hidden, secondWeights, result.secondAcc, result.output, false);
    return result;
}
} // namespace reference
