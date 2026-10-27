#include "experiments.h"
#include "reference/fixed_network.h"

#include <cstdio>

namespace experiments {
int RunFixedNetworkReference() {
    using namespace reference;
    FixedNetworkBytes input{}, first{}, second{};
    for (unsigned r = 0; r < FixedNetworkWidth; ++r)
        for (unsigned c = 0; c < FixedNetworkWidth; ++c)
            input[r * FixedNetworkWidth + c] = int8_t(int(r) * 8 + int(c) - 64);
    for (unsigned c = 0; c < FixedNetworkWidth; ++c) {
        first[c * FixedNetworkWidth + c] = 16;
        second[c * FixedNetworkWidth + (FixedNetworkWidth - 1 - c)] = 16;
    }
    const auto result = CpuFixedNetwork(input, first, second);
    bool ok = true;
    for (unsigned r = 0; r < FixedNetworkWidth; ++r) {
        for (unsigned c = 0; c < FixedNetworkWidth; ++c) {
            const unsigned i = r * FixedNetworkWidth + c;
            const int in = input[i];
            const int hidden = in > 0 ? in : 0;
            const int reversed = input[r * FixedNetworkWidth + 15 - c];
            const int output = reversed > 0 ? reversed : 0;
            if (result.firstAcc[i] != in * 16 || result.hidden[i] != hidden ||
                result.secondAcc[i] != output * 16 || result.output[i] != output)
                ok = false;
        }
    }
    // Literal ties-to-even and saturation anchors, independent of the loop's
    // diagonal/identity expectation above.
    input.fill(0); first.fill(0); second.fill(0);
    input[0] = 1;
    first[0] = 8; first[1] = 24; first[2] = 40;
    first[3] = -8; first[4] = 127;
    second[0] = 16; second[17] = 16; second[34] = 16;
    const auto ties = CpuFixedNetwork(input, first, second);
    if (ties.hidden[0] != 0 || ties.hidden[1] != 2 || ties.hidden[2] != 2 ||
        ties.hidden[3] != 0 || ties.hidden[4] != 8 ||
        ties.output[0] != 0 || ties.output[1] != 2 || ties.output[2] != 2) ok = false;
    input.fill(127); first.fill(127); second.fill(-128);
    const auto saturation = CpuFixedNetwork(input, first, second);
    if (saturation.firstAcc[0] != 16LL * 127 * 127 || saturation.hidden[0] != 127 ||
        saturation.secondAcc[0] != 16LL * 127 * -128 || saturation.output[0] != -128) ok = false;
    std::printf("M11 CPU-only fixed-network oracle: PASS (full gate: run fixed-network)\n");
    return ok ? 0 : 1;
}
} // namespace experiments
