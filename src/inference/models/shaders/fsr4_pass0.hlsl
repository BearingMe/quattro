// Our v07 input-convolution implementation. The numeric contract is FP16
// operands, FP32 dot2add accumulation, FP16 bias, RNE -> signed INT8. Addresses
// come from checked extraction; no AMD operator/runtime HLSL is included.
#include "weights.generated.hlsli"

ByteAddressBuffer features : register(t0);
ByteAddressBuffer weights : register(t1);
RWByteAddressBuffer output : register(u0);

float16_t2 HalfPair(ByteAddressBuffer data, uint address) {
    uint word = data.Load(address);
    return float16_t2(f16tof32(word & 65535u), f16tof32(word >> 16));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint pixel = tid.x;
    if (pixel >= 960u * 540u)
        return;
    const uint x = pixel % 960u;
    const uint y = pixel / 960u;
    // Read the four FP16x8 samples once. Channel 7 has zero weight in the
    // source payload; input fixtures initialize this padding channel to zero.
    float16_t2 samples[16];
    [unroll]
    for (uint ky = 0; ky < 2; ++ky)
        [unroll]
        for (uint kx = 0; kx < 2; ++kx)
            [unroll]
            for (uint pair = 0; pair < 4; ++pair)
                samples[(ky * 2 + kx) * 4 + pair] =
                    HalfPair(features, ((y * 2 + ky) * 1920 + x * 2 + kx) * 16 + pair * 4);

    [unroll]
    for (uint group = 0; group < 4; ++group) {
        int4 quantized;
        [unroll]
        for (uint lane = 0; lane < 4; ++lane) {
            const uint channel = group * 4 + lane;
            float sum = 0.0f;
            [unroll]
            for (uint sample = 0; sample < 16; ++sample) {
                const uint address = M12_P0_W0_OFFSET + channel * 64 + sample * 4;
                sum = dot2add(HalfPair(weights, address), samples[sample], sum);
            }
            const uint biasWord = weights.Load(M12_P0_W1_OFFSET + (channel & ~1u) * 2);
            const float bias = f16tof32((biasWord >> ((channel & 1u) * 16)) & 65535u);
            // Keep the source's int16 narrowing before the saturating INT8 pack.
            quantized[lane] = int16_t(round((sum + bias) * (1.0f / 0.014884727075695992f)));
        }
        output.Store(pixel * 16 + group * 4, pack_clamp_s8(quantized));
    }
}
