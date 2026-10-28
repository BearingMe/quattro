// FSR 4 v07 CNB_CT2D (model pass 13), reproduced in our own code as one fused
// kernel per input pixel: ConvNextBlock (3x3 depthwise partial conv, 1x1
// expand + ReLU, 1x1 contract, residual add) followed by a 2x2 transposed
// upscale that writes FP16 output. See EXP-014.
#include "weights.generated.hlsli"

#define IN_PIXELS 518400
#define IN_WIDTH 960
#define IN_HEIGHT 540
#define IN_CHANNELS 16
#define OUT_FEATURES 8
#define OUT_WIDTH 1920
#define OUT_HEIGHT 1080

ByteAddressBuffer src : register(t0);
ByteAddressBuffer w : register(t1);
RWByteAddressBuffer dst : register(u0);

int Lane(uint packed, uint lane) {
    return int(packed << (24u - lane * 8u)) >> 24;
}

float4 Half4(ByteAddressBuffer data, uint offset) {
    const uint lo = data.Load(offset);
    const uint hi = data.Load(offset + 4);
    return float4(f16tof32(lo & 0xFFFFu), f16tof32(lo >> 16),
                  f16tof32(hi & 0xFFFFu), f16tof32(hi >> 16));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint pixel = tid.x;
    if (pixel >= IN_PIXELS)
        return;
    const uint ox = pixel % IN_WIDTH;
    const uint oy = pixel / IN_WIDTH;
    const uint base = pixel * IN_CHANNELS;
    const uint4 inputDwords = src.Load<uint4>(base);

    ////////////////////////////////// 3x3 depthwise partial convolution /////////////////////////////////
    int convAcc[IN_CHANNELS];
    [unroll]
    for (uint z = 0; z < IN_CHANNELS; ++z)
        convAcc[z] = 0;
    [unroll]
    for (uint ky = 0; ky < 3; ++ky) {
        [unroll]
        for (uint kx = 0; kx < 3; ++kx) {
            const int ix = int(ox) + int(kx) - 1;
            const int iy = int(oy) + int(ky) - 1;
            if (ix < 0 || iy < 0 || ix >= IN_WIDTH || iy >= IN_HEIGHT)
                continue;
            const uint4 vs = src.Load<uint4>((uint(iy) * IN_WIDTH + uint(ix)) * IN_CHANNELS);
            [unroll]
            for (uint f = 0; f < IN_CHANNELS; ++f) {
                const uint4 wd = w.Load<uint4>(M12_P13_W0_OFFSET + kx * M12_P13_W0_STRIDE0 +
                                               ky * M12_P13_W0_STRIDE1 + f * M12_P13_W0_STRIDE3);
                convAcc[f] = dot4add_i8packed(vs.x, wd.x, convAcc[f]);
                convAcc[f] = dot4add_i8packed(vs.y, wd.y, convAcc[f]);
                convAcc[f] = dot4add_i8packed(vs.z, wd.z, convAcc[f]);
                convAcc[f] = dot4add_i8packed(vs.w, wd.w, convAcc[f]);
            }
        }
    }

    uint convResult[IN_CHANNELS / 4];
    const float convInputScale = M12_P13_INPUT_SCALE * M12_P13_W0_SCALE;
    [unroll]
    for (uint f = 0; f < IN_CHANNELS; f += 4) {
        const float4 b = Half4(w, M12_P13_W1_OFFSET + f * 2);
        const int4 acc = int4(convAcc[f], convAcc[f + 1], convAcc[f + 2], convAcc[f + 3]);
        convResult[f / 4] =
            pack_clamp_s8(int4(round(mad(float4(acc), convInputScale, b) * M12_P13_ARG0)));
    }

    ////////////////////////////////// 1x1 expand + ReLU /////////////////////////////////
    const uint expandChannels = 32;
    int reluOut[expandChannels];
    [unroll]
    for (uint i = 0; i < expandChannels; ++i)
        reluOut[i] = 0;
    [unroll]
    for (uint i = 0; i < expandChannels; ++i) {
        const uint4 wd = w.Load<uint4>(M12_P13_W2_OFFSET + i * M12_P13_W2_STRIDE3);
        reluOut[i] = dot4add_i8packed(convResult[0], wd.x, reluOut[i]);
        reluOut[i] = dot4add_i8packed(convResult[1], wd.y, reluOut[i]);
        reluOut[i] = dot4add_i8packed(convResult[2], wd.z, reluOut[i]);
        reluOut[i] = dot4add_i8packed(convResult[3], wd.w, reluOut[i]);
    }

    uint reluResult[expandChannels / 4];
    const float expandInputScale = M12_P13_ARG1 * M12_P13_W2_SCALE;
    [unroll]
    for (uint i = 0; i < expandChannels; i += 4) {
        const float4 b = Half4(w, M12_P13_W3_OFFSET + i * 2);
        const int4 v = int4(reluOut[i], reluOut[i + 1], reluOut[i + 2], reluOut[i + 3]);
        const int4 q = int4(round(mad(float4(v), expandInputScale, b) * M12_P13_ARG2));
        reluResult[i / 4] = pack_clamp_s8(clamp(q, int4(0, 0, 0, 0), int4(127, 127, 127, 127)));
    }

    ////////////////////////////////// 1x1 contract + residual /////////////////////////////////
    int conv3[IN_CHANNELS];
    [unroll]
    for (uint i = 0; i < IN_CHANNELS; ++i)
        conv3[i] = 0;
    [unroll]
    for (uint i = 0; i < IN_CHANNELS; ++i) {
        [unroll]
        for (uint j = 0; j < expandChannels; j += 16) {
            const uint4 wd = w.Load<uint4>(M12_P13_W4_OFFSET + j + i * M12_P13_W4_STRIDE3);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 0], wd.x, conv3[i]);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 1], wd.y, conv3[i]);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 2], wd.z, conv3[i]);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 3], wd.w, conv3[i]);
        }
    }

    uint packedVss[IN_CHANNELS / 4];
    const float contractScale = M12_P13_W4_SCALE * M12_P13_ARG3;
    [unroll]
    for (uint i = 0; i < IN_CHANNELS; i += 4) {
        const float4 b = Half4(w, M12_P13_W5_OFFSET + i * 2);
        const int4 v = int4(conv3[i], conv3[i + 1], conv3[i + 2], conv3[i + 3]);
        float4 result = mad(float4(v), contractScale, b);
        const uint packed = inputDwords[i / 4];
        result += float4(Lane(packed, 0), Lane(packed, 1), Lane(packed, 2), Lane(packed, 3)) *
                  M12_P13_INPUT_SCALE;
        packedVss[i / 4] =
            pack_clamp_s8(int4(round(result * (1.0 / M12_P13_ARG4))));
    }

    ////////////////////////////////// 2x2 transposed upscale -> FP16 output /////////////////////////////////
    [unroll]
    for (uint phase = 0; phase < 4; ++phase) {
        const uint px = phase & 1u;
        const uint py = phase >> 1u;
        const uint ox2 = ox * 2 + px;
        const uint oy2 = oy * 2 + py;
        // The reproduction predicates to the logical extent; the shipped store
        // is unpredicated and writes 8 rows past 1080 (EXP-010).
        if (ox2 >= OUT_WIDTH || oy2 >= OUT_HEIGHT)
            continue;
        const uint outPixel = oy2 * OUT_WIDTH + ox2;
        [unroll]
        for (uint group = 0; group < OUT_FEATURES; group += 4) {
            int accum[4] = {0, 0, 0, 0};
            [unroll]
            for (uint lane = 0; lane < 4; ++lane) {
                const uint woff = M12_P13_W6_OFFSET + px * M12_P13_W6_STRIDE0 +
                                  py * M12_P13_W6_STRIDE1 +
                                  (group + lane) * M12_P13_W6_STRIDE2;
                [unroll]
                for (uint c = 0; c < IN_CHANNELS; c += 16) {
                    const uint4 wd = w.Load<uint4>(woff + c * M12_P13_W6_STRIDE3);
                    accum[lane] = dot4add_i8packed(wd.x, packedVss[c / 4 + 0], accum[lane]);
                    accum[lane] = dot4add_i8packed(wd.y, packedVss[c / 4 + 1], accum[lane]);
                    accum[lane] = dot4add_i8packed(wd.z, packedVss[c / 4 + 2], accum[lane]);
                    accum[lane] = dot4add_i8packed(wd.w, packedVss[c / 4 + 3], accum[lane]);
                }
            }
            // The shipped operator narrows the scaled accumulator to FP16 and
            // then adds the bias in FP16 arithmetic.
            const float4 b = Half4(w, M12_P13_W7_OFFSET + group * 2);
            const float4 scaled = float4(accum[0], accum[1], accum[2], accum[3]) *
                                  M12_P13_CT2D_SCALE;
            const half4 value = half4(scaled) + half4(b.x, b.y, b.z, b.w);
            const uint lo = (uint)f32tof16(float(value.x)) |
                            ((uint)f32tof16(float(value.y)) << 16);
            const uint hi = (uint)f32tof16(float(value.z)) |
                            ((uint)f32tof16(float(value.w)) << 16);
            const uint outOffset = outPixel * OUT_FEATURES * 2 + group * 2;
            dst.Store(outOffset, lo);
            dst.Store(outOffset + 4, hi);
        }
    }
}
