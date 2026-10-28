// FSR 4 v07 FasterNetBlock (model passes 4/5/7/8/10), reproduced in our own
// code as a single fused kernel per output pixel:
//   grouped 3x3 partial convolution -> 1x1 expand + ReLU -> 1x1 contract
//   -> two-scale residual add
//
// The shipped operator is a DP4A kernel on this (non-WMMA) code path and its
// arithmetic is fused per pixel, so this reproduction mirrors that structure
// instead of decomposing into separate GEMM passes. See EXP-013.
#include "weights.generated.hlsli"

#if FNB_PASS == 4
#define FNB_FEATURES 32
#define FNB_PIXELS 129600
#define FNB_WIDTH 480
#define FNB_W0 M12_P4_W0_OFFSET
#define FNB_W0_S0 M12_P4_W0_STRIDE0
#define FNB_W0_S1 M12_P4_W0_STRIDE1
#define FNB_W0_S3 M12_P4_W0_STRIDE3
#define FNB_B0 M12_P4_W1_OFFSET
#define FNB_WSCALE0 M12_P4_W0_SCALE
#define FNB_W2 M12_P4_W2_OFFSET
#define FNB_W2_S3 M12_P4_W2_STRIDE3
#define FNB_B1 M12_P4_W3_OFFSET
#define FNB_WSCALE1 M12_P4_W2_SCALE
#define FNB_W4 M12_P4_W4_OFFSET
#define FNB_W4_S3 M12_P4_W4_STRIDE3
#define FNB_B2 M12_P4_W5_OFFSET
#define FNB_WSCALE2 M12_P4_W4_SCALE
#define FNB_S0 M12_P4_ARG0
#define FNB_S1 M12_P4_ARG1
#define FNB_ACT M12_P4_ARG2
#define FNB_EPI0 M12_P4_EPI0
#define FNB_EPI1 M12_P4_EPI1
#define FNB_EPI2 M12_P4_EPI2 
#define FNB_QOUT0 M12_P4_ARG3
#define FNB_QOUT1 M12_P4_ARG4
#elif FNB_PASS == 5
#define FNB_FEATURES 32
#define FNB_PIXELS 129600
#define FNB_WIDTH 480
#define FNB_W0 M12_P5_W0_OFFSET
#define FNB_W0_S0 M12_P5_W0_STRIDE0
#define FNB_W0_S1 M12_P5_W0_STRIDE1
#define FNB_W0_S3 M12_P5_W0_STRIDE3
#define FNB_B0 M12_P5_W1_OFFSET
#define FNB_WSCALE0 M12_P5_W0_SCALE
#define FNB_W2 M12_P5_W2_OFFSET
#define FNB_W2_S3 M12_P5_W2_STRIDE3
#define FNB_B1 M12_P5_W3_OFFSET
#define FNB_WSCALE1 M12_P5_W2_SCALE
#define FNB_W4 M12_P5_W4_OFFSET
#define FNB_W4_S3 M12_P5_W4_STRIDE3
#define FNB_B2 M12_P5_W5_OFFSET
#define FNB_WSCALE2 M12_P5_W4_SCALE
#define FNB_S0 M12_P5_ARG0
#define FNB_S1 M12_P5_ARG1
#define FNB_ACT M12_P5_ARG2
// This instantiation takes a quantized output tensor, so both output scales
// come from the tensor descriptor rather than from call arguments.
#define FNB_EPI0 M12_P5_EPI0
#define FNB_EPI1 M12_P5_EPI1
#define FNB_EPI2 M12_P5_EPI2 
#define FNB_QOUT0 M12_P5_OUTPUT_SCALE
#define FNB_QOUT1 M12_P5_OUTPUT_SCALE
#elif FNB_PASS == 7
#define FNB_FEATURES 64
#define FNB_PIXELS 32400
#define FNB_WIDTH 240
#define FNB_W0 M12_P7_W0_OFFSET
#define FNB_W0_S0 M12_P7_W0_STRIDE0
#define FNB_W0_S1 M12_P7_W0_STRIDE1
#define FNB_W0_S3 M12_P7_W0_STRIDE3
#define FNB_B0 M12_P7_W1_OFFSET
#define FNB_WSCALE0 M12_P7_W0_SCALE
#define FNB_W2 M12_P7_W2_OFFSET
#define FNB_W2_S3 M12_P7_W2_STRIDE3
#define FNB_B1 M12_P7_W3_OFFSET
#define FNB_WSCALE1 M12_P7_W2_SCALE
#define FNB_W4 M12_P7_W4_OFFSET
#define FNB_W4_S3 M12_P7_W4_STRIDE3
#define FNB_B2 M12_P7_W5_OFFSET
#define FNB_WSCALE2 M12_P7_W4_SCALE
#define FNB_S0 M12_P7_ARG0
#define FNB_S1 M12_P7_ARG1
#define FNB_ACT M12_P7_ARG2
#define FNB_EPI0 M12_P7_EPI0
#define FNB_EPI1 M12_P7_EPI1
#define FNB_EPI2 M12_P7_EPI2 
#define FNB_QOUT0 M12_P7_ARG3
#define FNB_QOUT1 M12_P7_ARG4
#elif FNB_PASS == 8
#define FNB_FEATURES 64
#define FNB_PIXELS 32400
#define FNB_WIDTH 240
#define FNB_W0 M12_P8_W0_OFFSET
#define FNB_W0_S0 M12_P8_W0_STRIDE0
#define FNB_W0_S1 M12_P8_W0_STRIDE1
#define FNB_W0_S3 M12_P8_W0_STRIDE3
#define FNB_B0 M12_P8_W1_OFFSET
#define FNB_WSCALE0 M12_P8_W0_SCALE
#define FNB_W2 M12_P8_W2_OFFSET
#define FNB_W2_S3 M12_P8_W2_STRIDE3
#define FNB_B1 M12_P8_W3_OFFSET
#define FNB_WSCALE1 M12_P8_W2_SCALE
#define FNB_W4 M12_P8_W4_OFFSET
#define FNB_W4_S3 M12_P8_W4_STRIDE3
#define FNB_B2 M12_P8_W5_OFFSET
#define FNB_WSCALE2 M12_P8_W4_SCALE
#define FNB_S0 M12_P8_ARG0
#define FNB_S1 M12_P8_ARG1
#define FNB_ACT M12_P8_ARG2
#define FNB_EPI0 M12_P8_EPI0
#define FNB_EPI1 M12_P8_EPI1
#define FNB_EPI2 M12_P8_EPI2 
#define FNB_QOUT0 M12_P8_ARG3
#define FNB_QOUT1 M12_P8_ARG4
#elif FNB_PASS == 10
#define FNB_FEATURES 32
#define FNB_PIXELS 129600
#define FNB_WIDTH 480
#define FNB_W0 M12_P10_W0_OFFSET
#define FNB_W0_S0 M12_P10_W0_STRIDE0
#define FNB_W0_S1 M12_P10_W0_STRIDE1
#define FNB_W0_S3 M12_P10_W0_STRIDE3
#define FNB_B0 M12_P10_W1_OFFSET
#define FNB_WSCALE0 M12_P10_W0_SCALE
#define FNB_W2 M12_P10_W2_OFFSET
#define FNB_W2_S3 M12_P10_W2_STRIDE3
#define FNB_B1 M12_P10_W3_OFFSET
#define FNB_WSCALE1 M12_P10_W2_SCALE
#define FNB_W4 M12_P10_W4_OFFSET
#define FNB_W4_S3 M12_P10_W4_STRIDE3
#define FNB_B2 M12_P10_W5_OFFSET
#define FNB_WSCALE2 M12_P10_W4_SCALE
#define FNB_S0 M12_P10_ARG0
#define FNB_S1 M12_P10_ARG1
#define FNB_ACT M12_P10_ARG2
#define FNB_EPI0 M12_P10_EPI0
#define FNB_EPI1 M12_P10_EPI1
#define FNB_EPI2 M12_P10_EPI2 
#define FNB_QOUT0 M12_P10_ARG3
#define FNB_QOUT1 M12_P10_ARG4
#elif FNB_PASS == 9
#define FNB_FEATURES 64
#define FNB_PIXELS 32400
#define FNB_WIDTH 240
#define FNB_W0 M12_P9_W0_OFFSET
#define FNB_W0_S0 M12_P9_W0_STRIDE0
#define FNB_W0_S1 M12_P9_W0_STRIDE1
#define FNB_W0_S3 M12_P9_W0_STRIDE3
#define FNB_B0 M12_P9_W1_OFFSET
#define FNB_WSCALE0 M12_P9_W0_SCALE
#define FNB_W2 M12_P9_W2_OFFSET
#define FNB_W2_S3 M12_P9_W2_STRIDE3
#define FNB_B1 M12_P9_W3_OFFSET
#define FNB_WSCALE1 M12_P9_W2_SCALE
#define FNB_W4 M12_P9_W4_OFFSET
#define FNB_W4_S3 M12_P9_W4_STRIDE3
#define FNB_B2 M12_P9_W5_OFFSET
#define FNB_WSCALE2 M12_P9_W4_SCALE
#define FNB_S0 M12_P9_ARG0
#define FNB_S1 M12_P9_ARG1
#define FNB_ACT M12_P9_ARG2
#define FNB_EPI0 M12_P9_EPI0
#define FNB_EPI1 M12_P9_EPI1
#define FNB_EPI2 M12_P9_EPI2
#define FNB_CT2D 1
#define FNB_FNB_OUT M12_P9_ARG3
#define FNB_QOUT0 M12_P9_ARG4
#define FNB_QOUT1 M12_P9_ARG5
#define FNB_OUT_FEATURES 32
#define FNB_OUT_WIDTH 480
#define FNB_OUT_HEIGHT 270
#define FNB_CT2D_W M12_P9_W6_OFFSET
#define FNB_CT2D_S0 M12_P9_W6_STRIDE0
#define FNB_CT2D_S1 M12_P9_W6_STRIDE1
#define FNB_CT2D_S2 M12_P9_W6_STRIDE2
#define FNB_CT2D_S3 M12_P9_W6_STRIDE3
#define FNB_CT2D_B M12_P9_W7_OFFSET
#define FNB_CT2D_SCALE M12_P9_CT2D_SCALE
#define FNB_SKIP_SCALE M12_P9_SKIP_SCALE
#elif FNB_PASS == 11
#define FNB_FEATURES 32
#define FNB_PIXELS 129600
#define FNB_WIDTH 480
#define FNB_W0 M12_P11_W0_OFFSET
#define FNB_W0_S0 M12_P11_W0_STRIDE0
#define FNB_W0_S1 M12_P11_W0_STRIDE1
#define FNB_W0_S3 M12_P11_W0_STRIDE3
#define FNB_B0 M12_P11_W1_OFFSET
#define FNB_WSCALE0 M12_P11_W0_SCALE
#define FNB_W2 M12_P11_W2_OFFSET
#define FNB_W2_S3 M12_P11_W2_STRIDE3
#define FNB_B1 M12_P11_W3_OFFSET
#define FNB_WSCALE1 M12_P11_W2_SCALE
#define FNB_W4 M12_P11_W4_OFFSET
#define FNB_W4_S3 M12_P11_W4_STRIDE3
#define FNB_B2 M12_P11_W5_OFFSET
#define FNB_WSCALE2 M12_P11_W4_SCALE
#define FNB_S0 M12_P11_ARG0
#define FNB_S1 M12_P11_ARG1
#define FNB_ACT M12_P11_ARG2
#define FNB_EPI0 M12_P11_EPI0
#define FNB_EPI1 M12_P11_EPI1
#define FNB_EPI2 M12_P11_EPI2
#define FNB_CT2D 1
#define FNB_FNB_OUT M12_P11_ARG3
// This instantiation takes a quantized output tensor: one scale for both halves.
#define FNB_QOUT0 M12_P11_OUTPUT_SCALE
#define FNB_QOUT1 M12_P11_OUTPUT_SCALE
#define FNB_OUT_FEATURES 16
#define FNB_OUT_WIDTH 960
#define FNB_OUT_HEIGHT 540
#define FNB_CT2D_W M12_P11_W6_OFFSET
#define FNB_CT2D_S0 M12_P11_W6_STRIDE0
#define FNB_CT2D_S1 M12_P11_W6_STRIDE1
#define FNB_CT2D_S2 M12_P11_W6_STRIDE2
#define FNB_CT2D_S3 M12_P11_W6_STRIDE3
#define FNB_CT2D_B M12_P11_W7_OFFSET
#define FNB_CT2D_SCALE M12_P11_CT2D_SCALE
#define FNB_SKIP_SCALE M12_P11_SKIP_SCALE
#else
#error "FNB_PASS must be 4, 5, 7, 8, 9, 10 or 11"
#endif

#define FNB_N FNB_FEATURES
#define FNB_H (FNB_N / 2)

ByteAddressBuffer src : register(t0);
ByteAddressBuffer w : register(t1);
RWByteAddressBuffer dst : register(u0);

// Signed byte `lane` of a packed dword.
int Lane(uint packed, uint lane) {
    return int(packed << (24u - lane * 8u)) >> 24;
}

// Four consecutive FP16 values; bias offsets are multiples of 8 bytes.
float4 Half4(ByteAddressBuffer data, uint offset) {
    const uint lo = data.Load(offset);
    const uint hi = data.Load(offset + 4);
    return float4(f16tof32(lo & 0xFFFFu), f16tof32(lo >> 16),
                  f16tof32(hi & 0xFFFFu), f16tof32(hi >> 16));
}

uint4 Dword4(ByteAddressBuffer data, uint offset) {
    return data.Load<uint4>(offset);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint pixel = tid.x;
    if (pixel >= FNB_PIXELS)
        return;
    const uint ox = pixel % FNB_WIDTH;
    const uint oy = pixel / FNB_WIDTH;
    const uint base = pixel * FNB_N;
    const int height = int(FNB_PIXELS / FNB_WIDTH);

    // First half of the pixel (channels 0..H-1) and the second half (H..N-1).
    // They are both the residual skip of their feature half and, for the
    // second half, the copied part of the concatenated activation.
    uint pre[FNB_H / 4];
    uint sec[FNB_H / 4];
    [unroll]
    for (uint d = 0; d < FNB_H / 4; ++d) {
        pre[d] = src.Load(base + d * 4);
        sec[d] = src.Load(base + FNB_H + d * 4);
    }

    uint convResult[FNB_N / 4];
    [unroll]
    for (uint d = 0; d < FNB_H / 4; ++d)
        convResult[FNB_H / 4 + d] = sec[d];

    ////////////////////////////////// First convolution //////////////////////////////////
    int convAcc[FNB_H];
    [unroll]
    for (uint z = 0; z < FNB_H; ++z)
        convAcc[z] = 0;
    [unroll]
    for (uint ky = 0; ky < 3; ++ky) {
        [unroll]
        for (uint kx = 0; kx < 3; ++kx) {
            const int ix = int(ox) + int(kx) - 1;
            const int iy = int(oy) + int(ky) - 1;
            if (ix < 0 || iy < 0 || ix >= int(FNB_WIDTH) || iy >= height)
                continue;
            const uint tapBase = (uint(iy) * FNB_WIDTH + uint(ix)) * FNB_N;
            uint vs[8];
            [unroll]
            for (uint v = 0; v < 8; v += 4) {
                const uint4 q = src.Load<uint4>(tapBase + v * 4);
                vs[v] = q.x; vs[v + 1] = q.y; vs[v + 2] = q.z; vs[v + 3] = q.w;
            }

            [unroll]
            for (uint f = 0; f < FNB_H; ++f) {
#if FNB_N == 64
                // Two 16-channel groups: features 0..15 read the first group,
                // 16..31 the second.
                const uint group = f >= 16 ? 4 : 0;
#else
                const uint group = 0;
#endif
                const uint4 wd =
                    Dword4(w, FNB_W0 + kx * FNB_W0_S0 + ky * FNB_W0_S1 + f * FNB_W0_S3);
                convAcc[f] = dot4add_i8packed(vs[group + 0], wd.x, convAcc[f]);
                convAcc[f] = dot4add_i8packed(vs[group + 1], wd.y, convAcc[f]);
                convAcc[f] = dot4add_i8packed(vs[group + 2], wd.z, convAcc[f]);
                convAcc[f] = dot4add_i8packed(vs[group + 3], wd.w, convAcc[f]);
            }
        }
    }

    [unroll]
    for (uint f = 0; f < FNB_H; f += 4) {
        const float4 b = Half4(w, FNB_B0 + f * 2);
        const int4 acc = int4(convAcc[f], convAcc[f + 1], convAcc[f + 2], convAcc[f + 3]);
        const int4 q = int4(round(mad(float4(acc), FNB_EPI0, b * (1.0 / FNB_S1))));
        convResult[f / 4] = pack_clamp_s8(q);
    }

    ////////////////////////////// Second convolution + ReLU //////////////////////////////
    int reluOut[FNB_N * 2];
    [unroll]
    for (uint z = 0; z < FNB_N * 2; ++z)
        reluOut[z] = 0;
    [unroll]
    for (uint i = 0; i < FNB_N * 2; ++i) {
        [unroll]
        for (uint j = 0; j < FNB_N; j += 16) {
            const uint4 wd = Dword4(w, FNB_W2 + j + i * FNB_W2_S3);
            reluOut[i] = dot4add_i8packed(convResult[j / 4 + 0], wd.x, reluOut[i]);
            reluOut[i] = dot4add_i8packed(convResult[j / 4 + 1], wd.y, reluOut[i]);
            reluOut[i] = dot4add_i8packed(convResult[j / 4 + 2], wd.z, reluOut[i]);
            reluOut[i] = dot4add_i8packed(convResult[j / 4 + 3], wd.w, reluOut[i]);
        }
    }

    uint reluResult[FNB_N * 2 / 4];
    [unroll]
    for (uint i = 0; i < FNB_N * 2; i += 4) {
        const float4 b = Half4(w, FNB_B1 + i * 2);
        const int4 v = int4(reluOut[i], reluOut[i + 1], reluOut[i + 2], reluOut[i + 3]);
        const int4 q = int4(round(mad(float4(v), FNB_EPI1, b * (1.0 / FNB_ACT))));
        reluResult[i / 4] = pack_clamp_s8(clamp(q, int4(0, 0, 0, 0), int4(127, 127, 127, 127)));
    }

    ////////////////////////////////// Third convolution //////////////////////////////////
    int conv3[FNB_N];
    [unroll]
    for (uint z = 0; z < FNB_N; ++z)
        conv3[z] = 0;
    [unroll]
    for (uint i = 0; i < FNB_N; ++i) {
        [unroll]
        for (uint j = 0; j < FNB_N * 2; j += 16) {
            const uint4 wd = Dword4(w, FNB_W4 + j + i * FNB_W4_S3);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 0], wd.x, conv3[i]);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 1], wd.y, conv3[i]);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 2], wd.z, conv3[i]);
            conv3[i] = dot4add_i8packed(reluResult[j / 4 + 3], wd.w, conv3[i]);
        }
    }

    ////////////////////////////////// Add + store //////////////////////////////////
#if FNB_CT2D
    // The block output (with its two-scale residual) is requantized into the
    // packed activation the transposed convolution consumes, then upscaled 2x
    // and added to the skip tensor at the upscaled resolution.
    uint packedVss[FNB_N / 4];
    [unroll]
    for (uint i = 0; i < FNB_N; i += 4) {
        const float4 b = Half4(w, FNB_B2 + i * 2);
        const int4 v = int4(conv3[i], conv3[i + 1], conv3[i + 2], conv3[i + 3]);
        float4 result = mad(float4(v), FNB_EPI2, b);
        float4 skip;
        if (i < FNB_H) {
            const uint packed = pre[i / 4];
            skip = float4(Lane(packed, 0), Lane(packed, 1), Lane(packed, 2), Lane(packed, 3));
            result += skip * FNB_S0;
        } else {
            const uint packed = sec[(i - FNB_H) / 4];
            skip = float4(Lane(packed, 0), Lane(packed, 1), Lane(packed, 2), Lane(packed, 3));
            result += skip * FNB_S1;
        }
        packedVss[i / 4] = pack_clamp_s8(int4(round(result * (1.0 / FNB_FNB_OUT))));
    }

    const uint skipBase = FNB_PIXELS * FNB_N;
    [unroll]
    for (uint phase = 0; phase < 4; ++phase) {
        const uint px = phase & 1u;
        const uint py = phase >> 1u;
        const uint ox2 = ox * 2 + px;
        const uint oy2 = oy * 2 + py;
        // The reproduction predicates to the logical output extent. Pass 11's
        // shipped 32/1 overload does not, so it must be compared against the
        // labelled predication control (EXP-010).
        if (ox2 >= FNB_OUT_WIDTH || oy2 >= FNB_OUT_HEIGHT)
            continue;
        const uint outPixel = oy2 * FNB_OUT_WIDTH + ox2;
        [unroll]
        for (uint f = 0; f < FNB_OUT_FEATURES; f += 4) {
            int accum[4] = {0, 0, 0, 0};
            [unroll]
            for (uint lane = 0; lane < 4; ++lane) {
                const uint woff = FNB_CT2D_W + px * FNB_CT2D_S0 + py * FNB_CT2D_S1 +
                                  (f + lane) * FNB_CT2D_S2;
                [unroll]
                for (uint c = 0; c < FNB_N; c += 16) {
                    const uint4 wd = w.Load<uint4>(woff + c * FNB_CT2D_S3);
                    accum[lane] = dot4add_i8packed(wd.x, packedVss[c / 4 + 0], accum[lane]);
                    accum[lane] = dot4add_i8packed(wd.y, packedVss[c / 4 + 1], accum[lane]);
                    accum[lane] = dot4add_i8packed(wd.z, packedVss[c / 4 + 2], accum[lane]);
                    accum[lane] = dot4add_i8packed(wd.w, packedVss[c / 4 + 3], accum[lane]);
                }
            }
            const float4 b = Half4(w, FNB_CT2D_B + f * 2);
            float4 result = mad(float4(accum[0], accum[1], accum[2], accum[3]),
                                FNB_CT2D_SCALE, b);
            const uint skipAt = skipBase + outPixel * FNB_OUT_FEATURES + f;
            uint4 sk = src.Load<uint4>(skipAt);
            result += float4(Lane(sk.x, 0), Lane(sk.x, 1), Lane(sk.x, 2), Lane(sk.x, 3)) *
                      FNB_SKIP_SCALE;
            const float rcpOut = f < FNB_OUT_FEATURES / 2 ? (1.0 / FNB_QOUT0)
                                                         : (1.0 / FNB_QOUT1);
            dst.Store(outPixel * FNB_OUT_FEATURES + f, pack_clamp_s8(int4(round(result * rcpOut))));
        }
    }
#else
    [unroll]
    for (uint i = 0; i < FNB_N; i += 4) {
        const float4 b = Half4(w, FNB_B2 + i * 2);
        const int4 v = int4(conv3[i], conv3[i + 1], conv3[i + 2], conv3[i + 3]);
        float4 result = mad(float4(v), FNB_EPI2, b);
        float4 skip;
        float4 q;
        if (i < FNB_H) {
            const uint packed = pre[i / 4];
            skip = float4(Lane(packed, 0), Lane(packed, 1), Lane(packed, 2), Lane(packed, 3));
            result += skip * FNB_S0;
            q = round(result * (1.0 / FNB_QOUT0));
        } else {
            const uint packed = sec[(i - FNB_H) / 4];
            skip = float4(Lane(packed, 0), Lane(packed, 1), Lane(packed, 2), Lane(packed, 3));
            result += skip * FNB_S1;
            q = round(result * (1.0 / FNB_QOUT1));
        }
        dst.Store(base + i, pack_clamp_s8(int4(q)));
    }
#endif
}






