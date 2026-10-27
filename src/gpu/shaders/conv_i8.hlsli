// Generic fixed-shape INT8 convolution packing + numeric epilogue. WMMA itself
// reuses gemm_i8.hlsl. No FSR model names or graph semantics belong here.
ByteAddressBuffer source : register(t0);
ByteAddressBuffer weights : register(t1);
RWByteAddressBuffer target : register(u0);
RWByteAddressBuffer residual : register(u1);

uint ByteAt(ByteAddressBuffer data, uint offset) {
    return (data.Load(offset & ~3u) >> ((offset & 3u) * 8)) & 255u;
}

// Dispatch (4096, ceil(dwords/(4096*64)), 1). Each invocation owns a complete
// dword, so byte packing cannot race neighboring elements.
[numthreads(64, 1, 1)]
void pack(uint3 tid : SV_DispatchThreadID) {
    const uint index = (tid.y * 4096u * 64u + tid.x) * 4;
    const uint aBytes = CONV_M * CONV_K;
    const uint bBytes = CONV_K * CONV_N;
    if (index >= aBytes + bBytes + 1024u)
        return;
    uint word = 0;
    [unroll]
    for (uint lane = 0; lane < 4; ++lane) {
        uint byte = 0;
        const uint position = index + lane;
        if (position < aBytes) {
            const uint pixel = position / CONV_K;
            const uint reduction = position % CONV_K;
            const uint channel = reduction % CONV_C;
            const uint tap = reduction / CONV_C;
            const int x = int(pixel % CONV_W) * CONV_STEP + int(tap % CONV_KERNEL) - CONV_PAD;
            const int y = int(pixel / CONV_W) * CONV_STEP + int(tap / CONV_KERNEL) - CONV_PAD;
            if (x >= 0 && y >= 0 && x < CONV_IW && y < CONV_IH)
                byte = ByteAt(source, (uint(y) * CONV_IW + uint(x)) * CONV_IC + channel);
        } else if (position < aBytes + bBytes) {
            const uint offset = position - aBytes;
            const uint channel = offset % CONV_N;
            const uint reduction = offset / CONV_N;
            const uint inputChannel = reduction % CONV_C;
            const uint tap = reduction / CONV_C;
            byte = ByteAt(weights, CONV_WEIGHT_OFF + (tap % CONV_KERNEL) * CONV_WS0 +
                          (tap / CONV_KERNEL) * CONV_WS1 + inputChannel * CONV_WS2 + channel * CONV_WS3);
        }
        word |= byte << (lane * 8);
    }
    target.Store(index, word);
}

[numthreads(64, 1, 1)]
void post(uint3 tid : SV_DispatchThreadID) {
    const uint pixel = tid.y * 4096u * 64u + tid.x;
    if (pixel >= CONV_M)
        return;
    [unroll]
    for (uint group = 0; group < CONV_N / 4; ++group) {
    const uint index = pixel * CONV_N + group * 4;
    int4 quantized;
    [unroll]
    for (uint lane = 0; lane < 4; ++lane) {
        const uint channel = group * 4 + lane;
#ifdef CONV_BIAS_WORD
        const uint biasWord = CONV_BIAS_WORD(channel / 2);
#else
        const uint biasWord = weights.Load(CONV_BIAS_OFF + (channel & ~1u) * 2);
#endif
        const float bias = f16tof32((biasWord >> ((channel & 1u) * 16)) & 65535u);
        const int accumulator = asint(source.Load((index + lane) * 4));
#if CONV_RESIDUAL
        // Express the two multiply-add dependencies observed in the reference
        // ISA explicitly. `precise` is intentionally absent: with it DXC
        // lowered mad to separate multiply/add instructions in this toolchain.
        float value = mad(float(accumulator), (CONV_SIN * CONV_SW), bias);
        const uint packed = residual.Load(index);
        const int skip = int(packed << (24u - lane * 8u)) >> 24;
        value = mad(float(skip), CONV_SKIP_SCALE, value);
#else
        float value = float(accumulator) * (CONV_SIN * CONV_SW) + bias;
#endif
        int q = int(round(value * CONV_RCPOUT));
#if CONV_RELU
        q = max(q, 0);
#endif
        quantized[lane] = q;
    }
    target.Store(index, pack_clamp_s8(quantized));
    }
}
