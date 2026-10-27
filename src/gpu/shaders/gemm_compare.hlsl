// M07 comparison workload: 64 independent 16x16xK GEMMs per dispatch (the
// dispatch geometry carries the GEMM count), one wave (32 threads) per GEMM
// for the WMMA kernels, one 256-thread group per GEMM for the DP4A control.
// Row-major: A is 16xK, B is Kx16, both in the kernel's operand format;
// initC/result tiles are 16x16 in the accumulator format. Input layout per
// GEMM: A | B | initC (packed, 4-byte aligned).
//
// FIXED GRAPH: K is compile-time (-D GEMM_K_TILES, one CSO per K) because the
// magic-code intrinsic sequences are phase-locked — the driver patches them
// reliably only in straight-line code, exactly like FSR 4's code-generated
// passes (ML2Code emits one pass per tile group, no runtime K loops).
// K-scaling between two benchmarked K values distinguishes arithmetic from
// fixed overhead (M07 C07).

#include "AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

#ifndef GEMM_K_TILES
#define GEMM_K_TILES 16
#endif

ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0);

// int8 WMMA path: A/B int8, C int32. Per-GEMM bytes: A 16*K + B K*16 + C 1024.
// FSR4-proven structure: all operand tiles are loaded BEFORE the WMMA chain —
// no buffer loads interleaved between chained magic-code sequences.
[numthreads(32, 1, 1)]
void gemm_int8_wmma(uint3 gid : SV_GroupID) {
    const uint k = GEMM_K_TILES * 16u;
    const uint gemmBase = gid.x * (16u * k * 2u + 1024u);

    AmdWaveMatrixA <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> matsA[16];
    AmdWaveMatrixB <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> matsB[16];
    AmdWaveMatrixAccumulator <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> acc;
    acc.Load(g_in, gemmBase + 16u * k * 2u, 64, false);

    [unroll]
    for (uint t = 0; t < GEMM_K_TILES; ++t) {
        const uint tileBase = gemmBase + t * 512u;
        matsA[t].Load(g_in, tileBase, 16, false);
        matsB[t].Load(g_in, tileBase + 256u, 16, false);
    }
    [unroll]
    for (uint t = 0; t < GEMM_K_TILES; ++t) {
        acc = AmdWaveMatrixMultiply<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8,
                                    AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8,
                                    AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32>(matsA[t], matsB[t], acc);
    }
    acc.Store(g_out, gid.x * 1024u, 64, false);
}

// fp16 WMMA path: A/B fp16, C fp32. Per-GEMM bytes: A 32*K + B 32*K + C 1024.
[numthreads(32, 1, 1)]
void gemm_fp16_wmma(uint3 gid : SV_GroupID) {
    const uint k = GEMM_K_TILES * 16u;
    const uint gemmBase = gid.x * (32u * k * 2u + 1024u);

    AmdWaveMatrixA <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matsA[16];
    AmdWaveMatrixB <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matsB[16];
    AmdWaveMatrixAccumulator <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32, 16, 16> acc;
    acc.Load(g_in, gemmBase + 32u * k * 2u, 64, false);

    [unroll]
    for (uint t = 0; t < GEMM_K_TILES; ++t) {
        const uint tileBase = gemmBase + t * 1024u;
        matsA[t].Load(g_in, tileBase, 32, false);
        matsB[t].Load(g_in, tileBase + 512u, 32, false);
    }
    [unroll]
    for (uint t = 0; t < GEMM_K_TILES; ++t) {
        acc = AmdWaveMatrixMultiply<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
                                    AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
                                    AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32>(matsA[t], matsB[t], acc);
    }
    acc.Store(g_out, gid.x * 1024u, 64, false);
}

// DP4A control: int8 math on the same GEMM set, 256 threads = one 16x16 tile.
// int8 layout matches gemm_int8_wmma (A 16*K bytes | B K*16 bytes | C 1024).
uint LoadByte(ByteAddressBuffer buf, uint offset) {
    const uint aligned = offset & ~3u;
    const uint shift = (offset & 3u) * 8u;
    return (buf.Load(aligned) >> shift) & 0xFFu;
}

[numthreads(256, 1, 1)]
void gemm_int8_dp4a(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint k = GEMM_K_TILES * 16u;
    const uint gemmBase = gid.x * (16u * k * 2u + 1024u);

    const uint i = tid.x >> 4;
    const uint j = tid.x & 15u;
    int acc = asint(g_in.Load(gemmBase + 16u * k * 2u + (i * 16u + j) * 4u));
    [unroll]
    for (uint t = 0; t < GEMM_K_TILES; ++t) {
        const uint tileBase = gemmBase + t * 512u;
        [unroll]
        for (uint k4 = 0; k4 < 4; ++k4) {
            uint aPacked = 0;
            uint bPacked = 0;
            [unroll]
            for (uint m = 0; m < 4; ++m) {
                const uint kk = k4 * 4u + m; // tile-local K index (tiles are 16x16)
                aPacked |= LoadByte(g_in, tileBase + i * 16u + kk) << (8u * m);          // A[i][kk]
                bPacked |= LoadByte(g_in, tileBase + 256u + kk * 16u + j) << (8u * m);   // B[kk][j]
            }
            acc = dot4add_i8packed(aPacked, bPacked, acc);
        }
    }
    g_out.Store(gid.x * 1024u + (i * 16u + j) * 4u, asuint(acc));
}
