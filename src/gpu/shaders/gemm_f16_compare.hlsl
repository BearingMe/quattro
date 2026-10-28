// Equivalent integer-domain GEMM via native FP16 operands + FP32 accumulation.
// All signed INT8 operands are exactly representable as half; for these K<=144
// shapes, integer partial sums are below 2^24. Store converted I32 for the
// unchanged numeric epilogue. This is a backend comparison, not new FP16 weights.
#include "AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

ByteAddressBuffer source : register(t0);
RWByteAddressBuffer target : register(u0);
static const uint BOffset = CMP_M * CMP_K;
static const uint ZeroOffset = BOffset + CMP_K * CMP_N;

[numthreads(64, 1, 1)]
void convert_packet(uint3 tid : SV_DispatchThreadID) {
    const uint wordIndex = tid.y * 4096u * 64u + tid.x;
    if (wordIndex * 4 >= ZeroOffset + 1024) return;
    const uint word = source.Load(wordIndex * 4);
    uint h[4];
    [unroll] for (uint lane = 0; lane < 4; ++lane) {
        const int value = int(word << (24u - lane * 8u)) >> 24;
        h[lane] = f32tof16(float(value));
    }
    target.Store2(wordIndex * 8, uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16)));
}

[numthreads(32, 1, 1)]
void gemm_raw(uint3 gid : SV_GroupID) {
    const uint n0 = gid.x * 16u, m0 = gid.y * 16u;
    AmdWaveMatrixA<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> a;
    AmdWaveMatrixB<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> b;
    AmdWaveMatrixAccumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32, 16, 16> acc;
    acc.Load(source, ZeroOffset * 2, 64, false);
    [unroll] for (uint tile = 0; tile < CMP_K / 16; ++tile) {
        const uint k = tile * 16u;
        a.Load(source, (m0 * CMP_K + k) * 2, CMP_K * 2, false);
        b.Load(source, (BOffset + k * CMP_N + n0) * 2, CMP_N * 2, false);
        acc = AmdWaveMatrixMultiply<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
            AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
            AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32>(a, b, acc);
    }
    AmdWaveMatrixAccumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> integer;
    // Both Copy and explicit truncation failed; first-product differences were
    // bounded to one integer unit. Round before conversion, then validate every
    // retained boundary rather than assuming exact real-arithmetic WMMA sums.
    [unroll] for (uint element = 0; element < 8; ++element)
        integer.SetElement(element, int(round(acc.Element(element))));
    integer.Store(target, (m0 * CMP_N + n0) * 4, CMP_N * 4, false);
}
