// M07 probe: native RDNA 3 FP16 WMMA through AMD's D3D12 shader-intrinsics
// extension. Same harness shape as int8_wmma.hlsl (one 16x16x16 tile per
// Wave32 group, row-major tiles in t0, result tile to u0).
//
// Two accumulator variants (M07 C05):
//   main_f32acc — WMMA_F32_16X16X16_F16: fp16 inputs, fp32 accumulator (1024 B out)
//   main_f16acc — WMMA_F16_16X16X16_F16: fp16 inputs, fp16 accumulator (512 B out)
//
// Layout (t0): A 16x16 fp16 (512 B) | B 16x16 fp16 (512 B) | initC (1024 B
// fp32, or 512 B fp16 for the f16acc entry point).

#include "AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0);

[numthreads(32, 1, 1)]
void main_f32acc() {
    AmdWaveMatrixA <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matA;
    AmdWaveMatrixB <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matB;
    AmdWaveMatrixAccumulator <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32, 16, 16> matC;

    matA.Load(g_in, 0, 32, false);
    matB.Load(g_in, 512, 32, false);
    matC.Load(g_in, 1024, 64, false);

    matC = AmdWaveMatrixMultiply<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
                                 AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
                                 AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32>(matA, matB, matC);

    matC.Store(g_out, 0, 64, false);
}

[numthreads(32, 1, 1)]
void main_f16acc() {
    AmdWaveMatrixA <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matA;
    AmdWaveMatrixB <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matB;
    AmdWaveMatrixAccumulator <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16, 16, 16> matC;

    matA.Load(g_in, 0, 32, false);
    matB.Load(g_in, 512, 32, false);
    matC.Load(g_in, 1024, 32, false);

    matC = AmdWaveMatrixMultiply<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
                                 AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16,
                                 AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16>(matA, matB, matC);

    matC.Store(g_out, 0, 32, false);
}
