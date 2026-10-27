// M06 probe: native RDNA 3 INT8 WMMA through AMD's D3D12 shader-intrinsics
// extension (the path FSR 4.0.2 shipped: AmdWaveMatrix*).
//
// One 16x16x16 tile per wave (numthreads(32,1,1) = Wave32, required by the
// RDNA 3 WMMA ISA). C = A x B + C with INT32 accumulation; A/B are INT8 or
// UINT8 per entry point (opcode selection: WMMA_I32_16X16X16_{I8,U8,IU8,UI8}).
//
// Data layout (row-major, no padding):
//   t0: A 16x16 int8 (256 B) | B 16x16 int8 (256 B) | initC 16x16 int32 (1024 B)
//   u0 space0: C 16x16 int32 (1024 B)
//   u0 space 0x7FFF0ADE: intrinsics dummy UAV (declared by the AMD header —
//   must be bound to a real buffer in the root signature).

#include "AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0); // space0 output; the AMD header owns u0 space 0x7FFF0ADE

template <uint FMT_A, uint FMT_B>
void RunTile() {
    AmdWaveMatrixA <FMT_A, 16, 16> matA;
    AmdWaveMatrixB <FMT_B, 16, 16> matB;
    AmdWaveMatrixAccumulator <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> matC;

    matA.Load(g_in, 0, 16, false);                                        // A, 16 B/row
    matB.Load(g_in, 256, 16, false);                                      // B, 16 B/row
    matC.Load(g_in, 512, 64, false);                                      // initC, 64 B/row

    matC = AmdWaveMatrixMultiply<FMT_A, FMT_B,
                                 AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32>(matA, matB, matC);

    matC.Store(g_out, 0, 64, false);
}

[numthreads(32, 1, 1)]
void main_i8() {
    RunTile<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8,
            AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8>();
}

[numthreads(32, 1, 1)]
void main_u8() {
    RunTile<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_U8,
            AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_U8>();
}

[numthreads(32, 1, 1)]
void main_iu8() {
    RunTile<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8,
            AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_U8>();
}

[numthreads(32, 1, 1)]
void main_ui8() {
    RunTile<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_U8,
            AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8>();
}
