// M09: generalized INT8 GEMM on the M06-validated native WMMA path.
//
// C = A(M x K) x B(K x N), all operands int8, INT32 accumulation via
// v_wmma_i32_16x16x16_iu8. Runtime M/N through the dispatch grid (one Wave32
// group per 16x16 output tile); EVERYTHING shape-related is compile-time
// (-D): GEMM_K_TILES, GEMM_N (padded), GEMM_M (padded), and the blob
// offsets — the M06/M07-established driver constraint (magic-code
// immediates bake strides; the analyzer rejects cbuffer-carrying WMMA
// shaders with E_INVALIDARG), and per-shape codegen is exactly FSR4's
// ML2Code model (one generated pass per tensor shape, all constants baked).
//
// Blob layouts (byte offsets are baked -D defines, derived from the shape):
//   gemm_raw:     [A: Mp*K | B: K*Np | zeroTile: 1024]  — u0: Mp*Np int32
//   gemm_requant: [acc: Mp*Np*4 | bias: N fp16]         — u0: Mp*Np int8
// Boundary tiles: zero-padded operands; padded rows/cols contribute zero
// and are cropped on readback.
//
// Entry points:
//   gemm_raw     — int32 accumulator tiles (vs CPU INT32 reference).
//   gemm_requant — AMD FNB_CT2D store idiom per element (the M08 contract):
//                  q = pack_clamp_s8(round((acc * (sIn*sW) + bias) * (1/sOut)))
//                  Separate pass, not fused into the WMMA chain — the fusion
//                  is M15 optimization territory (EXP-005); correctness
//                  first (M09 boundary). Runtime loops, no magic codes.

#include "AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

#ifndef GEMM_K_TILES
#define GEMM_K_TILES 4
#endif
#ifndef GEMM_N
#define GEMM_N 256
#endif
#ifndef GEMM_B_OFF
#define GEMM_B_OFF 0
#endif
#ifndef GEMM_ZERO_OFF
#define GEMM_ZERO_OFF 0
#endif
// requant only (scales ride on these #ifndef defaults; CMake passes per-CSO
// defines. The M11 bake — GEMM_SIN=1.0f, GEMM_SW=1.0f, GEMM_SOUT=16.0f — is
// EXACT power-of-two arithmetic, so requantization is bit-exact vs an integer
// oracle. Non-power-of-two scales make NO such guarantee — see EXP-009.):
#ifndef GEMM_ACC_OFF
#define GEMM_ACC_OFF 0
#endif
#ifndef GEMM_BIAS_OFF
#define GEMM_BIAS_OFF 0
#endif
#ifndef GEMM_N_REAL
#define GEMM_N_REAL 16
#endif
// M11: the fixed network's first layer applies ReLU at requantization; the
// second does not. Clamping lower bound is baked per CSO (codegen model).
#ifndef GEMM_RELU
#define GEMM_RELU 0
#endif
#ifndef GEMM_SIN
#define GEMM_SIN 0.014884727f
#endif
#ifndef GEMM_SW
#define GEMM_SW 0.02f
#endif
#ifndef GEMM_SOUT
#define GEMM_SOUT 0.5f
#endif

ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0);

// Strides MUST be compile-time: the WMMA Load/Store patch ops encode them in
// the magic-code immediates (M06/M07 finding).
static const uint KStride = GEMM_K_TILES * 16u; // bytes per A row (int8)

// Requant scales, derived at runtime from the baked per-tensor scales
// (bit-identical to the CPU's fp32 arithmetic).
static const float combined = GEMM_SIN * GEMM_SW;
static const float rcpOut = 1.0f / GEMM_SOUT;

[numthreads(32, 1, 1)]
void gemm_raw(uint3 gid : SV_GroupID) {
    const uint n0 = gid.x * 16u; // tile column (grid = (Np/16, Mp/16))
    const uint m0 = gid.y * 16u; // tile row

    AmdWaveMatrixA <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> matA;
    AmdWaveMatrixB <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> matB;
    AmdWaveMatrixAccumulator <AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> acc;
    acc.Load(g_in, GEMM_ZERO_OFF, 64, false); // zero tile (M06 Load pattern)

    [unroll]
    for (uint t = 0; t < GEMM_K_TILES; ++t) {
        const uint kt = t * 16u;
        matA.Load(g_in, uint(m0) * KStride + kt, KStride, false); // A@0
        matB.Load(g_in, GEMM_B_OFF + kt * GEMM_N + n0, GEMM_N, false);
        acc = AmdWaveMatrixMultiply<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8,
                                    AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8,
                                    AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32>(matA, matB, acc);
    }
    acc.Store(g_out, uint(m0) * GEMM_N * 4u + n0 * 4u, GEMM_N * 4u, false);
}

// Requant pass: AMD FNB_CT2D store idiom over the raw int32 accumulator.
// ONE THREAD PER 4 OUTPUT ELEMENTS (a dword = 4 int8 channels — GEMM_N is
// 16-aligned so a dword never crosses a row): ByteAddressBuffer::Store is
// always 4 bytes wide, so per-byte stores at unaligned offsets would clobber
// neighboring elements (AMD's operators enforce the same "aligned to 4"
// rule — see QuantizeLinear's perThreadWork.z % 4 check).
// Runtime loops, no magic codes. Scale arithmetic in the source's exact
// order: sIn*sW computed at runtime from the baked per-tensor scales
// (bit-identical to the CPU's fp32 product; baking the product instead
// could differ by 1 ULP).
[numthreads(64, 1, 1)]
void gemm_requant(uint3 dtid : SV_DispatchThreadID) {
    const uint dwordIdx = dtid.x;            // flat over Mp * Np / 4
    const uint elemBase = dwordIdx * 4u;     // first channel of the pack
    const uint m = elemBase / GEMM_N;
    const uint n0 = elemBase - m * GEMM_N;   // 4-aligned: GEMM_N % 4 == 0

    uint packed = 0;
    [unroll]
    for (uint lane = 0; lane < 4; ++lane) {
        const int32_t acc = asint(g_in.Load(GEMM_ACC_OFF + (elemBase + lane) * 4u));
        const uint biasCh = min(n0 + lane, GEMM_N_REAL - 1u); // padded cols reuse last bias
        // fp16 bias loads are 2-byte aligned at best — dword loads must
        // align down and select the half (same idiom as AMD's Load1hU in
        // ml2code_runtime/tensor_float16.hlsli).
        const uint biasByte = GEMM_BIAS_OFF + biasCh * 2u;
        const uint word = g_in.Load(biasByte & ~3u);
        const float bias = f16tof32((biasByte & 3u) ? (word >> 16) : (word & 0xFFFFu));

        // AMD FNB_CT2D L113: round((acc * (sIn*sW) + bias) * (1/sOut)), then
        // pack_clamp_s8. Same fp32 operation order as the source, per element.
        // GEMM_RELU=1 clamps to [0,127] (the ConvNextBlock/FasterNetBlock
        // activation idiom) instead of [-128,127].
        const float value = (float(acc) * combined + bias) * rcpOut;
#if GEMM_RELU
        const int q = clamp(int(round(value)), 0, 127);
#else
        const int q = clamp(int(round(value)), -128, 127);
#endif
        packed |= (uint(q) & 0xFFu) << (lane * 8u);
    }
    g_out.Store(dwordIdx * 4u, packed);
}
