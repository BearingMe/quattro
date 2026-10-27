// M06 control probe: the same 16x16x16 INT8 matrix tile computed with the
// standard SM 6.4 packed dot product (`dot4add_i8packed`) — AMD's own workhorse
// for FSR 4 INT8 passes (findings F1). This is NOT wave-matrix hardware; it is
// the baseline the WMMA probe is compared against.
//
// One output element C[i][j] per thread, 256 threads = one full 16x16 tile.
// Input/output layout matches int8_wmma.hlsl so both run on the same buffers.

ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0);

// Reads byte at `offset` from a 4-byte-aligned Load.
uint LoadByte(ByteAddressBuffer buf, uint offset) {
    const uint aligned = offset & ~3u;
    const uint shift = (offset & 3u) * 8u;
    return (buf.Load(aligned) >> shift) & 0xFFu;
}

[numthreads(256, 1, 1)]
void main(uint tid : SV_GroupThreadID) {
    const uint i = tid >> 4;  // output row    (0..15)
    const uint j = tid & 15u; // output column (0..15)

    // Accumulate over 16 K-elements in 4 packed dot products.
    // C[i][j] = sum_k A[i][k] * B[k][j]  (signed x signed) + initC[i][j].
    int acc = asint(g_in.Load(512 + (i * 16u + j) * 4u)); // initC
    [unroll]
    for (uint k4 = 0; k4 < 4; ++k4) {
        uint aPacked = 0;
        uint bPacked = 0;
        [unroll]
        for (uint m = 0; m < 4; ++m) {
            const uint k = k4 * 4u + m;
            aPacked |= LoadByte(g_in, i * 16u + k) << (8u * m);          // A[i][k]
            bPacked |= LoadByte(g_in, 256u + k * 16u + j) << (8u * m);   // B[k][j]
        }
        acc = dot4add_i8packed(aPacked, bPacked, acc);
    }
    g_out.Store((i * 16u + j) * 4u, asuint(acc));
}
