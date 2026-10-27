// M08 probe: executes AMD's SHIPPED QuantizeLinear operator (vendored
// verbatim from the pinned FSR 4.0.2 mirror, see src/gpu/amd/PROVENANCE.md)
// on gfx1101 — the C09 evidence that the CPU quantization contract matches
// the real implementation. The tensor/scale struct init follows the generated
// FSR 4 passes (quantizationScale baked as a shader constant, ML2Code style).
//
// Compile flags match the shipped model header: -O3 -enable-16bit-types -HV 2021.
//
// Shape: 1x1x256 NHWC tensor (256 fp16 channels in t0, 256 int8 channels
// out u0). One dispatch covers all channels; QuantizeLinear's internal
// work-split degenerates to thread 0 for this shape (deterministic).

#include "ml2code_runtime/operators/int8_NHWC/QuantizeLinear.hlsli"

#ifndef PROBE_SCALE
#define PROBE_SCALE 1.0f
#endif

ByteAddressBuffer g_in : register(t0);
RWByteAddressBuffer g_out : register(u0);

[numthreads(64, 1, 1)]
void quantize_amd(uint3 gid : SV_GroupID,
                  uint3 gtid : SV_GroupThreadID,
                  uint3 dtid : SV_DispatchThreadID) {
    const uint3 logical = uint3(1, 1, 256);
    const uint3 strides = uint3(512, 512, 2); // fp16: W/H stride = 2*C, C stride = 2

    Tensor3h_NHWC<BufferStorage> input = {
        logical,            // logicalSize
        int3(0, 0, 0),      // threadGroupSliceStart
        logical,            // threadGroupSliceSize
        logical,            // storageSize
        strides,            // storageByteStrides
        uint3(0, 0, 0),     // paddingBegin
        uint3(0, 0, 0),     // paddingEnd
        0,                  // threadGroupStorageByteOffset
        g_in                // storage
    };

    QuantizedTensor3i8_NHWC<RWBufferStorage> output = {
        logical,            // logicalSize
        int3(0, 0, 0),      // threadGroupSliceStart
        logical,            // threadGroupSliceSize
        logical,            // storageSize
        uint3(256, 256, 1), // storageByteStrides (int8: C stride = 1)
        uint3(0, 0, 0),     // paddingBegin
        uint3(0, 0, 0),     // paddingEnd
        0,                  // threadGroupStorageByteOffset
        float(PROBE_SCALE), // quantizationScale (baked, ML2Code style)
        g_out               // storage
    };

    ComputeShaderParams params;
    params.numThreads = uint3(64, 1, 1);
    params.groupID = gid;
    params.groupThreadID = gtid;
    params.dispatchThreadID = dtid;

    // Scale and zero-point tensors are ignored by the operator (scale is
    // baked into the output tensor metadata) — NullTensor placeholders.
    NullTensor unusedScale;
    NullTensor unusedZeroPoint;
    QuantizeLinear(input, unusedScale, unusedZeroPoint, output, params);
}
