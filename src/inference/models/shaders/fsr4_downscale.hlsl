// FSR 4 v07 quantized downscale convolution (model passes 3 and 6).
//
// Numeric contract taken from the pinned operator body
// (int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli, Conv_16_32 /
// Conv_32_64): INT32 accumulation over a 2x2 kernel with stride 2, FP16 bias
// added after scaling, and a per-tensor output factor that changes at the
// midpoint of the feature block (quantFactor0 below, quantFactor1 above).
// Packing, the native WMMA GEMM and the epilogue are the generic primitives in
// gpu/shaders/conv_i8.hlsli + gemm_i8.hlsl; only the fixed shapes and the
// operator-call constants are model-specific and live here.
#include "weights.generated.hlsli"

#if DS_PASS == 3
// pass 3: 960x540x16 INT8 -> 480x270x32 INT8
#define CONV_M 129600
#define CONV_K 64
#define CONV_C 16
#define CONV_N 32
#define CONV_W 480
#define CONV_IW 960
#define CONV_IH 540
#define CONV_WEIGHT_OFF M12_P3_W0_OFFSET
#define CONV_WS0 M12_P3_W0_STRIDE0
#define CONV_WS1 M12_P3_W0_STRIDE1
#define CONV_WS2 M12_P3_W0_STRIDE2
#define CONV_WS3 M12_P3_W0_STRIDE3
#define CONV_BIAS_OFF M12_P3_W1_OFFSET
#define CONV_SIN M12_P3_INPUT_SCALE
#define CONV_SW M12_P3_W0_SCALE
#define CONV_RCPOUT (1.0 / M12_P3_ARG0)
#define CONV_RCPOUT2 (1.0 / M12_P3_ARG1)
#define CONV_GROUP_SCALE0 M12_P3_EPI_SCALE0
#define CONV_GROUP_SCALE1 M12_P3_EPI_SCALE1
#define CONV_SCALE_SPLIT 16u
#elif DS_PASS == 6
// pass 6: 480x270x32 INT8 -> 240x135x64 INT8
#define CONV_M 32400
#define CONV_K 128
#define CONV_C 32
#define CONV_N 64
#define CONV_W 240
#define CONV_IW 480
#define CONV_IH 270
#define CONV_WEIGHT_OFF M12_P6_W0_OFFSET
#define CONV_WS0 M12_P6_W0_STRIDE0
#define CONV_WS1 M12_P6_W0_STRIDE1
#define CONV_WS2 M12_P6_W0_STRIDE2
#define CONV_WS3 M12_P6_W0_STRIDE3
#define CONV_BIAS_OFF M12_P6_W1_OFFSET
#define CONV_SIN M12_P6_INPUT_SCALE
#define CONV_SW M12_P6_W0_SCALE
#define CONV_RCPOUT (1.0 / M12_P6_ARG0)
#define CONV_RCPOUT2 (1.0 / M12_P6_ARG1)
#define CONV_GROUP_SCALE0 M12_P6_EPI_SCALE0
#define CONV_GROUP_SCALE1 M12_P6_EPI_SCALE1
#define CONV_SCALE_SPLIT 32u
#else
#error "DS_PASS must be 3 or 6"
#endif

#define CONV_STEP 2
#define CONV_KERNEL 2
#define CONV_PAD 0
// The 2x2 downscale has one channel group per tap, so the packed reduction
// channel count and the source channel stride are the same value.
#define CONV_IC CONV_C
// The shipped operator folds the output-scale reciprocal into the scale and
// the bias, so the epilogue is one FMA per element (see conv_i8.hlsli).
#define CONV_FOLDED_SCALES 1
#define CONV_RESIDUAL 0
#define CONV_RELU 0

#include "conv_i8.hlsli"
