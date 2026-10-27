// Fixed ConvNextBlock recipe; generic kernels remain model-independent.
#include "weights.generated.hlsli"
#define JOIN_IMPL(a,b) a##b
#define JOIN(a,b) JOIN_IMPL(a,b)
#define PFX JOIN(M12_P, CNB_PASS)
#define PARAM(suffix) JOIN(PFX,suffix)
#define CONV_W 960
#define CONV_IW 960
#define CONV_IH 540
#define CONV_IC CONV_C
#define CONV_STEP 1
#if CNB_STAGE == 0
#define CONV_KERNEL 3
#define CONV_PAD 1
#define CONV_WEIGHT_OFF PARAM(_W0_OFFSET)
#define CONV_WS0 PARAM(_W0_STRIDE0)
#define CONV_WS1 PARAM(_W0_STRIDE1)
#define CONV_WS2 PARAM(_W0_STRIDE2)
#define CONV_WS3 PARAM(_W0_STRIDE3)
#define CONV_BIAS_OFF PARAM(_W1_OFFSET)
#define CONV_BIAS_WORD(i) PARAM(_W1_WORDS)[i]
#define CONV_SIN PARAM(_INPUT_SCALE)
#define CONV_SW PARAM(_W0_SCALE)
#define CONV_RCPOUT PARAM(_ARG0)
#define CONV_RESIDUAL 0
#define CONV_RELU 0
#elif CNB_STAGE == 1
#define CONV_KERNEL 1
#define CONV_PAD 0
#define CONV_WEIGHT_OFF PARAM(_W2_OFFSET)
#define CONV_WS0 PARAM(_W2_STRIDE0)
#define CONV_WS1 PARAM(_W2_STRIDE1)
#define CONV_WS2 PARAM(_W2_STRIDE2)
#define CONV_WS3 PARAM(_W2_STRIDE3)
#define CONV_BIAS_OFF PARAM(_W3_OFFSET)
#define CONV_BIAS_WORD(i) PARAM(_W3_WORDS)[i]
#define CONV_SIN PARAM(_ARG1)
#define CONV_SW PARAM(_W2_SCALE)
#define CONV_RCPOUT PARAM(_ARG2)
#define CONV_RESIDUAL 0
#define CONV_RELU 1
#else
#define CONV_KERNEL 1
#define CONV_PAD 0
#define CONV_WEIGHT_OFF PARAM(_W4_OFFSET)
#define CONV_WS0 PARAM(_W4_STRIDE0)
#define CONV_WS1 PARAM(_W4_STRIDE1)
#define CONV_WS2 PARAM(_W4_STRIDE2)
#define CONV_WS3 PARAM(_W4_STRIDE3)
#define CONV_BIAS_OFF PARAM(_W5_OFFSET)
#define CONV_BIAS_WORD(i) PARAM(_W5_WORDS)[i]
#define CONV_SIN PARAM(_ARG3)
#define CONV_SW PARAM(_W4_SCALE)
#define CONV_RCPOUT (1.0f/PARAM(_OUTPUT_SCALE))
#define CONV_RESIDUAL 1
#define CONV_SKIP_SCALE PARAM(_INPUT_SCALE)
#define CONV_RELU 0
#endif
#include "conv_i8.hlsli"
