# M10 operator and tensor contract

## Scope and source trace

- [S] Selected architecture: **FSR 4.0.2, c5e34b4, v07 INT8 Quality,
  `passes_1080.hlsl`**. This is an open-source architectural reference, not
  evidence about FSR 4.1. Paths below are relative to
  `.references/sources/FSR-4.0.2-reference/Kits/FidelityFX/upscalers/fsr4/`.
- [S] `scripts/m10_inventory.py` enumerates all 14 generated main-pass calls and
  resolves their bias declarations, and records padding calls in
  `build/reports/m10/source-inventory.json`. **Every called bias is Tensor1h**;
  nearby INT32-bias overloads are not the selected calls. Overload selection
  and arithmetic below were traced manually, not proved by the script.
- [S] Generated graph inputs are FP16 NHWC features; the first convolution
  uses FP16 weights with FP32 accumulation, then quantizes to INT8. The last
  generated output is FP16 NHWC, 1920×1080×8 (lines 5079–5092). Consequently
  “INT8 network” does not mean all intermediates or all convolutions are INT8.

All runtime paths below start at `dx12/ml2code_runtime/operators/`:

| Generated call lines | Selected overload | Non-matrix constituents |
|---|---|---|
| 152 | `int8_NHWC/Conv2D_k2s2b.hlsli:280` (FP16 NHWC weights) | FP32 accumulator + FP16 bias, reciprocal-scale quantize, signed saturation |
| 528, 904, 4655 | `int8_NHWC/Fused/ConvNextBlock.hlsli:180` (FP16 biases) | INT32→FP32 scaling + FP16 bias; requantize; ReLU [0,127]; differently scaled residual; output quantize |
| 1139, 2392 | `int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput.hlsli:684`, dispatching FP16-bias helpers at 87 / 447 | Quantized convolution result and skip-group storage; two output channel-group scales |
| 1711, 2569, 2746, 3536 | `int8_NHWC/Fused/FasterNetBlock.hlsli:471` | Channel split/view; partial-conv result concatenated with untouched input half; affine/ReLU; scaled residual; two output-group scales |
| 2284 | same file, overload at 2216 | Same constituents, uniform output quantization scale rather than two explicit output-group scales |
| 2964 | `int8_NHWC/Fused/FNB_CT2D_ADD.hlsli:14`, `<64,2>` | Split/concat, affine/ReLU, two differently scaled residual halves, INT8 boundary before CT2D, four spatial phases, skip-add then output quantize with group scales |
| 4279 | same file, overload at 348, `<32,1>` | Same constituents, uniform final output scale |
| 5092 | `float16_NHWC/Fused/CNB_CT2D.hlsli:118`, `<8>` | INT8 CNB and residual requantization; four CT2D spatial phases; **half(acc×scale) then half(+FP16 bias)** at 75, 275–276; FP16 output |

- [S] Source arithmetic anchors: ConvNextBlock 269–300, 328–344;
  FasterNetBlock 683–705, 751–798, 830–866; FNB_CT2D_ADD 202–251,
  254–342 and 518–548, 620–628; CNB_CT2D 189–219, 247–285.
- [S] Post-pass `ResetPaddingSeparate` calls reset storage borders; tensors
  use explicit logical sizes, storage strides, offsets and channel grouping.
  Examples: generated lines 168–182 and 5079–5089. Split/concat can be fused
  register operations or strided views; they do not mandate graph dispatches.
- [S] Upsampling is **learned k2s2 transposed convolution**, with phase order
  (0,0), (1,0), (0,1), (1,1). There is no standalone nearest/bilinear
  interpolation, normalization, GELU or sigmoid in these selected main passes.
- [S] `pre.hlsl`/`post.hlsl` include model wrappers `pre_common.hlsli` /
  `post_common.hlsli`; feature construction, temporal resources and image
  reconstruction are outside the selected neural-main-pass inventory and
  remain FSR-specific work, not a hidden claim of M10 completion of M12–M14.

## Implemented constituent collection

- [S] `core/tensor.h` owns plain tensor/operation/transform metadata;
  `gpu/tensor_ops.*` and `gpu/shaders/tensor_ops.hlsl` implement the collection;
  `reference/tensor_ops.*` is the independent scalar CPU oracle.
- [S] Operations: signed-byte ReLU; FP16/FP32 + FP16-bias quantization;
  INT32 affine requantization (optional ReLU); INT32 affine + independently
  scaled INT8 residual; two-round FP16 output; strided copy/view materialization;
  channel concat; symmetric zero padding; four-phase spatial scatter.
- [S] `Spatial2x` only scatters **already computed phase channels**, with
  `input[y,x,phase*C+c] → output[2*y+dy,2*x+dx,c]`. It does not replace
  learned transposed-convolution arithmetic. Phase GEMMs/convolutions,
  model weights and full fused blocks are not implemented by M10.
- [S] No general graph, parser, allocator, interpolation kernel or speculative
  activation was added. The isolated runner still uploads and reads back each
  operation. GPU-resident multi-layer execution belongs to M11.

## Tensor/storage and numerical rules

- [S] Batch=1 NHWC, width/height/channels explicit. Byte offset and byte
  pixel/row strides allow padded storage and zero-copy channel views.
  Address = offset + y×rowStride + x×pixelStride + c×elementBytes.
  INT8=1, FP16=2, INT32/FP32=4 bytes. Multi-byte values are naturally aligned;
  INT8 channel views need not be dword-aligned. Reinterpretation never changes
  a type; shape-only views retain byte order, not an NCHW↔NHWC conversion.
- [S] Outputs are dense NHWC, padded to a whole dword with **zero trailing
  bytes**. One shader thread owns each dword, avoiding packed-byte write races.
  Dispatch is tiled over X/Y for buffers beyond 65,535 X groups, including the
  selected 1080p FP16 output. Each storage/output is bounded to 256 MiB.
- [S] Each output channel has `{inputScale,residualScale,reciprocalOutputScale,
  biasF16}`. Identical records represent per-tensor scales; repeated segments
  represent source channel groups. Concat copies bytes, **not** scales: callers
  must preserve the concatenated groups' metadata. No implicit rescaling.
- [S] For affine: explicitly rounded FP32 `float(acc)*inputScale`, `+bias`,
  optionally `+float(residual)*residualScale`, `*reciprocalOutputScale`, RNE,
  clamp to [-128,127] (or [0,127]). `precise` prevents contraction and the
  shader is compiled with `-denorm preserve`; the oracle
  uses independent scalar arithmetic and manual floor/fraction ties-to-even.
  This deliberately defined contract is **not universal bit-equivalence to
  AMD fused expressions**, whose multiply association/contraction can differ.
- [S] FP16 output explicitly rounds twice using IEEE RNE, preserving subnormals
  and overflowing to signed infinity. NaN/Inf floating quantization inputs,
  nonpositive/nonfinite scales, nonfinite biases, unsafe FP32 ranges, invalid
  strides/types/shapes/storage and invalid dispatch sizes reject before GPU work.
- [M] An initial `f32tof16` packing probe disagreed with the CPU at 65520
  (GPU 0x7bff vs expected 0x7c00) and at rounding boundaries. Integer-bit RNE
  packing replaced that dependency; negative evidence and fixed tests are in
  [EXP-008](../experiments/EXP-008-neural-operators.md). This is scoped to the
  tested shader/compiler/driver path, not a general statement about native half
  arithmetic or AMD's final fused kernel.
- [S] `ValidateInt8Accumulation` rejects K above 131071 for full-range signed
  INT8, accounting for -128×-128=16384 and optional accumulator-domain bias.
  M09 raw GEMM, convolution and benchmark callers now enforce it. Arbitrary
  GPU dispatch through the low-level runner still requires its caller's guard.

## Evidence boundary

- [M] Independent hardware comparisons and reproducible commands are in
  [EXP-008](../experiments/EXP-008-neural-operators.md); exact model-size layout
  scatter is tested, not a complete FSR network.
- [U] M09's cbuffer/analyzer failure is not an isolated proof that runtime
  constants are generally impossible. M10 uses an SRV header without making
  that claim. M09 GPU-validation-on timings are not a clean-path performance
  ceiling. The existing contracted M09 requantizer remains a separate contract;
  M10's contraction-sensitive fixture only proves the new operator path.
