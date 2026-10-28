# EXP-012: quantized downscale convolution (model passes 3 and 6)

Date: 2026-10-28 · Milestone: M12, incremental C04 evidence

Hardware: RX 7700 XT/gfx1101, AMD driver 32.0.32015.7029. Graph/oracle context:
[EXP-010](EXP-010-fsr4-model-graph.md). Operator/weight contract:
[EXP-011](EXP-011-fsr4-reproduction-operators.md).

## Objective and scope

Reproduce the two quantized stride-2 downscale convolutions of
`fsr4_model_v07_i8_quality` (`int8_NHWC/Fused/FusedConv2D_k2s2b_QuantizedOutput`)
as our own GPU code and compare them byte-for-byte against the isolated shipped
pass. This covers **passes 3 and 6 only**, not the complete 14-pass model and not
the M12 gate.

| Pass | Input | Weights | Output | Group scales |
|---|---|---|---|---|
| 3 | 960×540×16 INT8 | 2×2×16×32, scale 0.0034550556 | 480×270×32 INT8 | 0.015390855 / 0.018884974 |
| 6 | 480×270×32 INT8 | 2×2×32×64, scale 0.0031457620 | 240×135×64 INT8 | 0.019169601 / 0.028077777 |

## Method

- [S] Geometry, strides, scales and the operator-call scalars come from the
  pinned descriptors already validated by `scripts/m12_validate_graph.py`; no
  new model metadata is introduced.
- [S] Three fixed dispatches per pass, all GPU-resident: im2col/weight **pack**
  → native WMMA INT32 **GEMM** (`v_wmma_i32_16x16x16_iu8`) → **requant
  epilogue**. Packing and the epilogue are the generic primitives in
  `src/gpu/shaders/conv_i8.hlsli`; the model file only supplies the baked
  shapes and the operator-call constants.
- [S] The epilogue is not the source-text expression. The shipped pass's own
  ISA shows `v_fmaak_f32 sGroup, vAcc, lit(bias*rcp)` followed by `v_rndne_f32`
  and a narrowing clamp: the driver folded the output-scale reciprocal into
  both the weight·input scale and the bias, so there is **one FMA and no
  trailing multiply**. See
  [the epilogue finding](../findings/2026-10-28-fsr4-quantized-output-epilogue.md).
  The folded group scales are emitted by `scripts/m12_pack_model.py` as
  `M12_P{n}_EPI_SCALE{0,1}`.
- [S] The isolated reference is executed from the pinned tree with the
  provider's grids (8×270 for pass 3, 4×135 for pass 6) over a zero-initialized
  20,736,000 + 1,024 B scratch. Both passes bound their writes with the shipped
  `ValidPosition` check, so no overhang is involved (unlike pass 11).
- [S] Fixtures: all-zero INT8, and a deterministic LCG stream covering the full
  signed INT8 range. Both are fed byte-identically to our pass and to the
  shipped pass.

## Results

- [M] `run fsr4-downscale` — **pass 3: 0/4,147,200 differing bytes** on both
  fixtures; **pass 6: 0/2,073,600 differing bytes** on both. Maximum INT8 delta
  0, repeated execution bit-identical, `v_wmma_i32_16x16x16_iu8` present in the
  analyzed GEMM pipeline of both passes.
- [M] Full suites after the change, both configurations, running the same four
  byte-exact comparisons: **Debug 19/19 PASS (369.88 s)** and **Release 19/19
  PASS (371.28 s)**, including `experiment_fsr4_downscale` and every earlier
  M05–M11/M12 check.
- [M] The ISA-derived formula was first verified offline against the captured
  tensors: `round(fma(acc, fl(fl(sW·sIn)/qf), bias·(1/qf)))` reproduces the
  shipped output with 0 mismatches on all 129,600 (pass 3) and 32,400 (pass 6)
  pixels of the fuzz fixture.
- [M] Negative evidence retained during development: the naive source-form
  epilogue `(acc·sW·sIn + bias)·(1/qf)` produced 11/4,147,200 mismatches
  (max delta 1), all at exact half-integer boundaries. Exactness required
  reproducing the reference's *folded* form, not its source text.
- [M] Harness trap recorded: an isolated-reference run that binds a dummy in
  the `t1` slot (AMD's `InitializerBuffer`) hangs the GPU
  (`DXGI_ERROR_DEVICE_HUNG`) instead of returning an error, because the shader
  reads weight data from that root SRV. Passes whose weights are embedded
  constants (1/2/12) do not expose this; pass 6 reads 9,856…18,048 B of the
  initializer payload.

## Interpretation

- [M] Two of the six missing operator families now execute as our own code with
  byte-exact results against the shipped reference, on hardware, with native
  WMMA confirmed. This completes the downscale-convolution part of C04.
- [S/H] The epilogue finding generalizes the earlier ConvNextBlock result in
  [the CNB rounding finding](../findings/2026-10-27-fsr4-cnb-rounding.md):
  mathematical equivalence is not evidence of quantization-boundary exactness,
  and the *compiler-folded* form — not the source expression — is the contract.
  Every remaining operator family must be checked the same way before its
  comparison is trusted.
- [S] The reproduction predicates all stores to `logicalSize`; these passes
  already do so in the reference.

## Reproduction

```powershell
cmake --build --preset debug
build/bin/rdna3-int8.exe run fsr4-downscale
# optional captures under build/reports/m12/downscale/
$env:M12_DUMP = "1"; build/bin/rdna3-int8.exe run fsr4-downscale
```

Captures: `pass{3,6}-fixture{0,1}-{input,own,ref}.bin`, plus each pass's own
GEMM ISA (`pass{3,6}.isa`) and the shipped pass's ISA
(`reference-pass{3,6}.isa`).

## Files

- `src/inference/models/fsr4_downscale.{h,cpp}`
- `src/inference/models/shaders/fsr4_downscale.hlsl`
- `src/gpu/shaders/conv_i8.hlsli` (grouped output scale, folded epilogue)
- `src/experiments/fsr4_reproduction.cpp` (`run fsr4-downscale`)
- `scripts/m12_pack_model.py` (`M12_P{n}_EPI_SCALE{0,1}`)
- `CMakeLists.txt` (per-pass pack/GEMM/requant CSOs, `experiment_fsr4_downscale`)

## Open work

- FasterNetBlock (passes 4/5/7/8/10).
- FNB_CT2D_ADD (passes 9/11), including the pass-11 overhang contract.
- CNB_CT2D (pass 13), including its FP16 output boundary.
- Full GPU-resident graph execution and per-pass/end-to-end comparison
  (C07–C09).
