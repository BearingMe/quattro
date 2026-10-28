# EXP-011: checked FSR 4 weights and independent input/ConvNext operators

Date: 2026-10-27 · Milestone: M12, incremental C04/C05/C06 evidence

Hardware: RX 7700 XT/gfx1101, AMD driver 32.0.32015.7029; inventory
`build/reports/m12/hardware.json`.

## Objective and scope

Implement real operations from `fsr4_model_v07_i8_quality` without executing the
shipped operators as the reproduction. The pinned graph/oracle is defined in
[EXP-010](EXP-010-fsr4-model-graph.md). This experiment covers **passes 0, 1, 2
and 12**, not the complete 14-pass model or the M12 gate.

## Method

- [S] `scripts/m12_pack_model.py` checks pinned source/initializer SHA-256,
  packs the complete 89,216-byte initializer file and all 67 embedded literal
  arrays verbatim, resolves 78 weight tensors and captures operator-call
  numeric arguments. No weight conversion or requantization occurs.
- [S] Payload: **136,848 B**, SHA-256
  `1ce84b7fe0287835030ff8d0e4cf7f86d205340bad83306a6ae91533594cddf9`.
  Generated payload, C++/HLSL tables and manifest live in `build/reports/m12/`.
  CMake regenerates them offline; no Python dependency exists at runtime.
- [S] `inference::fsr4::LoadWeights` checks exact size before allocation,
  full-payload FNV-1a integrity, all descriptor ranges, physical/logical shapes,
  strides and positive finite scales. FNV detects accidental corruption, not
  adversarial tampering. Source identity is checked with SHA-256 offline.
- [S] `scripts/m12_validate_graph.py` validates all 14 expected logical output
  shapes, scratch bounds/lifetimes, skip producers and scalar/group scale
  chaining. Output: `tensor_validation.json`. This validates metadata, **not**
  the behavior of the ten reproduction passes not yet implemented.
- [S] Pass 0 uses our own HLSL: FP16 operand pairs, FP32 `dot2add`, FP16 bias,
  source-compatible rounding/narrowing and saturating INT8 stores. All stores
  are restricted to logical output pixels.
- [S] Each ConvNextBlock is a fixed nine-dispatch GPU-resident sequence:
  im2col/weight packing → existing native INT8 WMMA GEMM → our quantization
  epilogue, three times, including ReLU and the original input residual. No
  intermediate CPU processing occurs inside a block. Fixed data/scale selection
  belongs to `src/inference/models/`; packing/epilogue primitives stay generic
  in `src/gpu/shaders/conv_i8.hlsli`.
- [S] Comparisons execute an isolated shipped reference pass on identical
  inputs, not another experiment implementation. Native instruction evidence
  is obtained from the analyzed PSOs, not inferred from HLSL names.

## Results

- [M] `fsr4-weights`: accepted the complete pinned payload; rejected corrupted,
  truncated, extra-byte and missing files. Six offline packer regression tests
  pass, including deterministic regeneration, all array hashes and invalid
  identities/ranges/scales.
- [M] `fsr4-pass0`: **0/8,294,400 differing output bytes** on each zero,
  impulse and deterministic LCG fixture. Repeated execution is bit-identical.
- [M] `fsr4-cnb`: final implementation has **0/8,294,400 differing bytes** for
  each zero and full-range INT8 LCG fixture at passes **1, 2, 12**. Native
  `v_wmma_i32_16x16x16_iu8` is present in every convolution's captured ISA.
  Retained targeted Debug log: `build/reports/m12/cnb-mad-debug.log`.
- [M/R] Earlier variants failed by 6–95 values per pass due to epilogue
  code-generation differences. They were retained, investigated with ISA, and
  fixed without a numerical tolerance. Details and negative logs:
  [CNB rounding finding](../findings/2026-10-27-fsr4-cnb-rounding.md).

[M] Final verbose **Debug: 18/18 PASS, 286.60 s**, including three repeated
native ConvNext executions, each bit-identical, all exact comparisons and
capture-integrity regressions. Log: `build/reports/m12/debug-final-ctest.log`.
`debug-validation-manifest.json` records executable/CSO/payload/log identities
before the shared `build/bin/` is rebuilt for Release.

[M] Final verbose **Release: 18/18 PASS, 281.94 s**, including the same six
exact CNB comparisons and three bit-identical repeat checks. Log:
`build/reports/m12/release-final-ctest.log`; artifact identities:
`release-validation-manifest.json`, recorded immediately after testing, before
any further rebuild. Final adversarial review found no checkpoint blockers,
conditional on these validation results. No full-model success is inferred
from these operator fixtures.

## Reproduction

```powershell
cmake --build --preset debug
build/bin/rdna3-int8.exe run fsr4-weights
build/bin/rdna3-int8.exe run fsr4-pass0
build/bin/rdna3-int8.exe run fsr4-cnb
python -B scripts/m12_test_pack.py
python -B scripts/m12_validate_graph.py
```

## Open work

- FasterNetBlock (passes 4/5/7/8/10).
- FNB_CT2D_ADD (passes 9/11), predicated stores and skip arithmetic.
- CNB_CT2D (pass 13), including its intermediate FP16 rounding boundary.
- Full GPU-resident graph execution and per-pass/end-to-end comparison (C07–C09).

Downscale convolutions (passes 3/6) are covered by
[EXP-012](EXP-012-fsr4-downscale.md).
