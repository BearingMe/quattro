# EXP-014: CNB_CT2D (model pass 13) and the completed operator set

Date: 2026-10-28 · Milestone: M12, C04 evidence

Hardware: RX 7700 XT/gfx1101, AMD driver 32.0.32015.7029. Context:
[EXP-010](EXP-010-fsr4-model-graph.md) (reference), [EXP-013](EXP-013-fsr4-fasternet.md)
and [EXP-012](EXP-012-fsr4-downscale.md) (preceding operator families).

## Objective and scope

Reproduce the last operator of `fsr4_model_v07_i8_quality` — pass 13,
`float16_NHWC/Fused/CNB_CT2D` — and confirm that every one of the 14 model
passes now has an independent, byte-exact reproduction. This does **not** yet
execute the whole graph GPU-resident in one submission, which is C07.

## Method

- [S] Pass 13 is the ConvNextBlock of the bottleneck fused with the 2x2
  transposed upscale that produces the network's FP16 1920×1080×8 output:
  3×3 depthwise partial conv (16 channels), 1×1 expand + ReLU (16→32), 1×1
  contract (32→16), residual add, then the upscale.
- [S] Unlike every other pass it writes to a separate UAV (`u0`), not to the
  scratch image, so the isolated reference needs its own path: the scratch holds
  the input at offset 0 and the FP16 result is read from `u0`.
- [S] The upscale output is narrowed to FP16 **before** the bias is added, and
  the add happens in FP16 arithmetic: `half4(acc * fl(cnbOutputScale ·
  upscaleWeightScale)) + half4(bias)`. Reproducing that order matters; adding
  the bias in float and narrowing afterwards would round differently.
- [M] The shipped store is unpredicated and overruns the logical output by 8
  rows (245,760 B). The harness reserves that documented tail so the reference
  runs unmodified, and reads back only the logical extent. The reproduction
  predicates to the logical extent.
- [S] Fixtures: all-zero INT8 and a deterministic LCG stream over the full
  signed INT8 range.

## Results

- [M] `run fsr4-cnb` — pass 13: **0/16,588,800 halfword mismatches** on both
  fixtures, max |Δ| 0, repeated execution bit-identical.
- [M] With passes 0, 1, 2 and 12 already covered by EXP-011, this completes the
  current C04 evidence: **every one of the 14 passes** has an independent
  reproduction that is byte-identical to the shipped pass it replaces.

  | Family | Passes | Evidence |
  |---|---|---|
  | input conv, ConvNextBlock | 0, 1, 2, 12 | EXP-011 |
  | quantized downscale conv | 3, 6 | EXP-012 |
  | FasterNetBlock | 4, 5, 7, 8, 10 | EXP-013 |
  | FNB_CT2D_ADD | 9, 11 | EXP-013 |
  | CNB_CT2D | 13 | this experiment |

  [S] Pass 11 is compared against the labelled predication control; every other
  pass is compared against the unmodified shipped pass.

## Interpretation

- [M] The operator set is complete and independently validated. What remains for
  the M12 gate is integration, not new operators: executing the 14 passes in one
  GPU-resident command stream (C07), then per-pass and end-to-end comparison
  against the reference graph with the differences recorded (C08/C09).
- [S] The recurring result across all five families — the emitted, folded
  arithmetic is the contract — is the main durable finding of this milestone.
  See [the epilogue finding](../findings/2026-10-28-fsr4-quantized-output-epilogue.md).

## Reproduction

```powershell
cmake --build --preset debug
build/bin/rdna3-int8.exe run fsr4-cnb
$env:M12_DUMP = "1"; build/bin/rdna3-int8.exe run fsr4-cnb
```

Captures under `build/reports/m12/cnb-ct2d/`: `reference-pass13.isa` and
`pass13-fixture{0,1}-{input,own,ref}.bin`.

## Files

- `src/inference/models/fsr4_cnb_ct2d.{h,cpp}`
- `src/inference/models/shaders/fsr4_cnb_ct2d.hlsl`
- `src/experiments/fsr4_reproduction.cpp` (`run fsr4-cnb`, pass-13 reference)
- `scripts/m12_pack_model.py` (`M12_P13_CT2D_SCALE`)
- `CMakeLists.txt` (pass-13 CSO)

## Open work

- C07: one GPU-resident pass sequence over a single scratch image.
- C08/C09: per-pass and end-to-end numerical comparison, recorded differences.
