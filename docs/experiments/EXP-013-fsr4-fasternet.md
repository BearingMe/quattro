# EXP-013: FasterNetBlock (model passes 4, 5, 7, 8, 10) and FNB_CT2D_ADD (9, 11)

Date: 2026-10-28 · Milestone: M12, incremental C04 evidence

Hardware: RX 7700 XT/gfx1101, AMD driver 32.0.32015.7029. Graph/oracle context:
[EXP-010](EXP-010-fsr4-model-graph.md); epilogue contract:
[finding](../findings/2026-10-28-fsr4-quantized-output-epilogue.md).

## Objective and scope

Reproduce the five FasterNetBlocks and the two blocks that fuse them with a 2x
upscale and add (`FNB_CT2D_ADD`) as our own GPU code, and compare them
byte-for-byte against the isolated shipped passes. Covers **passes 4, 5, 7, 8, 9,
10 and 11 only**, not the complete model and not the M12 gate.

| Pass | Resolution / features | Groups | Operator instantiation |
|---|---|---|---|
| 4 | 480×270×32 | 1 | `FasterNetBlock<32, 1>`, 5 call scales |
| 5 | 480×270×32 | 1 | `FasterNetBlock<32, 1>`, 3 call scales + quantized output tensor |
| 7 | 240×135×64 | 2 | `FasterNetBlock<64, 2>` |
| 8 | 240×135×64 | 2 | `FasterNetBlock<64, 2>` |
| 9 | 240×135×64 → 480×270×32 | 2 | `FNB_CT2D_ADD<64, 2>` |
| 10 | 480×270×32 | 1 | `FasterNetBlock<32, 1>` |
| 11 | 480×270×32 → 960×540×16 | 1 | `FNB_CT2D_ADD<32, 1>` |

## Method

- [S] The shipped operator is a **fused per-pixel DP4A kernel** on this code
  path, not a WMMA one. The reproduction mirrors that structure — one thread per
  output pixel, all features in registers — instead of decomposing into GEMM
  passes, because the block's three convolutions share the same per-pixel
  activation (`conv_result`) and residual halves.
- [S] Structure: grouped 3×3 partial convolution (16 input channels; for the
  64-feature blocks two 16-channel groups feed feature halves), 1×1 expand with
  ReLU, 1×1 contract, then a residual add whose skip scale differs per feature
  half (`quantScale0` below the midpoint, `quantScale1` above).
- [S] Every epilogue constant was derived from the shipped pass's ISA, not from
  the source expression. The first two requantizations fold the output-scale
  reciprocal into the scale and the bias and emit one `v_fmaak_f32`; the third
  keeps an explicit trailing multiply (`v_fmamk_f32` for the skip term plus
  `v_mul_f32` by the reciprocal).
- [M] The folded constants are `fl(fl(a·b) · fl(1/c))` — a multiply by the
  reciprocal, **not** a division. Decoding the ISA scalars proved this: for pass
  7 `fl(fl(s0·w0s)/s1)` = `0x3B76A284` while the shipped scalar is
  `0x3B76A285`, and for pass 8 the E1 scalar is `0x3BE6E686` versus
  `0x3BE6E685` from the division form. Both folded scales are now emitted by
  `scripts/m12_pack_model.py` as `M12_P{n}_EPI0/EPI1/EPI2`.
- [S] Fixtures: all-zero INT8 and a deterministic LCG stream over the full
  signed INT8 range, fed byte-identically to our kernel and the shipped pass.

## Results

- [M] `run fsr4-fnb` — **0 differing bytes on every fixture and every pass**:
  passes 4, 5 and 10 → 0/4,147,200 each; passes 7 and 8 → 0/2,073,600 each;
  pass 9 → 0/4,147,200 (32 output channels at 480×270); pass 11 → 0/8,294,400
  (16 output channels at 960×540). Max delta 0, repeated execution bit-identical.
- [S] Passes 9 and 11 add a 2×2 transposed upscale and a residual add at the
  upscaled resolution. Their weight tensor is `hwcn`: strides are
  (kernel-x, kernel-y, output-feature, input-channel), so 32 contiguous bytes
  are the 32 input channels of one output feature — the opposite of the
  NHWC reading of the same descriptor.
- [S] Pass 11's shipped 32/1 `FNB_CT2D_ADD` overload stores without a
  logical-size predicate and corrupts its own output band (EXP-010). The
  reproduction predicates, so **pass 11 is compared against the labelled
  predication control** (`m12control/pass11`), not the unmodified pass; the
  result is 0 differing bytes against that control. Pass 9's shipped overload is
  already predicated and is compared against the unmodified reference.
- [M] Development history worth recording: an initial version read weights with
  four separate dword loads per 16-byte operand and pushed a single pass past
  the GPU watchdog. Loading `uint4` in one instruction restored normal runtime;
  no numerical change was involved. Passes 9 and 11 matched on their first
  numerical attempt, which is consistent with the arithmetic forms already
  established for the plain blocks.
- [M] The zero fixture was exact from the first implementation; the fuzz fixture
  exposed the epilogue form (28/4,147,200 for pass 4 before the folded scales,
  4 remaining after folding only the scales, 0 after the reciprocal-multiply
  correction). Each step is a measured delta, not an assumption.
- [M] Development history worth recording: an initial version read weights with
  four separate dword loads per 16-byte operand and pushed a single pass past
  the GPU watchdog. Loading `uint4` in one instruction restored normal runtime;
  no numerical change was involved.

## Interpretation

- [M] Five more model passes now execute as our own code, byte-exact against the
  shipped reference on hardware.
- [S] Together with EXP-012 this confirms the general rule for the remaining
  M12 operators: the contract is the *emitted* arithmetic (folded constants,
  reciprocal instead of division, contraction of adds into FMAs), and it must be
  read out of the shipped ISA per pass. Algebraic equivalence is not evidence.
- [S] These passes deliberately use DP4A rather than WMMA: that is what the
  pinned non-WMMA code path does, and the numerical contract is defined by it.

## Reproduction

```powershell
cmake --build --preset debug
build/bin/rdna3-int8.exe run fsr4-fnb
$env:M12_DUMP = "1"; build/bin/rdna3-int8.exe run fsr4-fnb  # optional captures
```

Captures under `build/reports/m12/fnb/`: per-pass reference ISA
(`reference-pass{4,5,7,8,10}.isa`), our kernel ISA (`pass*.isa`), and
tensor dumps `pass*-fixture*-{input,own,ref}.bin`.

## Files

- `src/inference/models/fsr4_fnb.{h,cpp}`
- `src/inference/models/shaders/fsr4_fnb.hlsl`
- `scripts/m12_pack_model.py` (`M12_P{n}_EPI0/EPI1/EPI2`)
- `src/experiments/fsr4_reproduction.cpp` (`run fsr4-fnb`, generic isolated pass)
- `CMakeLists.txt` (per-pass CSOs, `experiment_fsr4_fnb`)

## Open work

- FNB_CT2D_ADD (passes 9/11): the same block plus a transposed 2×2 upscale and
  an add at the upscaled resolution; pass 11 also carries the documented
  unpredicated-store overhang, so its comparison must use the predicated control.
- CNB_CT2D (pass 13), including its FP16 output boundary.
- Full GPU-resident graph execution and per-pass/end-to-end comparison
  (C07–C09).

