# FSR 4 v07 ConvNextBlock: code generation affects INT8 boundary exactness

Scope: gfx1101/RX 7700 XT, pinned v07 INT8 Quality 1080p, DXC from Windows SDK
10.0.26100.0, AMD driver **32.0.32015.7029** (CIM inventory retained as
`build/reports/m12/hardware.json`). See [EXP-011](../experiments/EXP-011-fsr4-reproduction-operators.md).

## Measured progression

The independent implementation decomposes each ConvNextBlock into GPU packing,
native INT8 WMMA/INT32 GEMM and requantization, repeated for its three
convolutions. All comparisons use the same input bytes as the isolated shipped
reference pass; no corrupted upstream tensor is involved.

| Variant | Random input mismatches, passes 1 / 2 / 12 | Maximum INT8 delta |
|---|---|---|
| Dynamic bias loads; dword-owned epilogue | 28 / 88 / 95 | 2 / 1 / 2 |
| Generated constant bias words; pixel-owned unrolled epilogue | 8 / 10 / 6 | 1 |
| Explicit `mad` with `precise` residual value | 7 / 6 / 8 | 1 |
| Explicit `mad` without `precise` | **0 / 0 / 0** | **0** |

[M] Each random fixture contains 8,294,400 signed INT8 values, including spatial
boundaries and the full signed range. Zero fixtures matched in all variants.
The final variant emits `v_wmma_i32_16x16x16_iu8` for all three convolutions.

Evidence (generated, retained under `build/reports/m12/`):

- `cnb-first-failure-debug.log`
- `cnb-baked-bias-debug.log`
- `cnb-isa-debug.log` (mismatch coordinates and values)
- `cnb-explicit-fma-debug.log`
- `cnb-mad-debug.log`
- `cnb/reference-pass{1,2,12}.isa`, `cnb/post-pass*conv*.isa`, `cnb/pass*conv*.isa`

## Explanation and limits

- [S/M] Integer convolution shapes and weight bits did not change between
  variants. The changed implementations are the bias placement, epilogue
  ownership/unrolling, and floating-point expression tree. The initial two
  changes were tested together; their individual contributions are **not**
  isolated.
- [R] The driver ISA for the shipped reference folds constant first/second
  convolution scale/bias arithmetic into multiply-add instructions. Its final
  residual arithmetic uses an accumulator/scale/bias multiply-add followed by
  a skip/scale multiply-add, then the output reciprocal and RNE conversion.
- [R] The ordinary independent epilogue permits reassociation of bias with the
  skip input on some channels. Algebraically equivalent expressions need not
  round identically near half-integer quantization boundaries.
- [R] In this toolchain, `precise` on the value caused even explicit HLSL `mad`
  expressions to lower to separate multiply/add instructions in the inspected
  ISA. Removing `precise` while retaining explicit `mad` dependencies restored
  the reference-like multiply-add chain and exact measured outputs.
- [H] This is not a universal statement about HLSL `mad`, `precise`, other
  drivers, or all model inputs. Preserve the regression fixtures and inspect
  emitted ISA when changing compilation or epilogue structure. Mathematical
  equivalence is not sufficient evidence for quantization-boundary exactness.

## Diagnostic-policy correction

[M] The first strict CNB run also recorded GBAV's encoded intrinsic access at
root parameter 4, offset 1,342,177,324, against a 16-byte dummy UAV. The legacy
GEMM policy exempts root 2, which is **the real weight SRV in the new signature**
and must never be exempted here.

[S] The new policy requires the exact diagnostic ID, unique dummy resource
name, UAV type, root 4, 16-byte size and observed magic address. Synthetic
regression cases reject root 2, SRV type, real-resource names, different sizes
and non-magic addresses. Other tensor/weight diagnostics remain fatal. The
initial strict failure log is retained; numerical comparison was not weakened.
