# EXP-015: GPU-resident FSR 4 passes 0 -> 1

Date: 2026-10-28
Milestone: M12, incremental C06/C07 evidence; not the full gate.

## Question

Can the independently verified operators connect on the GPU without rewriting
their shaders around AMD's single scratch allocation?

## Method

- [S] Reference: pinned FSR 4.0.2 `fsr4_model_v07_i8_quality`, 1080p,
  `FSR-4.0.2-reference` @ `c5e34b4`; not FSR 4.1.
- [S] Extracted append-only dispatch assembly from the input convolution and
  ConvNextBlock wrappers. Their isolated experiments and the chain share those
  same builders. `DispatchPlan` retains buffer descriptors and PSO lifetimes;
  the existing generic `gpu::RunBufferPasses` remains the sole executor.
- [S] Pass 0's output is a separate 8,294,400-byte GPU resource. Pass 1 reads it
  directly, including its residual through the existing `u1` binding. Packing,
  WMMA accumulators, and activations use separate workspace resources. No HLSL,
  arithmetic constants, resource-alias validation, or diagnostic filters changed.
- [S] One upload region per resource and one command-list submission contain
  pass 0 plus pass 1's nine pack/GEMM/post dispatches. The runner transitions
  each resource between roles; no CPU output is used as a later pass's input.
  Workspace zero-initialization is still uploaded by the existing runner; this
  experiment makes no transfer-volume or performance claim.
- [S] The reference also executes shipped passes 0 and 1 in one submission,
  using their original scratch offsets, dispatch grids and initializer binding.
  Rounded-grid input/output tail storage is retained. Only logical extents are
  compared; there is no predication control involved in these two passes.
- [S] Fixtures follow `fsr4-pass0`: zero, a central signed impulse, and
  deterministic FP16 features in approximately [-2, 2], seven populated channels
  with channel 7 and the padded tail zeroed (LCG seed `0x5EED4F12`).
- [S] A diagnostic run snapshots pass 0 without feeding that readback back into
  execution. A second run disables intermediate readback completely. Both are
  compared at the final output; three WMMA ISA strings are checked for
  `v_wmma_i32_16x16x16_iu8`. Tensor diagnostics remain fatal; only the existing
  exact named root4 intrinsic diagnostic is exempted.

## Result

- [M] RX 7700 XT/gfx1101, Debug, `run fsr4-chain`: **PASS**. All three fixtures
  show **0/8,294,400 differing bytes** at pass 0 and **0/8,294,400** at pass 1;
  max INT8 delta 0. Snapshot-free repeats are identical for every fixture.
  Native WMMA confirmed for all three convolutions. Six exact named root4
  intrinsic messages; no other validation errors accepted.
- [M] Initial log: `build/temp/m12-chain-debug.{log,err}`. Full regression
  results are recorded below when available.

## Interpretation

- [S] The previous integration blocker was a proposed allocation layout, not
  an inability of the runner to connect operators. Separate live resources
  already permit outputs to become inputs of later dispatches without SRV/UAV
  aliasing within a dispatch. A shared scratch allocation and absolute shader
  offsets are **not prerequisites** for this correctness-first route.
- [M] This establishes GPU-resident composition for **2/14 model passes** only.
  Full-graph C06/C07 and comparisons C08/C09 remain incomplete; the M12 gate
  remains **UNVERIFIED**.
- [U] Fused upscale input/skip concatenation is still unresolved for full graph
  assembly. Candidate next step: GPU-side concatenation copies or a separate
  read binding, not CPU round-trips. Neither is claimed implemented here.

## Reproduction

```powershell
cmake --build --preset debug
build/bin/rdna3-int8.exe run fsr4-chain
ctest --preset debug
cmake --build --preset release
ctest --preset release
```

Optional `M12_DUMP=1` writes logical boundary captures to
`build/reports/m12/chain/fixture{0,1,2}-pass{0,1}-{own,ref}.bin`.
The normal path reads back only the chain's final output.

## Next

Extend assembly through the encoder, validating each new boundary before
adding decoder skip connections. Keep distinct live tensors first; defer
single-scratch allocation and memory reuse. Do not mark the full gate PASS
from this two-pass checkpoint.

## Addendum: integration completed

[M] The first full-suite attempt after this checkpoint was cancelled by a
server restart, not passed. Complete replacement Debug/Release suites now
pass **22/22** (1750.60 / 1866.75 s), including this chain and the complete
graph. [EXP-016](EXP-016-fsr4-resident-graph.md) supersedes the open integration
work above: GPU-side skip concatenation and all 14 passes are now validated.
