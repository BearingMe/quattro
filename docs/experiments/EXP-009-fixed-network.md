# EXP-009: Fixed two-layer INT8 network on gfx1101

Date: 2026-10-27 · Milestone: M11

## Question

Can a complete quantized neural network execute end-to-end on gfx1101 with
native INT8 WMMA — GPU-resident intermediates, every tensor compared against an
independent CPU reference, deterministic repeated execution, and recorded
end-to-end GPU timing — using a small fixed network independent of FSR 4?

## Fixed graph (C01)

- [S] A batch of 16 signed-INT8 vectors, each 16 channels, passes through two
  16×16 signed-INT8 dense matrices with INT32 accumulators. Each layer applies
  nearest-even division by 16 and signed saturation; the first additionally
  applies ReLU. This is a **handcrafted test network**, not a trained model or
  an FSR graph.
- [S] Symmetric quantization, zero point 0. The accumulator-to-output factor is
  exactly 1/16, baked as `GEMM_SIN=1.0f, GEMM_SW=1.0f, GEMM_SOUT=16.0f` —
  exact powers of two, so requantization is exact integer division with no
  fp32-contract ambiguity (acc ≤ 16·127·127 < 2^18 is exactly representable).
  This isolates network-execution correctness from the open fused-ISA rounding
  questions recorded in EXP-007/EXP-008. **This exactness is a property of the
  M11 power-of-two bake, not of the requant kernel generally** — non-power-of-
  two scales (e.g. FSR's real scales) make no bit-exactness claim. Weight
  layout is K-major `weights[k*16 + outputChannel]`, matching the WMMA GEMM B
  operand.
- [S] The fixed shape is one native WMMA tile. Buffers: `[A input | B w1 |
  zeroTile]`, `[acc1 | fp16 bias zeros]`, `[A hidden | B w2 | zeroTile]`,
  `[acc2 | fp16 bias zeros]`, output. The first requantization writes the
  hidden activation directly into the second GEMM's A region — no copy between
  layers. Hidden/output snapshots are read back only for comparison
  (inspection), never re-uploaded for computation.
- [S] `reference/fixed_network.cpp` is the independent oracle: INT64 dot
  products, exact integer nearest-even division (magnitude-based, verified
  equivalent to signed ties-to-even), INT8 saturation, ReLU on the hidden
  layer. `m11-reference` verifies diagonal/reversal hand-cases, tie anchors
  (8/24/40/-8/127 × acc=1) and positive/negative saturation.

## Method

1. [S] **Weight loading (C03)**: deterministic LCG (seed 0xC0FFEE11) produces
   input and both weight matrices in [-100,100]. The contract is validated
   before GPU submission: sizes (static), signed INT8 domain, and
   `core::ValidateInt8Accumulation(16, 0)` (reused rejection path). FNV-1a
   variant hashes (64-bit, **non-standard offset basis** 1469598103934665603 —
   the standard basis is 14695981039346656037; hashes are code-internal) make
   the staged data checkable:
   `in=abd0a96612b7f555 w1=db1d8a00f9020547 w2=a002528721fef137`.
2. [S] **GPU-resident execution (C04)**: new `gpu::RunDispatchSequence`
   records all four stages (GEMM → requant+ReLU → GEMM → requant) in one
   command list over default-heap buffers — upload once, no CPU roundtrip
   between layers, UAV barrier between stages, explicit state transitions.
   Analyzer PSOs (M09 pattern) provide ISA evidence; the GEMM CSO must contain
   `v_wmma_i32_16x16x16_iu8`.
3. [S] **Comparisons (C05/C06)**: after warmup networks, snapshots of acc1,
   hidden, acc2 and the output are read back and compared element-wise against
   the CPU reference; the output is additionally read back after the timed
   loop. Mismatch-free is required — exactness holds by construction
   (INT32-exact WMMA accumulation, exact power-of-two requantization).
4. [S] **Repetition (C08)**: the timed loop repeats the complete network
   100 times with a GPU timestamp pair around each execution (dispatches +
   inter-stage barriers only; upload/readback excluded). The timed output
   snapshot must equal the warmup snapshot. A full preliminary run is
   discarded (cross-run clock state — M07/M09 methodology).
5. [S] **Timing (C07)**: min/avg/max over the 100 timed executions.
6. [S] Debug layer + GPU-based validation on; device-removal is checked
   explicitly after every sync (a removal surfaces as silently-zero readbacks
   otherwise — this actually happened, see Negative evidence). The known
   GBAV intrinsics-dummy filter (EXP-004) is the only tolerated diagnostic.

## Result

[M] **PASS on hardware — both Debug and Release configurations.**
Exit 0; final rebuilt CTest **11/11 on both presets** (127.42 s Debug, 128.07 s
Release); direct run log `build/reports/m11/debug-network.log`.

```
kernel gemm_raw_m16k16n16       | wmma x1 | vgpr 24 | ISA saved
kernel gemm_requant_m16n16      | wmma x0 | vgpr 14 | (runtime requant, no WMMA)
kernel gemm_requant_m16n16_relu | wmma x0 | vgpr 14 | (adds [0,127] clamp)
acc1       | 0 mismatches vs CPU INT64 reference
hidden     | 0 mismatches vs CPU reference
acc2       | 0 mismatches vs CPU INT64 reference
output     | 0 mismatches vs CPU reference (warmup == timed snapshot)
network 2x(16x16x16 INT8 WMMA + requant) | min 40.68 us | avg 46.68 us |
  max 60.44 us over 100 timed executions (3 warmup), Debug
diagnostics: no unexplained error-severity messages (64 known-benign
  GBAV intrinsics-dummy reports, see EXP-004)
```

- [M] Performance baseline: **~41 µs end-to-end** per two-layer network
  execution (min of 100, Debug), dominated by dispatch/barrier overhead at
  this single-tile size — the baseline M12+ must beat, not a kernel ceiling.

### Negative evidence and repairs (debugging trail)

1. [M] **Silent device removal read back as zeros**: the first version of the
   sequence runner staged every buffer's initial data into ONE upload region
   reused per buffer; the recorded GPU copies each read upload offset 0, and
   later memcpys overwrote earlier data before execution. The GPU faulted
   (`CopyBufferRegion` out of bounds) — but the fence signalled first, so the
   sync "succeeded", readbacks mapped fine and contained zeros, and the first
   failure looked like "WMMA produced zeros". Repaired with per-buffer upload
   regions and an explicit `GetDeviceRemovedReason` check after every sync.
2. [M] **M09 benchmark spread gate became flaky under CTest scheduling**
   (observed twice while validating M11: Debug 35% spread; Release 10.6%).
   Root cause is measurement methodology, not M11 code: the discarded warm
   cycle was too short to reach steady-state clocks when other GPU experiments
   ran first, and min-of-100 (≈4.5 ms) is a shallow estimator. Repairs, gate
   unchanged at 10%: the discard cycle now runs 1000 timed dispatches
   (~0.2 s sustained load) and measured runs use min-of-1000. Both presets
   then passed 11/11. EXP-007's historical numbers remain valid as recorded;
   current methodology supersedes them for future re-measurement.

## Interpretation

- [M] All seven M11 gate items hold: the entire fixed network executes on
  gfx1101; every layer runs (ISA evidence + per-layer comparisons); all
  intermediate tensors have correct shapes; the final output matches the CPU
  quantized reference exactly; no unexplained overflow or corruption
  (accumulation guard enforced, GBAV clean, removal-checked); execution is
  deterministic across 100+ repeated executions; end-to-end GPU timing is
  recorded.
- [S] This validates the fixed execution graph, weight contract and GPU-resident
  sequencing path — still a handcrafted network, not an FSR 4 model. No graph
  parser, allocator or model runtime was introduced.

## Next

- M12 (Phase V) reproduces the FSR 4 graph on top of these validated
  components; temporal resources and image pre/post remain FSR-specific work.

## Files

- `src/gpu/sequence.{h,cpp}` (dispatch-sequence runner), `src/gpu/d3d12util.h`
- `src/gpu/shaders/gemm_i8.hlsl` (+`GEMM_RELU` bake; 3 new CSO shapes)
- `src/reference/fixed_network.{h,cpp}` (independent CPU oracle)
- `src/experiments/fixed_network_inference.cpp` (the gate),
  `fixed_network_reference.cpp` (C01/C02), `src/main.cpp`, `CMakeLists.txt`
- Evidence: `build/reports/m11/` (ISA, logs, input/weights/hidden/output
  tensors), `build/reports/m11/{debug,release}-ctest.log`
