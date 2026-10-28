# EXP-016: Complete GPU-resident FSR 4 INT8 model reproduction

Date: 2026-10-28 · Milestone: M12, C06–C09 and validation gate.

## Question

Can all 14 independently reproduced operators execute as one GPU-resident
graph, with every logical intermediate and final output compared exactly?

## Method

- [S] Reference: FSR 4.0.2 `fsr4_model_v07_i8_quality`, 1080p, pinned tree
  `FSR-4.0.2-reference` @ `c5e34b4`. Public generated non-WMMA reference path;
  not FSR 4.1 or signed-runtime equivalence. Pre/post-processing and temporal
  integration are outside M12.
- [S] `inference::fsr4::RunModelGraph` assembles the fixed graph through the
  same append-only builders used by isolated operator experiments. One command
  list contains **42 compute dispatches**: input conv (1), three ConvNextBlocks
  (27), two downscales (6), seven FasterNet/fused-upscale blocks (7), final
  CNB_CT2D (1). Uploads occur before execution, never between model passes.
  The normal path reads back only the 33,177,600-byte final FP16 output.
- [S] Distinct live tensor resources replace the proposed single-scratch
  layout. Pass 9 retains pass 5's output as its skip; pass 11 retains pass 2's.
  Four GPU-only `CopyBufferRegion` operations assemble the two fused kernels'
  `[input | skip]` source buffers immediately before their dispatches. No HLSL
  or requantization constants changed for this integration.
- [S] Generic runner additions are bounded pre-dispatch buffer copies and
  per-dispatch snapshot-resource selection. Copy ranges use subtraction-based
  overflow-safe checks; same-resource copies remain unsupported. SRV/UAV
  aliasing within a dispatch remains rejected.
- [S] C06 checks combine `m12_tensor_metadata`'s extracted logical shapes,
  producer lifetimes and group-scale chain with builder input-size guards and
  observed logical snapshot byte counts at all 14 model boundaries. Snapshots
  are after model passes, not after every packing/GEMM micro-dispatch.
- [S] Fixtures: zero, central signed impulse, deterministic FP16 features in
  approximately [-2,2]; seven populated channels, padded channel/tail zero,
  LCG `seed=0x5EED4F12`, `seed=seed*1664525+1013904223`,
  `value=((seed>>16)%4095-2047)/1024`. Same recipe as EXP-015/`fsr4-pass0`.
- [S] For each fixture, run our graph with all logical boundary snapshots,
  repeat with **no intermediate readback**, then run the reference with only
  the labelled pass-11 store predication correction. Exact byte comparison;
  no widened tolerance. On the deterministic-feature fixture also run the
  **unmodified shipped graph** and measure its differences separately.
- [S] Pass-0 input padding, scratch +1,024-byte tail, and pass-13 +8 output rows
  are reserved in the reference harness. The reproduction predicates logical
  stores. See EXP-010 for the reference race and causal control; it is never
  silently relabelled as shipped behavior.
- [S] CTest forces `M12_LAST_PASS=13` for the gate experiment; a developer's
  prefix-debugging environment cannot silently turn it into a partial test.
  `fsr4-chain` remains the original small 0->1 regression.

## Result

Hardware: RX 7700 XT/gfx1101; driver **32.0.32015.7029**.

- [M] Encoder checkpoint (`M12_LAST_PASS=8`): all nine boundaries exact on
  all three fixtures, snapshot-free repeats identical, native WMMA **8/8**.
  Log: `build/temp/m12-encoder.{log,err}`.
- [M] Full graph, Debug **and** Release: **0 differing bytes at every one of
  14 boundaries on all three fixtures**, against the predication-control
  graph. Through pass 10 these are also the unmodified shipped results.

  | Passes | Logical bytes per tensor | Different bytes vs control |
  |---|---:|---:|
  | 0, 1, 2, 11, 12 | 8,294,400 | 0 |
  | 3, 4, 5, 9, 10 | 4,147,200 | 0 |
  | 6, 7, 8 | 2,073,600 | 0 |
  | 13 | 33,177,600 (16,588,800 halfwords) | 0 |

- [M] Snapshot-free repeats are identical in both configurations, all fixtures.
  **11/11** native `v_wmma_i32_16x16x16_iu8` convolution ISA strings confirmed.
  Fused DP4A families remain DP4A as documented in EXP-013/014.
- [M] Final outputs are finite/nonzero. Exact FP16 ranges:

  | Fixture | Nonfinite | Nonzero halfwords | Min | Max |
  |---|---:|---:|---:|---:|
  | zero | 0 | 16,588,677 | -1.833984375 | 9.6328125 |
  | impulse | 0 | 16,588,676 | -10.5078125 | 9.7109375 |
  | deterministic features | 0 | 16,588,402 | -25.65625 | 39.46875 |

- [M] Deterministic-feature final SHA-256 in both configurations:
  `a57bcd063967affddf29cc24709b9d8a7fd9f8e7244e7010521d7b27f605b6de`.
  This also matches the independently recorded predicated-control hash in
  EXP-010, not just the new comparison harness.
- [M] Unmodified shipped graph comparison, deterministic-feature fixture:

  | Boundary | Debug differing units | Release differing units | Unit | Known band | Outside band |
  |---|---:|---:|---|---|---:|
  | 0–10 | 0 | 0 | INT8 bytes | none | 0 |
  | 11 | 265,774 | 271,411 | INT8 bytes | x<64 | 0 |
  | 12 | 454,093 | 464,539 | INT8 bytes | x<66 | 0 |
  | 13 | 1,129,213 | 1,133,780 | FP16 halfwords | x<132 | 0 |

  Final differing **bytes** are 1,795,125 / 1,811,082; these are not the
  halfword counts above. Maximum final absolute difference is
  **17.24609375 / 18.588623046875**. Pass-11 max INT8 delta is 122 in both
  captures; pass-12 max delta is 92 / 73. Race-dependent counts are observations
  of these captures, not a stable corruption rate or cross-configuration error.
- [M] Full Debug CTest: **22/22 PASS, 1750.60 s**. Full Release CTest:
  **22/22 PASS, 1866.75 s**. The full-graph test took 891.41 / 904.79 s.
  These include all prior isolated operator checks after dispatch extraction,
  the small chain, metadata, weights, and capture-integrity tests.
- [M] New fail-closed checks reject eight invalid copy descriptors (indices,
  same-resource, zero length, overflow/out-of-range offsets and extents), an
  invalid snapshot index, and illegal SRV/UAV aliasing before submission.
  Existing exact named root4 intrinsic filtering is unchanged; the graph
  experiment records six such messages in each configuration. Other diagnostics
  are fatal; no validation was disabled to pass the gate.

## Evidence and re-comparison

- [M] Verbose logs: `build/reports/m12/integration/ctest-{debug,release}.log`.
- [M] Logical captures: `build/reports/m12/graph-captures/` directories
  `resident-debug-final` and `resident-release-final`, with
  `fixture{0,1,2}-pass{0..13}-{own,control}.bin`, and fixture-2 shipped tensors.
- [M] Capture-time executable, CSO, weight and source SHA-256 ledgers:
  `integration/{debug,release}-identities.json`, recorded before switching the
  shared executable to the next configuration. These are collector ledgers,
  not signed attestations or an automated historical-provenance proof.
- [M] `integration/{debug,release}-numerics.json`: independent offline
  re-comparison, per-tensor SHA-256, exact byte/halfword counts, deltas, bands,
  finiteness and the prior control hash. Both report `numeric_pass=true`.
  The reporter binds each supplied ledger by hash without pretending to
  revalidate historical execution after `build/bin/rdna3-int8.exe` changes.

```powershell
cmake --build --preset debug
$env:M12_DUMP = "1"
$env:M12_RUN_ID = "choose-a-new-debug-id"
ctest --preset debug -j1 -V
Remove-Item Env:M12_DUMP
Remove-Item Env:M12_RUN_ID
# Retain logs and record Get-FileHash SHA256 for the executable, weights,
# source and all m12/m12ref/m12control CSOs BEFORE building Release.
cmake --build --preset release
ctest --preset release -j1 -V

# Recompare retained final Debug captures (writes a NEW report only):
python -B scripts/m12_report_graph.py `
  build/reports/m12/graph-captures/resident-debug-final `
  --identities build/reports/m12/integration/debug-identities.json `
  --output build/reports/m12/integration/debug-numerics-recheck.json
```

[S] Explicit capture directories are created exclusively. Existing IDs fail
instead of overwriting evidence. Interrupted suites are not complete suites;
the earlier server-restart cancellation is superseded by the two complete
runs above, not counted as a pass.

## Interpretation and gate

- [M] **M12 validation gate PASS**, for the pinned generated INT8 network at
  1080p on gfx1101, with the pass-11 predication deviation explicitly recorded.
  The reproduction is exact against the labelled corrected contract; it is
  **not byte-identical to the raced, unmodified graph inside its corruption band**.
- [S] All operators and metadata are accounted for, weights checked, complete
  graph GPU-resident, all logical boundaries validated, finite output observed,
  both reference comparisons and differences recorded. No substitute weights,
  tolerance relaxation, or ignored tensor validation errors.
- [S] This does not establish image quality, temporal integration, speedup,
  signed FSR runtime equivalence, alternate resolutions, or FSR 4.1 behavior.
  Single-scratch allocation and memory reuse are deliberately deferred.

## Next

Stop at the M12 gate. M13 requires explicit authorization. Before touching
these paths again, read the [blocker-prevention finding](../findings/2026-10-28-fsr4-integration-blockers.md).
