# EXP-019: Native inference transfers and equivalent FP16 matrix costs

Date: 2026-10-28 · Milestone: M15.

## Question and scope

What dominates the complete 14-operator experimental 4.0.2 network, and can one
measured optimization improve it without changing its numerical contract?
How does replacing its eleven INT8 matrix products with native FP16/F32 WMMA
change the complete **network invocation** cost?

This measures the EXP-016 neural graph, not the signed provider or a game's
frame time. Temporal preprocessing/reconstruction remain the EXP-017 path;
their regression timestamps are separate, not silently added to these medians.
Neither the fastest possible implementation nor the hardware ceiling is proven.

## Method

- [S] `fsr4-profile` assembles the existing graph once and retains its PSOs for
  the whole process. Network input/output storage is the EXP-016 fixed shape
  (1920x1080 physical layout). All 14 operators, packing, skip copies and
  requantization execute GPU-resident, with no CPU feedback between layers.
- [S] Each invocation still allocates resources, uploads input/weights, executes
  the graph, synchronizes and reads back the final 33,177,600-byte output.
  Wall time includes those host costs, but excludes graph/PSO assembly and
  the subsequent CPU comparison. Assembly is separately printed.
- [S] Timestamp pairs distinguish initial upload, each dispatch, final readback,
  and total GPU span. Dispatch intervals include their pre-dispatch copies and
  barriers; span includes remaining inter-dispatch barriers. Medians of separate
  components need not add to the median span. Snapshot-free measured runs avoid
  intermediate readback distortion.
- [S] Every process retains **two warmups and all seven measured samples** in
  `samples.csv` and every dispatch in `dispatches.csv`. Min/median/max span and
  per-layer medians are printed. No outlier removal or best-sample selection.
  Runs are sequential on RX 7700 XT/gfx1101, driver **32.0.32015.7029**, Release,
  with D3D12/GPU validation enabled. Clocks are not locked; these are instrumented
  research measurements, not production latency or theoretical throughput.
- [S] Before timing, final code compares **all 14 boundaries on three fixtures**
  (zero, signed impulse, seeded deterministic features) against retained
  `build/reports/m12/graph-captures/resident-release-final/fixture*-pass*-control.bin`.
  The pass-11 predication control remains explicitly labelled; it is not the
  unmodified corruption-band output. Every measured final output must also match.
- [S] `BufferInit.initialize` defaults to true. `skip` explicitly disables
  initialization only for profiler workspace buffers without supplied data;
  it does not alter normal runner/temporal-history defaults. `poison` uploads
  0xA5 to that same workspace, checking hidden dependencies on zero-filled bytes.
  An input with supplied data and initialization disabled must fail closed.
- [S] `gpu-zero` uploads one maximum-size zero pool and GPU-copies it into the
  workspace before the first dispatch: a competing, retained candidate.
- [S] `fp16` inserts eleven signed-I8-to-F16 packet conversions and replaces
  eleven products with native `v_wmma_f32_16x16x16_f16`. It rounds the F32
  accumulator to nearest integer before storing I32. All quantized weights,
  DP4A operators, requantization and other epilogues stay identical. This is an
  **equivalent-workload matrix-backend comparison**, not an independently trained
  FP16 model, dequantized-FP16 network, or official FP16 implementation. Conversion,
  widened packets and final I32 conversion are not excluded from the cost.

## Results

### All retained process runs (medians of seven measured invocations)

[M] Times in milliseconds; default-buffer byte sums exclude resource-allocation
alignment, driver-private memory and small intrinsic/query allocations.

| Run | Upload | Dispatch sum | Readback | GPU span | Wall |
|---|---:|---:|---:|---:|---:|
| baseline-release-1 | 26.2958 | 27.1452 | 1.1960 | 55.7648 | 289.605 |
| baseline-release-2 | 27.2670 | 26.7323 | 1.2619 | 57.0631 | 279.652 |
| baseline-boundaries-release-1 | 26.8394 | 27.1788 | 1.1876 | 56.1340 | 256.712 |
| skip-release-1 | 1.1810 | 31.4650 | 1.2023 | 35.0523 | 124.802 |
| skip-release-2 | 1.1791 | 31.2570 | 1.1876 | 35.2076 | 124.626 |
| skip-boundaries-release-1 | 1.1797 | 31.2770 | 1.3155 | 34.8994 | 124.624 |
| poison-release-1 | 26.7179 | 27.3285 | 1.2841 | 56.3266 | 273.727 |
| poison-boundaries-release-1 | 27.0277 | 28.5891 | 1.1940 | 57.8960 | 279.537 |
| gpu-zero-release-1 | 3.9132 | 33.9840 | 1.2206 | 40.5431 | 155.602 |
| fp16-round-release-1 (skip) | 1.1800 | 46.9979 | 1.3423 | 51.7390 | 187.333 |
| fp16-round-release-2 (skip) | 1.1845 | 44.3114 | 1.2154 | 49.6817 | 181.674 |

- [M] Baseline: **48 resources, 42 dispatches, 688,594,376 B** default-buffer
  sum and upload bytes. Skip: same default-buffer sum, **33,314,448 B upload**
  (95.16% reduction). Latest baseline/skip span improves **56.1340→34.8994 ms**
  (37.83% reduction); wall **256.712→124.624 ms** (51.45%). Earlier pairs agree
  on the direction and approximately 35 ms optimized span.
- [M] Dispatch sum becomes **slower** when uploads are omitted (roughly
  27→31 ms); transfer savings dominate. [H] Changed cache/clock state may
  contribute; cache hit rates, bandwidth counters and wave occupancy were not
  measured, so none is claimed as the established cause.
- [M] GPU-zero uses **763,247,304 B** default storage, remains exact on all three
  final fixtures, but loses to skip (40.5431 vs approximately 35 ms). The initial
  workspace copies fall in pass0's timestamp, not upload time; this is not a
  faster convolution. The extra pool/copies are retained as an unsuccessful
  alternative, not adopted by default.
- [M] FP16 uses **59 resources, 53 dispatches, 1,310,737,352 B** default storage;
  upload remains 33,314,448 B with skip. Its repeated 49.68–51.74 ms spans are
  slower than INT8's 34.90–35.21 ms for this implementation. In the second FP16
  run, median summed conversion dispatches cost **2.03672 ms** and WMMA-labelled
  dispatches **32.06908 ms**; INT8 latest WMMA-labelled sum is **19.07976 ms**.
  This does not establish that an optimized FP16 network must be slower.
- [M] The first baseline warmup wall time was **131333.130 ms**, while its GPU
  span was 54.0967 ms. It is retained, not counted as steady-state latency.
  [U] The precise host/driver cold-start cause was not isolated.

### Layer and register evidence

- [M] Baseline's largest layers are pass1 (~8.3–8.7 ms), pass2 (~6.2–6.4 ms),
  pass12 (~6.1–6.2 ms). They are ConvNext matrix blocks. Upload initialization
  was selected first because it consumes nearly half the baseline GPU span.
- [M] The repeat INT8 skip run's layer medians (pass0→13) are:
  `0.533480, 8.768400, 8.006520, 0.900760, 0.395680, 0.388760, 0.963800,`
  `0.726160, 0.724840, 1.263600, 0.403240, 0.635600, 6.919720, 0.681400 ms`.
- [M] FP16 repeat layer medians (pass0→13) are:
  `0.480680, 13.873160, 13.088920, 1.394440, 0.282400, 0.274720, 0.993280,`
  `0.499840, 0.500280, 0.877560, 0.283400, 0.474360, 10.646400, 0.651480 ms`.
- [M] Analyzer reports INT8 raw CNB products **26/32/26 VGPRs** versus FP16
  **35/39/35**. Other analyzed INT8/fused convolution stages reach 72 VGPRs;
  all recorded stages use **0 B/group LDS**. These are compiled register counts,
  **not measured occupancy**; the analyzer's 1536/2048 budget strings are not
  a license to infer resident waves without allocation granularity/slot limits.
- [M] This agrees directionally with [EXP-005](EXP-005-fp16-wmma-comparison.md):
  native FP16 works, but operand staging/register costs matter. M07's small
  tile timing is not substituted for this network-level measurement. Existing
  M07 F32-accumulator probes already showed a tiny deviation from integer-grid
  expectations; exact arithmetic reasoning alone does not prove WMMA bit-exactness.

### Correctness and failed FP16 attempts

- [M] `baseline-boundaries-release-1`, `skip-boundaries-release-1`,
  `poison-boundaries-release-1`, `fp16-round-release-2` each pass **42/42 retained
  boundary comparisons**, plus nine snapshot-free exact final outputs. Info
  queues contain only the previously bounded intrinsic-dummy diagnostic; no
  other validation errors. The contradictory initialization descriptor is rejected.
- [M] Initial `integer.Copy(acc)` fails zero-fixture output at pass1 (pass0 exact).
  Explicit per-element `int(acc.Element(i))` fails identically. Retained runs:
  `fp16-skip-release-1`, `fp16-copy-diagnostic`, `fp16-elements-release-1`.
  Therefore the earlier Copy-specific causal hypothesis is **not established**.
- [M] `fp16-raw-diagnostic-1` compares the first product before its epilogue:
  **3,618,610 differing I32 elements; maximum absolute delta 1**. Examples:
  INT8/F16 `153/152`, `-49/-48`, `-532/-531`, `162/161`, `249/248`.
  The emitted ISA has native FP16 WMMA followed by `v_cvt_i32_f32`.
- [S] DXC's `fptosi` semantics round toward zero:
  [DXC LangRef](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/LangRef.rst),
  accessed 2026-10-28. [M] Explicit `int(round(acc.Element(i)))` restores all
  retained boundaries, including deterministic nonzero features. [U] Raw F32
  residuals/internal WMMA rounding were not separately characterized; the
  experiment establishes the conversion fix, not the precise hardware mechanism
  or exactness for arbitrary tensors. Maintain these regressions when changing it.
- [M] Failed binaries/CSO hashes and failed output captures remain under the
  named run directories; `fp16-copy-diagnostic/failed-csos/` preserves the old
  conversion implementation. ISA/stats now persist **before** correctness tests
  so another early failure cannot erase its diagnostic evidence.

## Validation and reproduction

- [M] Final selected Release CTest **21/21 PASS (790.50 s)**, including temporal
  integration, both WMMA probes, numerical/operator regressions and image metric
  tests. The selected run excludes the old standalone CNB, chain and graph
  experiments because this profiler already exercises their complete graph
  against 42 retained layer controls; it is **not a new full-suite verdict**.
- [M] **M15 bounded network-performance gate PASS**: repeated baseline and
  before/after measurements, exact layer/output controls, accounted transfer and
  conversion costs, equivalent-workload native FP16 comparison, and retained
  unsuccessful candidates. Scope and unmeasured hardware-ceiling/whole-SR
  questions remain as stated above. No new Debug suite verdict is claimed.

```powershell
cmake --build --preset release
$env:M15_RUN_ID = "choose-a-fresh-id"
$env:M15_INIT_POLICY = "skip" # baseline, poison, gpu-zero are also supported
$env:M15_MATRIX_BACKEND = "int8" # fp16 for equivalent-workload comparison
build/bin/rdna3-int8.exe run fsr4-profile
Remove-Item Env:M15_RUN_ID,Env:M15_INIT_POLICY,Env:M15_MATRIX_BACKEND
ctest --preset release --exclude-regex 'experiment_fsr4_(chain|graph|cnb)$'
```

- [S] All raw evidence lives under `build/reports/m15/`: run `.log/.err`, exclusive
  run directories with `samples.csv`, `dispatches.csv`, `shader-stats.txt`,
  `release-identities.json`, and `ctest-release-targeted.log`. Golden tensors and
  weight extraction prerequisites are documented in EXP-016, not regenerated
  opportunistically. Build/test sequentially: Debug/Release share the executable.
- [U] Remaining opportunities: persistent resources/uploads, reduced packet
  expansion, LDS/tile reuse, fusion and layout changes. None is implemented or
  claimed to meet a practical hardware ceiling here. No cache-counter, occupancy,
  locked-clock, production-validation-off or whole-SR FP16 comparison is claimed.
