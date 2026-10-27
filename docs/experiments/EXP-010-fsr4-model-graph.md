# EXP-010: FSR 4 v07 INT8 model graph extraction and reference execution

Date: 2026-10-27 · Milestone: M12 (in progress)

## Question

Which exact execution graph, tensor geometry, weight payloads, scales and
dispatch structure does the selected FSR 4 INT8 model use — and can a same-version
numeric reference be executed locally so that a reproduction can be compared
against it?

## Method

1. [S] **Pin the reference.** `FSR-4.0.2-reference` @ `c5e34b4`, model
   `fsr4_model_v07_i8_quality`, `passes_1080.hlsl` +
   `initializers.bin`; hashes recorded in
   [the findings document](../findings/2026-10-27-fsr4-v07-graph.md) §1.
   Read-only tree; nothing under `.references/sources/` is modified.
2. [S] **Extract the graph mechanically** with `scripts/m12_extract_graph.py`.
   The script parses every pass block: entry point, `numthreads`, operator
   include and call, all tensor descriptors (logical/storage sizes, byte
   strides, storage byte offset, quantization scale, backing buffer) and all
   embedded constant array inventory (not payloads). Output: `build/reports/m12/graph.json` +
   `graph.md`. Dispatch grids are taken from the provider's non-WMMA table
   (`ffx_provider_fsr4_dx12.cpp`), whose selection is env-gated by `MLSR-WMMA`.
3. [S] **Execute the shipped passes as the reference.** CMake compiles the 14
   passes with the same flags the model ships with (`-T cs_6_6 -O3
   -enable-16bit-types -HV 2021`, `MLSR_PASS_<n>`), from the pinned tree into
   `build/bin/shaders/m12ref/`. `run fsr4-oracle` uploads deterministic FP16
   input features (7 valid channels, exactly-representable values) and
   `initializers.bin`, runs pass0…pass13 in one command list through
   `gpu::RunBufferPasses` (t0/t1 SRV, u0/u1 UAV, per-pass UAV barrier,
   COMMON↔SRV/UAV transitions, device-removal check), snapshots the scratch
   buffer after each pass (`M12_SNAPSHOTS=1`) and reads back the FP16 output.
4. [S] **Smoke validation only**: the final output is 1920×1080×8 FP16 with
   no NaN/Inf and nonzero values. This does not validate intermediate INT8
   semantics or numerical correctness. Group-scale chain validation remains C06.

## Result

[M] **Reference executes on gfx1101.** 16,588,800 output halfwords, 0 NaN,
0 Inf, 16,588,400 non-zero in the first retained run, min −25.65625,
max 39.46875 (`build/reports/m12/ref/run1/output.bin`, 33,177,600 B;
log `ref/run1/run.log`). The nonzero count is not stable across executions.

[M] **Determinism is partial and the cause is in the reference.**
Passes 0–10 are bit-identical across the first retained three-run collection.
Pass 11 varies in 58,119 bytes, x∈[0,63] of 64 rows; final output varies in
347,317 of 16,588,800 halfwords, x∈[0,131], |Δ| up to 18.8828125.
These are union-of-differences counts for this run set, not a stable race rate.
Cause [S/M]: the **32/1 DP4A overload** of `FNB_CT2D_ADD` stores with
`output.storage.Store4(...)` and **no logical-size predicate**, while the 128×2
group tile with the provider's 8-group grid addresses 1024 columns for a
960-column tensor; the 64-column overhang aliases into the next row's first 64
columns and races the groups that own them. The **64/2 DP4A overload** used by
pass 9 carries the explicit predicate at L263; the earlier attribution of this
line to WMMA was wrong. A generated, labelled predication control changes only
the 32/1 store guard and never modifies the pinned source tree.

[M] All three predicated controls are bit-identical, including final output
SHA-256 `a57bcd063967affddf29cc24709b9d8a7fd9f8e7244e7010521d7b27f605b6de`.
Unmodified run1/control1 differ in 1,131,402 halfwords across 1,076 rows but
only x<132: stable corruption exceeds run-to-run variation. Original pass11
writes 1,021 nonzero bytes into the 1,024-byte scratch tail; pass9/pass10 and
all predicated controls leave it zero. Recompare retained data with
`scripts/m12_diff_snapshots.py --legacy`; report `build/reports/m12/determinism.json`.

[M] Provenance-complete repetition: `verified-determinism.json` verifies the
same executable/CSO/control identities and input hash across all six
`ref/verified-*` runs. Original final output varies in 138,585 halfwords; all
three controls are again bit-identical with the same output hash. Recompare
with `python -B scripts/m12_diff_snapshots.py --prefix verified-`. Five mock-only
capture-integrity regression tests reject missing/mismatched provenance and
verify that a losing concurrent collector cannot overwrite the winner's log.

[S] The scratch overhang footprint is **1,024 B**, not the historical “≤995 B”
inferred from one validation diagnostic. Pass 13's unpredicated FP16 store
overhangs 1080 rows by 8 (245,760 bytes). The harness allocates these extents and
retains the scratch tail for inspection, with zero-initialized scratch on each
invocation. No persistent-frame/runtime-equivalence claim is made.

## Interpretation

- [S] M12 C01–C03 and the metadata part of C06 have a mechanically extracted
  contract: 14 fixed passes, fixed grids, fixed scales,
  weights split between 67 embedded arrays and 11 initializer tensors
  (79,360 B of an 89,216 B file).
- [S/H] The selected generated model is a numeric reference for C08, with a
  predicted corruption footprint x<132 for the final output. Quantify this band
  separately; do not equate stability outside it with numerical correctness.
  Runtime shader permutations/preprocessing are not being reproduced in M12;
  no signed-runtime equivalence or permutation invariance is assumed.
- [U] Whether the signed FSR 4.0.2 runtime shows the same band on RDNA 3, and
  whether the RDNA 4 WMMA path avoids it, are open; nothing here is claimed
  about FSR 4.1.
- [H] The reproduction should predicate stores to `logicalSize`. Exactness
  outside the corruption footprint remains an experimental target, not a
  guarantee; differences still require per-operator investigation.

## Not yet done (M12 gate not met)

- C04: remaining FasterNetBlock, FNB_CT2D_ADD, CNB_CT2D and quantized downscale
  convolutions. Independent pass 0 and ConvNextBlocks are covered by EXP-011.
- C05 is now covered by [EXP-011](EXP-011-fsr4-reproduction-operators.md):
  checked full-payload loading and 78 weight descriptors. The original graph
  extractor is inventory-only; the separate packer extracts the actual data.
- C07: our own executor running the graph GPU-resident.
- C08/C09: per-stage numeric comparison against the reference and the recorded
  differences.

The reference executing AMD's own shaders is **not** a reproduction and does not
satisfy the gate.

## Reproduction

```powershell
python scripts/m12_extract_graph.py
cmake --build --preset debug
build/bin/rdna3-int8.exe run fsr4-oracle
python -B scripts/m12_diff_snapshots.py --collect --prefix new-  # choose a fresh prefix
python -B scripts/m12_diff_snapshots.py --prefix new-            # recompare captures
```

The normal oracle smoke test writes no captures. Explicit dumps require
`M12_DUMP=1`; snapshots additionally require `M12_SNAPSHOTS=1`. `M12_RUN_ID`
selects a capture subdirectory under `build/reports/m12/ref/`. The collection
script and the oracle writer refuse existing run directories. New captures
record executable/CSO/control SHA-256 identities and the input recipe/hash in
per-run manifests. Controls additionally
set `M12_PREDICATED=1` and are never labelled as the shipped reference.
Comparison requires identical implementation/input identities across all six
new captures. `--legacy` explicitly allows older manifest-free captures and
marks their report provenance as unverified; it cannot be used for collection.

## Files

- `scripts/m12_extract_graph.py`
- `src/gpu/graph.{h,cpp}` (generic multi-pass runner), `src/gpu/d3d12util.h`
- `src/experiments/fsr4_oracle.cpp`
- `CMakeLists.txt` (reference CSO rules, `experiment_fsr4_oracle` test)
- `docs/findings/2026-10-27-fsr4-v07-graph.md`
