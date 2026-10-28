# FSR 4 v07 integration: blockers, causes, and prevention

Scope: FSR 4.0.2 `fsr4_model_v07_i8_quality`, 1080p on RX 7700 XT/gfx1101.
Results: [EXP-016](../experiments/EXP-016-fsr4-resident-graph.md).
Earlier numerical contract: [epilogue finding](2026-10-28-fsr4-quantized-output-epilogue.md).

## Summary

- [M] All 14 passes now execute GPU-resident and exactly match the labelled
  pass-11 predication-control graph. The initial SRV/UAV obstacle was caused
  by a proposed single-resource layout, **not** by an inability to connect
  operators. Do not reintroduce that layout as a correctness prerequisite.
- [S] Keep this document as causes/prevention; measurements belong to EXP-010
  through EXP-016 and progress/gate status belongs to `docs/MILESTONES.md`.

## Integration traps and regression checks

| Trap | Verified cause / resolution | Prevention / evidence |
|---|---|---|
| One scratch allocation appears mandatory | [S/M] Existing shaders use SRV input and UAV output; the runner forbids the same resource occupying both roles in a dispatch. Separate live allocations compose correctly without shader changes. | Keep alias rejection. `fsr4-chain` proves 0->1; `fsr4-graph` proves all 14 (EXP-015/016). Never "fix" this by removing the guard. |
| Lost or wrong decoder skip | [S] Pass 9 needs **pass 5**'s output; pass 11 needs **pass 2**'s. Reusing their allocations too early loses live encoder data. | Explicit producer indices in `RunModelGraph`; preserve separate resources. Static lifetime metadata plus all-boundary exact comparison. |
| Fused kernels require concatenated source/skip | [S/M] Their verified source addresses expect `[input | skip]`. Four pre-dispatch GPU copies assemble two new buffers; no CPU round-trip or arithmetic rewrite. | Keep concatenation offsets in bytes. Overflow-safe source/destination bounds and distinct-resource checks; eight negative cases in `fsr4-graph`. |
| Wrong intermediate snapshot | [S] Own tensors have separate resources; the reference uses fixed offsets in a scratch resource, except pass 13 which writes `u0`. Micro-dispatches are not model boundaries. | Snapshot after the last dispatch of each model pass. Runner supports per-pass snapshot-resource override; compare logical byte counts and contents. Never compare entire differently laid-out scratch allocations. |
| Dispatch plan outlives input or PSOs | [S] `BufferInit` holds raw initial-data pointers; `BufferPass` holds raw PSO pointers. | Inputs/weights remain alive through submission/sync; `DispatchPlan.psos` owns all PSOs. Builders append only and may partially mutate on failure: discard failed plans. |

## Reference and arithmetic traps

| Trap | Verified cause / resolution | Prevention / evidence |
|---|---|---|
| Non-deterministic reference treated as our numerical error | [S/M] Shipped pass 11's 32/1 overload lacks a logical store guard. A 64-column overhang races next-row data. | Compare exactly to the **labelled** predication control, also measure shipped differences. Bands propagate to x<64 / x<66 / x<132 at passes 11/12/13. Zero differences outside these bands; never generalize a race count (EXP-010/016). |
| Physical OOB while executing reference | [S/M] Rounded dispatches can read beyond input rows and store past logical scratch/output. A race within valid allocation is a different issue from physical OOB. | Reference input uses 1088 padded rows, scratch is 20,736,000+1,024 B, final output reserves 1088 rows (+245,760 B). Compare logical extents; do not suppress tensor validation errors (EXP-010/014). |
| Dummy initializer binding hangs GPU | [M] Some passes read weights through `t1`; a dummy there caused `DXGI_ERROR_DEVICE_HUNG`. | Bind the checked payload's **89,216-byte initializer prefix** for every reference graph run. Do not infer an unused binding from one operator family (EXP-012 and isolated-reference source). |
| Algebraically equivalent epilogue differs by one INT8 unit | [R/M] Emitted constants/FMA contraction differ from literal source evaluation; reciprocal multiplication is not binary32 division. | Preserve ISA-derived `fl(fl(a*b)*fl(1/c))`, baked constants, contraction order and rounding. Existing full-range isolated fixtures plus all-boundary graph comparison are mandatory (EXP-012/013). |
| Transposed weights look like NHWC | [S/M] They are **HWCN**: kernel-x, kernel-y, output feature, input channel. | Use recorded strides, not guessed axis names; EXP-013/014 exact tests. |
| Final FP16 result off by a halfword | [S/M] Pass 13 narrows the scaled accumulator **before** adding bias, then adds in FP16. | Preserve that order, not float bias-add followed by one narrowing. Require exact halfwords, not a floating tolerance (EXP-014/016). |
| Driver intrinsic diagnostic mistaken for tensor error, or vice versa | [S/M] The known synthetic access is the named 16-byte dummy at root4 with an exact address/message pattern. Real weights are root2. | Keep the narrow existing `IsBufferPassWmmaDiagnostic` predicate. All other diagnostics fail. Never broaden it by root number alone (EXP-011). |
| Scalar weight loads trigger watchdog | [M] Early fused-kernel development used separate dword loads; loading `uint4` restored bounded execution. | Preserve verified vector loads when reorganizing fused kernels; record any new hang as negative evidence, not a reason to disable GPU validation (EXP-013). |

## Build, test, and evidence traps

- [S] Debug and Release share `build/bin/rdna3-int8.exe` and the shader output
  directory. **Build -> test -> record artifact hashes, then switch config**.
  Never build one while testing the other or run concurrent GPU experiments.
  Use `ctest --preset <config> -j1 -V` for retained evidence.
- [M] The earlier Debug regression attempt was cancelled by a server restart.
  A cancelled process/log is **not** a suite PASS. EXP-016 records complete
  replacement suites, with counts and durations. Do not infer success from
  partial output or a blank `Start-Process.ExitCode`.
- [S] CMake shader rules previously retained stale CSOs when only `-D` flags
  changed. When changing shader compile definitions, force regeneration of the
  **specific generated `build/bin/shaders/m12/*.cso` outputs**, verify the build
  command/ISA, and retain hashes. Do not clean broad directories or touch the
  read-only `.references/sources/` tree.
- [S] `M12_LAST_PASS` is an incremental debugging aid, not a gate shortcut.
  The full-graph CTest explicitly sets it to 13. Prefix passes must be reported
  as partial even when exact. `M12_DUMP` enables captures; use a fresh
  `M12_RUN_ID` so retained evidence cannot be silently truncated.
- [S] Own/reference-control equality is bytewise. Shipped final differences
  must distinguish **bytes** from **FP16 halfwords**; pixel bands use output
  pixel coordinates, not flattened byte offsets. `scripts/m12_report_graph.py`
  independently recomputes the counts, deltas, SHA-256 and bands from captures.
- [S] Artifact identity ledgers are collected before changing binaries.
  Re-comparing tensor files does not prove historical execution provenance;
  the report explicitly states that limitation. Do not claim equivalence of
  runs with different input, CSO or weight identities.

## Open questions / non-goals

- [U] Signed runtime behavior and other hardware/resolutions are not inferred
  from this generated-graph experiment. FSR 4.1 remains outside scope.
- [S] Allocator reuse, upload-volume reduction, caching and performance are
  not prerequisites for the M12 numerical gate. Introduce them only with an
  authorized objective and rerun exact intermediate comparisons afterwards.

## Sources

- `src/gpu/graph.{h,cpp}`: copies, snapshots, alias/range rejection.
- `src/inference/models/fsr4_graph.{h,cpp}`: model topology and lifetimes.
- `src/experiments/fsr4_reproduction.cpp`: negative cases, fixtures, comparisons.
- EXP-010–EXP-016 and the linked epilogue finding; no claim here extends beyond
  their pinned version, hardware and measured fixtures.
