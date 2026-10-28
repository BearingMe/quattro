# EXP-017: Standalone GPU-resident temporal Super Resolution pipeline

Date: 2026-10-28 · Milestone: M13.

## Question

Can the exact M12 neural graph consume actual scene-derived temporal features
and reconstruct consecutive 1080p images, with controlled reset/invalidation
and observable stage timings?

## Method

- [S] Explicit permutation: FSR 4.0.2 v07 INT8 Quality source contract,
  1280x720 -> 1920x1080, linear color, manual unit exposure/preExposure,
  low-resolution unjittered motion, inverted depth, sharpening disabled.
  [Contract and limitations](../findings/2026-10-28-fsr4-temporal-contract.md).
- [S] Our buffer preprocessor performs Gaussian input reconstruction,
  motion dilation, Catmull-Rom history reprojection, recurrent reprojection,
  branch-specific invalidation/reset, feature conversion/normalization.
  The shared `AppendModelGraph` then appends all 14 unchanged neural passes.
  Postprocessing activates recurrence, evaluates the learned 3x3 filter,
  blends history, retains FP16 history/RFM and converts final color.
- [S] All three frames are recorded in one command list, **132 dispatches**:
  44 per frame (pre + 42 neural + post), with GPU resource transitions and
  bounded skip copies from M12. Earlier output resources become subsequent
  history SRVs; CPU snapshots are never supplied as later inputs.
- [S] Deterministic fixture: checker/gradient scene and HDR foreground rectangle,
  depth 0.8 foreground / 0.2 background. Frame0 resets with jitter (0.25,-0.25).
  Frame1 translates by one LR pixel, uses motion (-1,0) and opposite jitter;
  a central 32x32 LR region has NaN motion to exercise local invalidation.
  Frame2 is a camera cut to the exact frame0 input/settings, with explicit
  reset while the previous GPU history is dirty.
- [S] Assertions: exact frame0/frame2 packed output and pre-feature equality;
  reset Y-current/Y-history equality, normalized recurrent features -1,
  zero padding; actual nonzero recurrent persistence on frame1; invalid-motion
  center has zero-history/recurrence features; finite color, recurrence in
  [0,1], correct output size, positive finite GPU timestamps.
- [S] Host validation also rejects empty sequences, missing initial reset,
  unsupported dimensions/permutations and zero preExposure. Invalid motion is
  intentionally accepted by the host and handled by the GPU source contract.
- [S] Optional runner timestamp queries surround each dispatch's pre-copies,
  state transitions and dispatch. Aggregated pre/inference/post sums exclude
  initial upload and readback/snapshot copies. GBV remains enabled; these are
  diagnostics, not M15 baseline performance or a speedup claim.
- [S] Raw inputs and packed outputs plus display PPMs are written to exclusive
  `M13_RUN_ID` directories. The offline inspector checks all stored components,
  hashes inputs/outputs, and writes PNGs with an explicit display-only transform.

## Result

Hardware: RX 7700 XT/gfx1101, driver 32.0.32015.7029.

- [M] **PASS** in Debug and Release: three consecutive 1920x1080 frames;
  6,220,800 nonzero RGB components per frame, zero nonfinite color components,
  zero invalid recurrence values, zero reset-feature/padding errors.
- [M] Camera-cut reset is byte-identical to the clean initial frame, including
  packed history, activated recurrence and RGB output. NaN-motion local
  invalidation passes. Frame1 contains **8,276,544 persisted recurrent feature
  components**, proving history was not merely replaced with zeros.
- [M] Linear RGB ranges: reset frames approximately [0.0499995,1.99999];
  continued frame approximately [0.0473637,2.1143]. Inspectable image shows the
  checker/gradient scene and rectangle. This is not a quality-equivalence claim.
- [M] Complete suites, including all M12 operator/graph regressions:
  Debug **23/23 PASS (1880.33 s)**; Release **23/23 PASS (1701.30 s)**.
  One exact named root4 intrinsic message is filtered in each temporal test;
  all other validation errors remain fatal. No checks were disabled.
- [M] Release per-frame diagnostic GPU timestamp sums:

  | Frame | Pre (ms) | Neural inference (ms) | Post (ms) |
  |---|---:|---:|---:|
  | 0 reset | 0.493960 | 29.328440 | 0.473360 |
  | 1 continued | 0.678080 | 24.301960 | 0.546760 |
  | 2 camera cut | 0.188000 | 24.310160 | 0.538920 |

  Debug suite neural sums: 25.345960 / 24.323840 / 24.288600 ms.
  The initial standalone run measured 159.279120 / 161.753520 / 162.621080 ms.
  This variability is retained as negative performance evidence; its cause is
  not established. No comparison with the different-workload M03 official timing.
- [M] Independent offline inspection passes for both configurations: all packed
  half/float components finite, recurrence bounded, reset equality exact,
  continued output different. Raw numerical outputs match across Debug/Release
  (check per-frame hashes in the inspection reports).

## Evidence and reproduction

- `build/reports/m13/ctest-{debug,release}.log`: full verbose suites.
- `build/reports/m13/{debug,release}-final/`: three
  `input-frame*.bin` files (64-B constants + 1280x720x24-B scene records),
  `frame*.bin` (1920x1080x32-B results), and display `frame*.ppm` images.
- `build/reports/m13/{debug,release}-identities.json`: capture-time
  executable/CSO/weight/source hash ledgers recorded before switching builds.
- `build/reports/m13/{debug,release}-inspection/`: `inspection.json` and PNGs,
  generated by `scripts/m13_analyze_sequence.py`. Viewer transform is
  `v/(1+v)` then gamma 1/2.2, not raw HDR ground truth.
- Initial standalone run log `build/reports/m13/debug.{log,err}` retains
  the earlier timing observations; later exclusive captures do not overwrite it.

```powershell
cmake --build --preset debug
$env:M13_RUN_ID = "choose-a-fresh-id"
build/bin/rdna3-int8.exe run fsr4-temporal
Remove-Item Env:M13_RUN_ID
ctest --preset debug -j1 -V
# Record artifact hashes, THEN switch configurations (shared output executable).
cmake --build --preset release
ctest --preset release -j1 -V
python -B scripts/m13_analyze_sequence.py build/reports/m13/release-final `
  --output-dir build/reports/m13/choose-a-fresh-inspection-dir
```

[S] Omitted `M13_RUN_ID` generates a unique process/tick-based ID. Explicit
existing capture IDs fail rather than overwrite evidence. Sequence API is
currently bounded to 1–3 frames and requires an initial reset; it is not yet an
indefinitely streaming resource pool.

## Interpretation

- [M] **M13 execution gate PASS for the explicit experimental permutation**:
  complete scene input, preprocessing, inference, reconstruction, persistent
  intra-batch temporal history, camera cut/local invalidation, inspectable
  images and stage timestamps all execute on hardware.
- [S] No selected operation is silently bypassed. Auto-exposure, RCAS,
  alternate color/depth/MV permutations and other resolutions are not the
  chosen contract. **Official texture-sampler/numerical/visual equivalence is
  UNVERIFIED**, and software bilinear precision is an explicit M14 question.
- [S] General streaming beyond a three-frame batch, broad content coverage,
  long-sequence stability, quality claims and performance optimization remain
  outside this execution gate. M14/M15 require their own evidence.

## Next

M14 is authorized: feed these identical serialized inputs to an explicitly
versioned official reference path, compare raw outputs, and categorize static
and temporal differences. Do not substitute existing M03 Sponza screenshots
or compare tone-mapped visualizations as if they were identical inputs.
