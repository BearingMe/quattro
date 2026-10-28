# EXP-018: Same-input official/experimental image and temporal comparison

Date: 2026-10-28 · Milestone: M14.

## Question and scope

How do the experimental 4.0.2 v07 pipeline and the available official provider
differ on identical static and moving/reset sequences? This is an explicitly
**cross-version image analysis**, not a same-model INT8 error measurement.

## Method

- [M] Public-API preflight on RX 7700 XT/gfx1101 enumerates 4.1.2/4.1.1/3.1.5/
  2.3.4 from the SDK DLL and 4.1.2/3.1.5/2.3.4 from the DLL in the
  `FSR-4.0.2-reference` tree. Both default contexts report **4.1.2**. A directory
  name is not a runtime version. No 4.0.2 provider was enumerated on this setup.
- [S] `fsr4-reference-capture` selects the enumerated **4.1.2** ID
  `0xf5a5ca1e81001002`, then checks the actual created context's provider ID.
  No silent fallback. It uses the SDK public DX12 API, not a patched binary.
- [S] The exact M13 serialized color/depth/MV/constants are uploaded to typed
  textures: RGBA32F color, R32F depth, RG32F motion, R32F unit exposure. Output
  is 1920x1080 RGBA32F, sharpening off, HDR/linear, inverted depth, LR motion.
  Jitter, motion scales, preExposure and reset are copied from the serialized
  constants. API camera parameters: near 0.1, far 1000, vertical FOV pi/3,
  view-space-to-meters 1, frame time 1000/60 ms. These do not turn the synthetic
  scene into a calibrated real-camera benchmark.
- [S] One official context retains its internal history across three dispatches.
  Per-frame uploads/readbacks are synchronized; no CPU history feedback. The
  collector checks finite RGB and a clean D3D12 info queue without new filters.
- [S] Two three-frame sequences: (1) M13 moving/reset/NaN-motion fixture;
  (2) stationary scene with alternating jitter and continued history after
  frame0. `M13_SEQUENCE=static` enables the latter; the M13 CTest explicitly
  forces `moving-reset`, so that environment cannot bypass reset/invalidation.
- [S] `m14_compare_images.py` **refuses unequal input bytes**. Metrics use raw,
  unclipped linear RGB, ignoring output alpha. No exposure fitting or image
  alignment. MAE/RMSE/max difference and fixed-peak-2 PSNR are reported, with
  full-frame metrics retained alongside region metrics. Display images use
  Reinhard+gamma only for inspection; difference images use absolute RGB x16.
- [S] A fixed, scene-derived edge mask covers checker/rectangle boundaries
  within +/-2 LR pixels (=3 HR pixels). The invalid-motion patch is separately
  reported, not excluded. Static temporal metrics compare successive outputs;
  the moving continuation uses the known 1.5-HR-pixel translation with bilinear
  sampling of the previous image over valid overlap. Reset compares to frame0.
  These are short-window residuals, **not isolated flicker or ghosting scores**.
- [S] Five independent small metric tests check exact equality, known MAE/RMSE/
  PSNR, no HDR clipping, nonfinite rejection, and empty-region handling.

## Results

Hardware: RX 7700 XT/gfx1101, driver 32.0.32015.7029.

- [M] All official frames finite, both sequences. Official RGB ranges: reset
  approximately [0.0499993,1.99999], moving continuation [0.0459281,2.30123],
  static continuations [0.0498957,1.99995] and [0.0498717,1.99975].
- [M] Repeating the official moving sequence produces identical RGB arrays on
  every frame. Both official and experimental camera-cut outputs equal their
  respective initial frame exactly.
- [M] Additional current-Debug collector run `official-exit-verified` completed
  all three official frames with **process exit 0** and no validation errors.
  This is a targeted collector check, not a replacement full Debug suite.
- [M] Experimental-vs-official **cross-version** differences:

  | Sequence/frame | MAE | RMSE | Maximum absolute RGB difference | PSNR, peak=2 |
  |---|---:|---:|---:|---:|
  | reset (both sequences), moving frame2 | 0.000441795 | 0.004045693 | 0.622665405 | 53.8807 dB |
  | moving frame1 | 0.000993021 | 0.007530628 | 1.15269172 | 48.4840 dB |
  | static frame1 | 0.000742372 | 0.006023023 | 0.891847551 | 50.4243 dB |
  | static frame2 | 0.000649993 | 0.004951316 | 0.808505446 | 52.1262 dB |

- [M] More than **99.9% of squared RGB difference** falls inside the predefined
  edge mask on every frame. Reset interior RMSE is 0.0000132529; moving interior
  RMSE is 0.000184507. The moving invalid-motion patch RMSE is 0.000498058 and
  maximum difference 0.00705421: that patch is not the dominant global error.
  Large edge maxima must not be hidden behind high global PSNR.
- [M] Static successive-frame RMSE (experimental / official):
  frame0->1 **0.01659671 / 0.01475009**;
  frame1->2 **0.00385766 / 0.00339735**.
  Moving compensated residual RMSE: **0.01927788 / 0.01453174**.
  Reset residual is exactly zero in both implementations. These metrics include
  reconstruction/jitter-phase response, not just temporal instability.
- [M] Moving-frame components above the known input maximum 2: **866 / 608**
  (experimental / official). Peak output is higher in the official capture
  despite fewer such components. Neither statistic alone establishes better
  perceptual quality or closeness to ground truth.
- [M] Release full CTest **24/24 PASS (2035.72 s)**, including all M12/M13
  regressions and the new metric tests. Official capture/probe and static
  collection additionally pass as explicit CLI runs. The subsequent Debug
  suite was **interrupted during `experiment_fsr4_graph`**, not hung; its log
  reaches exact fixture1 results. No process remained after interruption.
  **There is no complete Debug-24 verdict for this revision.** The earlier
  M13 revision's Debug 23/23 result is not relabelled as current coverage.

## Interpretation and limitations

- [M] **M14 analysis gate PASS for these bounded synthetic, cross-version
  comparisons**: identical input sequences verified, official/experimental
  images captured, raw methodology documented, static/moving/reset numerical
  differences and temporal residuals measured, dominant visible differences
  categorized as edge/reconstruction/history response.
- [S/M] M12 establishes zero neural-tensor arithmetic difference against the
  pinned 4.0.2 contract. M14's combined remaining comparison sources are the
  **different closed 4.1.2 model/temporal implementation and the experimental
  pre/post adapter**, including software-vs-fixed-function sampling precision.
  Edge localization identifies the dominant observed error category; it does
  **not** isolate which weights, operator or sampler rounding caused it.
- [U] Same-version signed-4.0.2 images, the individual contribution of INT8
  quantization versus model version/sampler differences, native-resolution
  ground truth, long-sequence flicker, general game-scene ghosting, and alternate
  exposure/permutation quality remain unverified. No numerical equivalence,
  perceptual superiority, or real-game quality claim is made.
- [S] No kernels were optimized in this analysis milestone. M15 gets a separate
  profiling/optimization experiment with exact correctness regression.

## Evidence and reproduction

- `build/reports/m14/reference-probe-release.{log,err}`: actual version/context
  enumeration from both DLL bundles.
- `official-release-1/`, `official-release-repeat/`, `official-static-release-1/`:
  official raw RGBA32F images and exact input copies. Corresponding `.log/.err`
  files identify the selected provider and validation result.
- Experimental moving captures: `build/reports/m13/release-final/`; static:
  `build/reports/m13/m14-static-release-1/`.
- `build/reports/m14/{moving,static}-comparison-1/`: full `comparison.json`,
  experimental/reference PNGs and difference images. `release-identities.json`
  is the capture-time executable/DLL/source SHA-256 ledger, not an attestation
  that the folder-named DLL contains only that model or bypasses driver routing.
- `build/reports/m14/ctest-release.log` is complete;
  `ctest-debug.log` is the retained **incomplete** attempt.

```powershell
cmake --build --preset release
build/bin/rdna3-int8.exe run fsr4-reference-probe
$env:M14_INPUT_DIR = "build/reports/m13/release-final"
$env:M14_RUN_ID = "choose-a-fresh-official-id"
build/bin/rdna3-int8.exe run fsr4-reference-capture
Remove-Item Env:M14_INPUT_DIR
Remove-Item Env:M14_RUN_ID

$env:M13_SEQUENCE = "static"
$env:M13_RUN_ID = "choose-a-fresh-static-id"
build/bin/rdna3-int8.exe run fsr4-temporal
Remove-Item Env:M13_SEQUENCE
Remove-Item Env:M13_RUN_ID
# Capture the identical new static input directory through the official command.
python -B scripts/m14_compare_images.py build/reports/m13/release-final `
  build/reports/m14/official-release-1 --sequence moving-reset `
  --output-dir build/reports/m14/choose-a-new-comparison-dir
python -B scripts/m14_test_metrics.py
```

Capture/analysis directories are exclusive. Do not silently overwrite a run or
restart complete 30-minute suites merely because a progress report was missing;
use targeted checks during analysis, and retain interrupted evidence as such.
