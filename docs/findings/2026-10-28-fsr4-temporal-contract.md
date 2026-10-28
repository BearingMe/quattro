# FSR 4.0.2 v07 temporal input and output contract

Scope: pinned `FSR-4.0.2-reference` @ `c5e34b4`,
`fsr4_model_v07_i8_quality`, 1280x720 render -> 1920x1080 output.
Execution evidence: [EXP-017](../experiments/EXP-017-fsr4-temporal-pipeline.md).

## Source contract

- [S] `include/gpu/fsr4/pre_common.hlsli:141–213` builds **current Y, history Y,
  chroma distance, four recurrent features**, then a padding channel. Earlier
  shorthand "RGB + RFM" in the M12 fixture commentary was inaccurate. M12
  remains valid as a raw-tensor numerical test; its arbitrary inputs were not
  claimed to be real reconstructed scene features.
- [S] The color features are YCbCr luminance (0.2126 R + 0.7152 G + 0.0722 B),
  and chroma distance is `sqrt(deltaCb^2 + deltaCr^2 + 1e-6)`. Luminances and
  recurrence are cast to FP16 and normalized by `2*x-1`; distance is cast to
  FP16 without normalization. The eighth storage channel is zero.
- [S] Current color is reconstructed from four LR neighbors with separable
  Gaussian weights (`sigma=0.47`) at
  `(-0.5 + invScale/2 + outputPixel*invScale) + jitter`. Color loads clamp.
  Linear color uses `mu(c)=max(0.1174*log(1+150*c),0)` after exposure scaling.
- [S] Manual effective exposure is `(inputExposure==0 ? 1 : inputExposure) /
  preExposure`. Auto-exposure and alternate color spaces are separate source
  permutations, not implicit requirements of the selected one.
- [S] `mlsr_optimized_includes.hlsli:115–180`: low-resolution depth selects the
  nearest of a center and eight neighbors in a fixed order; inverted depth
  chooses the greatest depth. Raw LR-pixel motion is scaled to normalized UV
  offsets; optional jitter cancellation is subtracted. High-resolution MV is a
  different permutation. Depth/velocity out-of-range loads return zero rather
  than using the color-load clamp.
- [S] Local invalidation is **not** a full depth-disocclusion test: the source
  checks NaN/Inf velocity or a component **greater than** 1,000,000. It does not
  check absolute magnitude or compare previous/current depth. Do not silently
  add a different rejection algorithm and call it the same contract.
- [S] Reprojection uses `outputPixel + velocity*outputSize`. Onscreen history
  color uses nine-fetch Catmull-Rom and recurrence uses bilinear filtering, both
  with a clamp sampler (provider at line 969). History is clamped nonnegative
  and blended with current color by `0.1 * largestCurrentGaussianWeight`.
  Offscreen reprojection starts with black history/zero recurrence and still
  applies the current-color blend. Invalid motion skips that blend.
- [S] Reset uses **current reconstructed color as history** and **zero
  recurrence**; zeroing both is wrong. It follows that normalized recurrent
  features are **-1**, not zero. Current/history luminance feature bits should
  be identical on a reset frame.
- [S] Network outputs 0–3 are reconstruction-filter parameters, **not RGB**.
  Outputs 4–7 become sigmoid-activated recurrence. `post_common.hlsli:7–53,
  279–316` uses a learned anisotropic correlated 3x3 Gaussian, sigmoid history
  blend, retains model-space history, then inverse mu-law/exposure conversion
  and clamp to [0,64000]. RCAS-disabled is an explicit supported permutation.
- [S] Provider allocations use FP16 history/reprojected color and recurrence.
  Preserve storage conversion at those boundaries, not FP32 history forever.

## Experimental adapter and limits

- [S] `fsr4_temporal.{h,cpp,hlsl}` implements these operations with byte-addressed
  buffers so the M12 neural graph remains unchanged. Scene records are 24 B
  (`float3 color`, `float depth`, `float2 motion`) after a 64-B constant header.
  History/result records are 32 B (`half4 history`, `half4 recurrence`,
  `float4 linear RGB output`). Reprojected color uses a separate 8-B half4.
- [S] Selected gate fixture: linear color, manual unit exposure/preExposure,
  low-resolution **unjittered** motion, inverted depth, no RCAS. Other
  resolutions, non-inverted/high-resolution motion and implicit first-frame
  history fail validation instead of being bypassed. The optional jittered-MV
  branch exists but was not the gate fixture; non-unit exposure is implemented
  but has not been numerically compared against the official runtime.
- [S] Current API is a bounded batch of **1–3 frames**, initial reset required.
  Color history and recurrence persist on GPU within the batch; they are not
  fed back through CPU readback. This is not yet a persistent streaming API.
  Separate per-frame workspaces trade memory for simple lifetime correctness.
- [S] Software bilinear filtering has no claim of exact fixed-function texture
  sampler precision. Pre/post equivalence must be measured separately in M14;
  M12's exact neural-tensor contract must not be weakened to hide that issue.
- [M] Reset, actual nonzero recurrent persistence, invalid-motion local reset,
  finite outputs, correct dimensions and per-stage GPU timestamps pass in Debug
  and Release (EXP-017). Visual quality equivalence is **UNVERIFIED**.

## Prevention before changing this code

1. [S] Never interpret the eight final neural channels as RGB+padding. Preserve
   learned filtering, history blending and recurrence activation.
2. [S] Keep the different reset/offscreen/invalid-motion branches explicit.
   Re-test reset equality against a clean frame with identical jitter/input.
3. [S] Keep exact M12 graph comparisons as a regression when extracting dispatch
   assembly; use optional snapshots only for diagnostics, never frame feedback.
4. [M] Initial isolated M13 timestamps differed greatly from warmed suite runs
   (about 159–163 ms inference versus 24–29 ms). The cause is unverified. Do not
   call cache/clock/instrumentation variance an optimization result. M15 needs
   a new controlled, repeated baseline; these are validation-enabled diagnostics.
5. [S] PPM/PNG inspection uses Reinhard+gamma only for viewing. Numerical
   comparisons must use raw linear outputs and identical exposure/jitter/motion,
   not tone-mapped screenshots or the M03 Sponza screenshot as a substitute.

## Sources

- Pinned `pre_common.hlsli`, `post_common.hlsli`,
  `mlsr_optimized_includes.hlsli`, `ffx_fsr4_upscale_resources.h`,
  `ffx_fsr4_gpu_colorspace_common.h` and `dx12/ffx_provider_fsr4_dx12.cpp`.
- `src/inference/models/fsr4_temporal.{h,cpp}` and
  `src/inference/models/shaders/fsr4_temporal.hlsl`.
- EXP-017; [integration blocker prevention](2026-10-28-fsr4-integration-blockers.md).
