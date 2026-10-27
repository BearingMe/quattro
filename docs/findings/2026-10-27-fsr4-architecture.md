# FSR 4 (model v07) INT8 architecture — investigation report

Date: 2026-10-27
Milestone: M04 (gate record)
Sources: FSR-4.0.2-reference @ `c5e34b4` (MIT mirror, source-level), lhl/fsr4-rdna3-optimization @ `49015b7` (independent audit, "lhl"), FidelityFX-SDK @ `60f4ea8` (closed runtime, black-box), EXP-002 runtime observations. Tags per [CONVENTIONS.md](../CONVENTIONS.md). Every claim cites its source; unresolved items are explicit.

---

## 1. Model identity and representations available (C02)

- [S] Model id **`fsr4_model_v07`**, generated **2025-02-26 12:40:27 by ML2Code** from ONNX `models\2024\October\Release7\Quality\fsr4_model_v07_i8.onnx` — header comment, `internal/shaders/fsr4_model_v07_i8_quality/passes_1080.hlsl` L2. Same v07 model family as shipped in 4.0.2/4.1 (mirror README).
- [S] Representations in the mirror: generated HLSL (per quality tier × resolution), `initializers.bin` weight blobs (87.1 KB INT8 / 127.0 KB FP8 per tier), ML2Code **runtime** (`ml2code_runtime/` operators + tensor system). **Not** available: the ONNX files themselves, the ML2Code compiler, training code (§8).
- [S] Two separate generated families: `fsr4_model_v07_i8_*` (INT8) and `fsr4_model_v07_fp8_no_scale_*` (FP8 e4m3). **Do not assume they are the same network quantized differently** — the operator sets differ (i8: `int8_NHWC/` ops; fp8: `float8_NHWC/` ops with WMMA + `groupshared` staging; both have FasterNet/ConvNext-style fused blocks, but the fp8 set is smaller, e.g. no `FusedConv2D_k3p1b_*` chains).
- [S] Six quality tiers (quality, balanced, performance, ultraperf, native, drs) — each an independently generated model with its own weights; tier **auto-selected per frame by upscale ratio** in the provider: ≥1.5 Quality, ≥1.69 Balanced, ≥1.99 Performance, ≥2.99 UltraPerformance, else NativeAA; DRS is sticky (`ffx_provider_fsr4_dx12.cpp` L1696–1705).
- [M] The 4.1 runtime (SDK 2.3.0 signed binaries) reports the same resource contract for 4.1.1/4.1.2 (`required_resources: 15, optional_resources: 0`) — see EXP-002. Internal 4.1.x HLSL/weights are closed ([U] whether graph changed vs v07).

## 2. Frame I/O contract (C01)

All from [S] `include/gpu/fsr4/ffx_fsr4_upscale_resources.h` L93–137:

**Inputs (per frame):**

| Resource | Type | Role |
|---|---|---|
| `r_input_color` | float3 (LR) | current low-resolution color |
| `r_velocity` | float2 | motion vectors (jitter-cancelled in shader, L34–41) |
| `r_depth` | float | depth (inverted-Depth supported, L148–152) |
| `r_history_color` | float3 | previous output (temporal) |
| `r_recurrent_0/1` | float16_t**4** | recurrent state channels (2 textures; repackaged to ease reprojection, L124–125) |
| `r_input_exposure` / `r_auto_exposure` | float / structured | exposure (game-provided or SPD-computed) |
| `r_previous_velocity` / `r_previous_depth` / `r_previous_auto_exposure` | — | temporal references |

**Outputs:** `rw_mlsr_output_color` (final, float3), `rw_history_color` (next frame's history), `rw_recurrent_0/1` (state), `rw_sr_mv_dilated` (dilated velocity), `rw_auto_exposure_texture`, optional `rw_output_color_for_rcas`/`rw_rcas_output`, `rw_debug_visualization` (R = predicted blend factor).

**Normalization:** [S] color and recurrent features are normalized to [−1, 1] (`MODEL_COLOR_NORMALIZE(x) = 2x−1`, denormalized `(x+1)/2`, L40–76); MuLaw applied to input samples in the optimized pre-upscale (L85–88).

**Temporal contract:** history + 2 recurrent textures + previous exposure/velocity/depth + `reset` cbuffer flag (L167). This is the minimum temporal state our reproduction must manage (M13).

## 3. Execution graph (C03)

### 3.1 Frame sequence (host side)

[S] `ffx_provider_fsr4_dx12.cpp` `Dispatch()` L1603–1990:

1. Align upscale size to 8 px (L1619–1620).
2. Auto-select quality tier by ratio (§1); recreate model if tier/colorspace changed ("This is slow!" — L1741).
3. Optional: SPD **auto-exposure** dispatch (L1899).
4. **Pre pass** — `model_pso[0]`, `[numthreads(8,8,1)]` (two variants: WMMA-table `RoundUpDiv(w,64),RoundUpDiv(h,2)` vs standard `preGroupSize` 8×8 tiling, L1907–1913).
5. **Body passes 1–12** — `model_pso[i]`, each followed by a **padding dispatch** (`padding_pso[i]`, `FFX_GPU_JOB_FLAGS_SKIP_BARRIERS`, L1921–1926) that fills tile-edge overhang in the byte-addressed scratch.
6. **Post pass** — `model_pso[13]`, 8×8 (L1936–1940).
7. Optional: **RCAS** sharpening (L1958), **debug view** (L1964), watermark.

Count: **14 model PSOs + 13 padding dispatches** per frame — cross-checked [S-lhl] ("14 model passes (`pass0..pass13`) plus 13 post passes (`pass0_post..pass12_post`)" — lhl README L24; the "post passes" there are the padding dispatches observed in the provider; 27 dispatches total at 1080p).

### 3.2 Network architecture (v07 INT8)

[S-lhl] L17–24, corroborated [S] by the mirror's operator library and generated code:

```
input (7 ch: color 3 + RFM 4, NHWC)
  → encoder1: strided Conv2×2 downscale      (weights EMBEDDED in pre.hlsl as packed fp16 constants)
  → encoder2: 2× ConvNext-style blocks        (conv_dw + pw_expand + pw_contract)
  → encoder3: 2× FasterNet-style blocks       (spatial-mixing partial convolutions)
  → bottleneck: 2× residual blocks + ConvTranspose2×2 upsample
  → decoder3/2/1: mirror stages w/ skip connections + upscale convs
output → post (KPN): kernel-predicted upsample of upsampled_color + reprojected history blend
       → RCAS (optional) → output
```

- [S] First layer weights embedded in `pre.hlsl`: `embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight_dwords` (packed fp16 pairs, channel-padded to 4, float comments decode e.g. `0x294ba1cb` = fp16 `0x294b`=0.04135, `0xa1cb`=−0.011314) + `downscale_quantizationScale = 0.014884727…` (L1–71). Later layers' weights live in `initializers.bin` at hardcoded offsets.
- [S] Three encoder resolutions in the provider (`upscaleW/2, /4, /8` — L1632–1633) = the three U-Net spatial scales; dispatch tables per scale (body passes 1–2 ↔ enc0, 3–5 ↔ enc1, 6–9 ↔ enc2, 10–12 ↔ enc1/enc0 decode).
- [S] The post pass is a **kernel-prediction network (KPN)**: `rw_upsampled_color`/`rw_reprojected_color` "used in post shader KPN" (resources header L118, L121) — the network predicts per-pixel kernel weights/blend rather than emitting final color directly; debug view exposes the predicted blend factor.
- [U] Exact per-pass channel counts and kernel shapes: present in the generated code (deducible), not yet tabulated — see §8.

### 3.3 Instruction usage

- [S] Pass 0: FP16 `dot2add` path (lhl L70); **passes 1–13 dominated by `dot4add_i8packed`** (SM 6.4 packed INT8→INT32 dot) via `DOT4_ENABLED 1` fused operators (`post_common.hlsli` L76–218).
- [S] FP8 variant: **WMMA-only** via `AmdWaveMatrixMultiply` 16×16 tiles + `groupshared` LDS staging (`pre_wmma.hlsl`, `post_wmma.hlsl`, `float8_NHWC/Conv2D_k2s2b.hlsli`).
- [M] On gfx1101 (RDNA 3), the INT8 family is the relevant path: no WMMA required for reproduction — DP4A is AMD's own workhorse (F1).

## 4. Tensor formats (C06)

- [S] Layout: **NHWC**, 3D/4D tensors, spatial tiling with per-pass `groupStart/groupSize`, tile-edge `paddingBegin/paddingEnd`.
- [S] Generated tensor records, e.g. `passes_1080.hlsl` (i8_quality):
  `QuantizedTensor3i8_NHWC { logicalSize, groupStart, groupSize, storageSize, tensorByteStrides=(16,15360,1), paddingBegin/End, threadGroupByteOffset + 0 | 8294400 | 12441600, quantizationScale, storage }` (L143–149, L519–525, L1131–1136).
  - Scratch is one 19.78 MB `RWByteAddressBuffer` ("Scratch memory size needed: 20736000", L5); passes ping-pong between offset regions **0 / 8 294 400 / 12 441 600** B (double-buffered encoder halves + extra band) — "spilling here because of pass fusion" (resources header L133).
  - `QuantizedTensor` carries a per-tensor `quantizationScale`; plain `Tensor3i8_NHWC` slices (no scale field) appear where no requantization happens between fused stages.
- [S] Byte strides at 1080p: row-block 16 B × 15360 B per plane-slice — consistent with 960×(16 ch /… ) tiling at enc0 resolution (`logicalSize` per slice to be tabulated in M09/M12 when implementing specific layers).
- [S] Network outputs go to `StructuredBuffer` (`r_network_output_rfm`/`other`) consumed by the KPN post pass (resources header L105–106).

## 5. Quantization and scaling (C04)

- [S] **Per-tensor (per-layer) symmetric scales** in the generated code: `downscale_quantizationScale = 0.0148847…` (pre.hlsl L71); per-slice `quantizationScale` in every `QuantizedTensor` (§4). FP8 variant is explicitly "**no_scale**" (no per-tensor scale in filenames) — contrast confirms INT8 uses scales, FP8-v07 ships without them.
- [S-lhl] **Accumulate INT32, quantize once at store**: `round(value * rcpScale)` at the end of each fused convolution (lhl L85); per-iteration requantization is an anti-pattern (their harness: INT8 +194%, FP8 +476% — F6).
- [S] `QuantizeLinear`/`DequantizeLinear` operators exist for int8/uint8/fp16 NHWC (`operators/*_NHWC/`) — used at precision-boundary ops.
- [U] Whether 4.1.x uses identical scale placement (closed) — assume v07 semantics until falsified; verify numerically in M12 against the oracle.

## 6. Version-specific differences (C07)

- [S] 4.0.2 = model v07, generated 2025-02-26, "generated for **navi48** SKU" header note (`passes_1080.hlsl` L7 — internal generator SKU tag; [U] why "navi48"; generated code is resolution- not SKU-specialized beyond that comment).
- [M] 4.1.1/4.1.2 (SDK 2.3.0 signed runtime): same `ffx` API, `required_resources: 15, optional_resources: 0` (vs FSR 3.1.5/2.3.4: 15 + 48 optional) — EXP-002; internals closed.
- [S] INT8 vs FP8 are separately generated networks (§1); RDNA 3 ships INT8, RDNA 4 FP8 [D-press TechSpot].
- [U] 4.1.x graph deltas vs v07: not publicly documented; the companion RE project (deferred, `.references/README.md`) is the sanctioned route if M12 needs them.

## 7. Feasibility assessment (gate)

- [S] **Complete INT8 v07 graph is reproducible from public artifacts**: every operator used by the i8 tier exists as HLSL source in the mirror (`int8_NHWC/` + `Fused/`), the pre/post semantics are readable in full, weight payloads are present (embedded first layer + `initializers.bin`), and the dispatch structure is fully visible in the provider.
- [S] Runtime support needed: byte-addressed scratch ping-pong, `StructuredBuffer` network I/O, per-tensor scales, temporal state (history + 2 recurrent fp16×4 textures) — all plain D3D12 compute (SM 6.4 `dot4add_i8packed`), no vendor extensions.
- [H] A faithful re-execution of v07-i8 should land within tens of percent of the 1.36 ms oracle (M03); deviation sources: pass-fusion quality, tile shapes, dispatch count.
- [U] FP8/WMMA path reproduction requires AMD intrinsics extension or SM 6.9 CoopVec mapping — separate decision (M06), not blocking INT8 reproduction.

## 8. Unavailable / proprietary components (C08)

| Component | Status | Impact |
|---|---|---|
| ONNX files (`fsr4_model_v07_i8.onnx`) | **Not public** (path referenced in generated header only) | Graph must be read from generated HLSL, not ONNX |
| ML2Code compiler | **Closed** | We must hand-replicate the graph, not regenerate it |
| 4.1.x HLSL/weights | **Closed** (signed DLLs only) | v07 is the source-visible proxy; 4.1.x oracle is black-box (EXP-002) |
| Training data / quantization calibration sets | **Not public** | Quantization behavior inferred from generated code + measurements |
| Per-pass channel/kernel-shape table for all 12 body passes | Present in source, **not yet tabulated** ([U] open work) | Needed for M09/M12 implementation; extractable mechanically |
| `amd_fidelityfx_dx12.dll` (driver store) role | [U] | Sample runs without it being loaded; presumed driver-side integration path for games |

---

**Gate summary:** I/O contract documented (§2), representations identified (§1), execution graph reconstructed with provider-level dispatch evidence (§3), quantization classified with explicit [U]s (§5), tensor formats documented (§4), version differences recorded (§6), feasibility assessed (§7), missing components listed (§8). No component was invented; all claims carry citations.
