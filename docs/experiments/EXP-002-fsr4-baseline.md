# EXP-002: Does the official FSR 4 implementation execute its ML upscaling path on the RX 7700 XT, and what does it cost?

Date: 2026-10-27
Milestone: M03

## Question

Can the official (closed-source) FSR 4 "FSR Upscaling" implementation run unmodified on the gfx1101 target, confirmed on its ML path, with captured frames and reproducible GPU timings — establishing the behavioral oracle for all later milestones?

## Hypothesis

FSR SDK 2.3.0 distributes FSR 4.1 for RDNA 3 as closed signed binaries; the official `FidelityFX_FSR` DX12 sample drives them through the `ffx` API. If the binaries enumerate an FSR 4.1.x upscaler and the ML upscaler context is created by default, the official baseline runs on this machine without any 4.0.x substitute.

## Reference sources

- [S] FidelityFX-SDK @ `60f4ea8` (tag v2.3.0): `Kits/FidelityFX/upscalers/include/ffx_upscale.h` (`FFX_UPSCALER_VERSION` 4.1.1), `Kits/FidelityFX/signedbin/`, `Samples/Upscalers/FidelityFX_FSR/`, `readme.md` (ML-Upscaler 4.1.1 row), `UpdateMedia.bat` (media bundle SHA256).
- [D] GPUOpen FSR SDK 2.3 announcement (FSR 4.1 RDNA 3 support), see [`.references/topics/fsr4.md`](../../.references/topics/fsr4.md).
- Official prebuilt samples: `FidelityFX-Samples-v2.3.0-prebuilt.zip`, release asset of `GPUOpen-LibrariesAndSDKs/FidelityFX-SDK` v2.3.0.

## Configuration

- Binary: official prebuilt `FidelityFX_FSR.exe` (+ `amd_fidelityfx_loader_dx12.dll` 2.3.0.2740, `amd_fidelityfx_upscaler_dx12.dll` 4.1.1.2740 "AMD FidelityFX (DirectX) Upscaler Library"), executed unmodified from the extracted release zip.
- Media: official bundle SHA256 `48cfdbd0b65a2429b1078d820e4ad9bc115c356afda705325a94494cf6cf8e01` via `Tools/media_delivery/MediaDelivery.exe` (Sponza scene + IBL, 1.4 GB).
- Command line (identical for both runs, recorded by the sample itself in the JSON `CmdLine` field):
  `FidelityFX_FSR.exe -resolution 1920 1080 -benchmark json duration=600 path=<build>/temp/m03-bench -screenshot`
- Benchmark mode forces the 60 FPS GPU limiter; 600 frames; JSON + screenshot outputs.
- Working copies of all outputs: `build/captures/m03/`.

## Procedure

1. Download official release asset, extract under `build/temp/` (never inside the pinned submodule).
2. Fetch official media bundle with the pinned SHA256.
3. Run the sample twice with the command line above; collect `Cauldron.log`, benchmark JSONs, screenshots.
4. Cross-check GPU identity, upscaler version selection, resolutions, and per-pass GPU timings.

## Expected result

- Runtime enumerates FSR 4.1.x upscaler versions and creates an ML upscaler context by default.
- Frame captures show a valid upscaled scene; GPU timings measurable and repeatable.

## Actual result

[M] All checkpoints verified on hardware. Two runs, both completed their 600-frame benchmark (a 0xC0000005 access violation occurs at *teardown*, after all outputs are written — see Limitations).

**Version identification (C01).** Runtime version enumeration from `Cauldron.log` (identical both runs):

```
Upscaler 4.1.2 *   required_resources: 15 optional_resources:  0
Upscaler 4.1.1     required_resources: 15 optional_resources:  0
Upscaler 3.1.5     required_resources: 15 optional_resources: 48
Upscaler 2.3.4     required_resources: 15 optional_resources: 48
```

Default context creation: `Upscaler Context versionid 0xf5a5ca1e81001002, 4.1.2 *` — the closed DLL (file version 4.1.1.2740) ships **FSR 4.1.2 as its default upscaler**, newer than the API header the sample was built against (`FFX_UPSCALER_VERSION` 4.1.1; the `*` marks "newer than built-against"). FSR 3.1.5 / 2.3.4 remain selectable via the sample's UI override but were **not** selected. No 4.0.x artifact involved.

**ML path confirmation (C03).** The default-created upscaler context is the ML upscaler (FSR 4.1.2), confirmed by the context-creation log line and by the active `FFX API FSR Upscaler` GPU profile label every frame. The analytical FSR 2/3 versions were enumerated but not used.

**Execution + frames (C02/C04/C05).** Both runs executed on **"AMD Radeon RX 7700 XT"** (recorded in the JSON `GPU` field from the D3D12 device; screenshot filenames embed the adapter name). Scene: Sponza (`MainSponza.gltf`), IBL-lit, particles active. Captured frames (`build/captures/m03/*.jpg`, 1920×1061) show a fully reconstructed, sharp image with readable arch inscriptions.

**Resolutions (C07).** Requested 1920×1080 windowed → effective client area 1920×1061 display, render 1280×707 (qualityMode 1 = Quality preset, 1.5× ratio; log: `qualityMode 1 renderResolution 1280 x 720` pre-client-area).

**GPU timing (C06).**

| Label | Run 1 avg | Run 2 avg | Run 1 min/max |
|---|---|---|---|
| **FFX API FSR Upscaler** | **1.3645 ms** | **1.3528 ms** | 1.3321 / 1.5935 (run 1), 1.3353 / 1.3792 (run 2) |
| FFX API FSR Frame Generation Prepare | 0.0475 ms | — | — |
| AutomaticExposure | 0.0140 ms | — | — |
| RasterShadow (scene, dominant) | 3.3649 ms | — | — |
| GBuffer (scene) | 1.3809 ms | — | — |

Upscaler time is reproducible across runs (Δ ≈ 0.9%). Full per-pass tables in `build/captures/m03/*.json` (all 19 GPU labels + CPU labels + `CmdLine`).

**Reproduction (C08).** Command line, media bundle SHA256, DLL file versions, sample log, and both JSONs archived. Full procedure above is self-contained.

## Measurements

- Run 1: `build/captures/m03/FidelityFX FSR FFXAPI v1.3-perf-2026-10-27T13-14-35.json` (sha256 `0d3dea7e7d796a94…`), screenshot `…13-14-35.jpg` (`b87bef07db104b36…`)
- Run 2: `build/captures/m03/FidelityFX FSR FFXAPI v1.3-perf-2026-10-27T13-15-54.json` (sha256 `0a06b5d6e47441ec…`), screenshot `…13-15-54.jpg` (`2195fc8efacca4de…`)
- AvgFPS ≈ 36.9 both runs (windowed mode, DWM composited; the sample's 60 FPS GPU limiter does not reach 60 in this configuration — the per-pass GPU timings are timestamp-query based and unaffected)

## Conclusion

- [M] The official closed FSR 4 implementation (FSR Upscaling **4.1.2** default, in FSR SDK 2.3.0) executes its ML upscaling path on the RX 7700 XT unmodified, at ~**1.36 ms** per frame for 720p→1080p Quality mode — this is the project's behavioral and performance oracle. M03 gate **PASSES**.
- [M] The runtime exposes FSR 4.1.1 and 4.1.2 side by side; 4.1.1 is UI-selectable for exact-version A/B comparison later (M04/M14), while 4.1.2 is the shipping default.
- [M] FSR 4.1 RDNA 3 INT8 runs ~1.36 ms at Quality 720p→1080p — a loose upper reference for our own reproduction attempts (M12+).

## Limitations

- Whole-frame `GPUTime` label reports a broken 4×10⁻⁵ ms in benchmark JSONs (both runs; per-label timings are valid timestamp queries). Recorded as an observed quirk of the sample's whole-frame timer; per-pass evidence satisfies C06.
- 0xC0000005 at process teardown, after all outputs are written (both runs). Does not affect measurements; noted for anyone rerunning.
- Windowed client area (1920×1061) is 19 px short of the requested 1920×1080 (title bar). Recorded as-is; a `-fullscreen` run would remove the chrome if exact-display-resolution captures are needed later.
- `amd_fidelityfx_dx12.dll` v1.0.1.39157 in the driver store (M01 baseline note) played no role in this run — the sample ships its own signed runtime DLLs.

## Next

- M04 uses this baseline as oracle: reconstruct the execution graph from the 4.0.2 source mirror + lhl audit, reconciled against observed 4.1.2 behavior (dispatch counts, scratch sizes).
