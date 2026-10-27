# FSR 4 architecture

Status: starting document. Tags per [docs/CONVENTIONS.md](../../docs/CONVENTIONS.md). Access dates 2026-10-27.

## Version landscape (distinguish explicitly)

- [D] FSR 4 → renamed **"FSR Upscaling"** (ML-based); FSR Upscaling 4.1 RDNA 3 support shipped in FSR SDK 2.3 (2026-06-24). Source: https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog , https://gpuopen.com/amd-fsr-upscaling
- [D-press] RDNA 4 runs FSR 4.1 in FP8; RDNA 3 build is INT8; INT8 is more demanding on RDNA 3 than FP8 on RDNA 4. Source: TechSpot (2026-07-02), https://www.techspot.com/article/3142-amd-fsr-4-upscaling-rdna-3
- [S] FidelityFX SDK v2.3.0 (public repo) contains **no FSR4 source** — only docs/media (`Kits/FidelityFX/docs/techniques/media/super-resolution-ml/`); FSR 4.1 ships closed. Verified locally @ `60f4ea8` (glob for `*fsr4*`).
- [S] The **FSR 4.0.2 MIT source** (mirror @ `c5e34b4`) is therefore the only public source-level view of FSR 4 internals; its model id is `fsr4_model_v07`, shared with 4.1 per the mirror's README.

## Network architecture (as observed / reported)

- [S-lhl] v07 = quantized CNN, encoder→bottleneck→decoder with skip connections:
  `encoder1` strided 2×2 conv (downsample) → `encoder2` 2× ConvNext-style residual blocks (depthwise + expand/contract pointwise) → `encoder3` 2× FasterNet-style blocks (spatial-mixing partial conv) → `bottleneck` 2× residual blocks + transposed 2×2 conv (upsample) → `decoder3/2/1` mirroring encoder stages → post: RCAS sharpening + SPD downsample + auto exposure.
  Source: lhl/fsr4-rdna3-optimization @ `49015b7`, README "FSR4 Model Overview" (their audit of the v07 generated HLSL). **Corroborate against the archive locally before treating as [S]-direct.**
- [S] Execution shape @1080p: 14 model passes + 13 post passes, ~20 MB scratch, inputs NHWC with 7 channels (current + history frame data); resolution-specialized shader sets exist for 1080p/2160p/4320p.
  Source: lhl repo README; pass file names confirmed in archive (`passes_1080/2160/4320.hlsl`).
- [S] Per-variant files in the archive: `pre.hlsl`, `passes_*.hlsl`, `post.hlsl`, `initializers.bin` per tier; shared `pre_wmma.hlsl`, `post_wmma.hlsl`, `rcas.hlsl`, `spd_auto_exposure.hlsl`, `debug_view.hlsl`.
- [U] Full tensor schema (mirror README mentions 78 tensors with exact shapes/offsets — from the companion RE project; re-derive locally in an experiment).

## Unknowns

- [U] 4.0.2 → 4.1 deltas (RDNA 3 INT8 execution details in 4.1, dispatch contract changes).
- [U] How the temporal-history channels are composed and updated frame-to-frame.
- [U] Whether "native"/"drs" tiers differ beyond tile sizes.

## Questions requiring experiments

1. Map the pass graph: for each of the 14 model passes, list tensor inputs/outputs and shapes from generated HLSL + initializers (EXP-002 candidate).
2. Identify where `dot4add_i8packed` vs WMMA vs plain F16/F32 math occurs per pass (feeds M5 kernel design).

## Sources

Inline links; submodules pinned per [README](../README.md); press sources in [environment baseline](../findings/2026-10-27-environment-baseline.md).
