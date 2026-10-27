# quattro — Native INT8 Inference on AMD RDNA 3

Standalone research and engineering project investigating **native INT8 neural-network inference on AMD RDNA 3 GPUs** (`gfx1101`), using **AMD FSR 4 Super Resolution (FSR Upscaling 4.1, INT8 variant)** as the case study.

The goal is to understand how AMD executes neural inference with native GPU capabilities and to progressively reproduce the relevant techniques from scratch.

## Platform

| | |
|---|---|
| GPU | AMD Radeon RX 7700 XT — Navi 32, `gfx1101`, RDNA 3 |
| OS | Windows 11 Pro |
| Graphics API | DirectX 12 |
| Shaders | HLSL, targeting native INT8 WMMA with INT32 accumulation |
| Dependencies | Minimal; no CUDA / ZLUDA / Vulkan / NVIDIA APIs / ML frameworks |

HIP may be introduced later strictly as a comparison point.

## Repository layout

```text
.references/      Research material (managed separately)
docs/
  CONVENTIONS.md  Evidence classification & documentation conventions
  MILESTONES.md   Milestone tracking, gates, and status
  findings/       Durable research findings with sources
  experiments/    Experiment records (question → method → result)
src/              Implementation
tests/            Validation harnesses
```

## Status

See [docs/MILESTONES.md](docs/MILESTONES.md). Engineering milestones require explicit authorization before implementation begins.

## Working rules

1. Internet research first; official AMD documentation takes priority; source code second; reverse engineering when the first two are insufficient.
2. Every claim is tagged with its evidence class — see [docs/CONVENTIONS.md](docs/CONVENTIONS.md).
3. Speculation is never presented as fact.
4. Failed experiments are preserved and recorded.
5. No milestone is marked complete without passing its executable validation gate on hardware.
