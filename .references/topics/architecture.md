# RDNA 3 architecture (focus: gfx1101 / Navi 32)

Status: starting document. Tags per [docs/CONVENTIONS.md](../../docs/CONVENTIONS.md). Access dates 2026-10-27.

## Established facts

- [D] RX 7700 XT = RDNA 3, **gfx1101**, 54 compute units, 12 GiB, wavefront size 32 or 64, LDS 128 KiB/WGP-class; gfx11 shader ISA.
  Source: AMD ROCm — Accelerator and GPU hardware specifications, https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
- [D] RDNA 3 organizes CUs as dual-CU **WGPs**; per-WGP throughput was the design focus; chiplet package (GFX die + cache dies).
  Source: AMD RDNA 3 ISA guide + AMD RDNA page, https://amd.com/en/technologies/rdna (overview); whitepaper via hub below.
- [D] Canonical ISA/microarchitecture documents (official): RDNA 3 ISA reference https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf — collected at the AMD GPU architecture documentation hub https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html (also RDNA 3.5/4 ISA, whitepapers, performance counters).
- [D] Graphics-work scheduling on RDNA 3 (MES firmware / queue manager) is publicly documented: https://gpuopen.com/download/documentation/micro_engine_scheduler.pdf
- [D] AMD publishes machine-readable ISA specs + IsaDecoder API: https://gpuopen.com/machine-readable-isa/
- [M] Our target machine: driver 32.0.32015.7029 (2026-10), D3D12 FL 12_2, WDDM 3.2 — see [environment baseline](../findings/2026-10-27-environment-baseline.md).

## Relevant implementation details

- Wave32/Wave64 both exist; WMMA instructions are wave-level — the wave mode affects operand tiling (see [wmma.md](wmma.md)).
- LDS = 128 KiB per WGP (per ROCm spec table); it is the staging area FSR 4's FP8/WMMA path relies on (see [inference.md](inference.md)).
- Navi 32 dies span 60 CU max; 7700 XT enables 54 — never tune against 7900-XT (gfx1100, 96 CU) numbers.

## Unknowns

- [U] VGPR allocation limits and occupancy behavior for our specific kernels on gfx1101 (compiler- and config-dependent).
- [U] Effective dual-issue / scheduler behavior of matrix vs vector ALU workloads on Navi 32.

## Questions requiring experiments

1. Wave32 vs Wave64 for INT8 convolution kernels on gfx1101 (occupancy × ISA study; M4).
2. Measured memory behavior for ~20 MB scratch working sets vs L2/Infinity Cache (M4).

## Sources

See inline links. Primary: AMD ROCm gpu-arch hub and gpu-arch-specs (accessed 2026-10-27).
