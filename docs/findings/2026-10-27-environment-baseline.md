# Environment baseline (M0)

Date: 2026-10-27
Milestone: M0 (gate record)
Type: environment verification — measurements on the target machine plus documented facts.

## Summary

Target hardware and OS verified on-machine. Project identity confirmed: **RX 7700 XT = Navi 32 = gfx1101** (AMD official docs + observed PCI IDs agree). D3D12 feature level 12_2, WDDM 3.2. Toolchain present: VS 2022, Windows SDK 26100 (DXC 1.8), CMake/Ninja/Python/Git. HIP SDK intentionally absent. Key open question carried forward: SM 6.9 Cooperative Vector / DirectX Linear Algebra driver support on gfx1101 (decides the M3 path).

## Hardware

- [M] Adapter: **AMD Radeon RX 7700 XT**, VideoProcessor ID `0x747E`, Vendor `0x1002`, dedicated VRAM 12 080 MB, driver **32.0.32015.7029** (dated 2026-10-17). Measured via `Win32_VideoController` and dxdiag XML export.
- [D] RX 7700 XT is RDNA 3, **gfx1101**, 54 CUs, 12 GiB.
  Source: AMD ROCm — Accelerator and GPU hardware specifications, https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html (accessed 2026-10-27)
- [D] PCI device `1002:747E` = Navi 32 [Radeon RX 7700 XT / 7800 XT] — matches the observed `0x747E`.
  Source: linux-hardware.org PCI database, https://linux-hardware.org/?id=pci:1002-747e-1da2-d475 (accessed 2026-10-27; secondary source corroborating the AMD mapping)
- [M] D3D12 feature levels: `12_2, 12_1, 12_0, ...`; driver model **WDDM 3.2**. Measured via dxdiag XML export.
- [M] A second adapter, **"SudoMaker Virtual Display Adapter"** (WDDM 3.2), is present, and its dxdiag entry reports the *same* DeviceID `0x747E`, Vendor `0x1002`, and VRAM as the physical card. Consequence: M1 must select the physical AMD adapter explicitly (e.g., by enumeration + verification), never "the first adapter".

## OS

- [M] Windows 11 Pro, 10.0.26200, x64.

## Toolchain

- [M] Visual Studio 2022 Community at `C:\Program Files\Microsoft Visual Studio\2022\Community`; MSVC toolset **14.44.35207** present. `cl.exe` not on PATH → build via `vcvars` or CMake generator `Visual Studio 17 2022`.
- [M] Windows SDK **10.0.26100.0**; `dxc.exe` version **1.8.2502.11** at `Windows Kits\10\bin\10.0.26100.0\x64`.
- [M] CMake, Ninja, Python 3.13, Git available on PATH.
- [M] HIP SDK not installed (`C:\Program Files\AMD\HIP` absent) — consistent with the plan (D3D12/HLSL first; HIP only as a later comparison).
- [M] Adrenalin driver store present (`u0205306.inf`, driver 32.0.32015.7029, files dated 2026-10-18). Note: `amd_fidelityfx_dx12.dll` v1.0.1.39157 exists in `system32` — a driver-side FidelityFX component; potential relevance to M6 (FSR 4 execution path). Role: [U].

## Case-study background

- [D] FSR Upscaling 4.1 (formerly FSR 4) RDNA 3 support shipped in the FSR SDK 2.3 update (2026-06-24).
  Source: AMD GPUOpen — "AMD FSR Upscaling 4.1 RDNA 3 Support Now Available in FSR SDK 2.3 Update", https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog (accessed 2026-10-27)
- [D] AMD states DirectX SM 6.9 Cooperative Vector / DirectX Linear Algebra give HLSL "direct access to AMD WMMA cores from shaders on AMD Radeon graphics cards".
  Source: AMD GPUOpen — "AMD and Microsoft partner on DirectX ML… at GDC 2026", https://gpuopen.com/learn/amd-microsoft-gdc-2026 (accessed 2026-10-27)
- [D] Microsoft: Cooperative Vector introduced in SM 6.9 (preview 2025-06-02, retail 2026-02-26); DirectX Linear Algebra announced 2026-03-12 as the successor unifying vector-matrix and matrix-matrix ML ops in HLSL.
  Sources: DirectX Developer Blog — https://devblogs.microsoft.com/directx/cooperative-vector, https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more, https://devblogs.microsoft.com/directx/evolving-directx-for-the-ml-era-on-windows/ (accessed 2026-10-27)
- [D] Press-level documentation: RDNA 4 executes FSR 4.1 in FP8; the RDNA 3 build is an **INT8** model of the same (or near-identical) network; INT8 is more demanding on RDNA 3 than FP8 on RDNA 4.
  Sources: TechSpot — https://www.techspot.com/article/3142-amd-fsr-4-upscaling-rdna-3 (2026-07-02); gamegpu — https://en.gamegpu.com/news/zhelezo/glubokoe-testirovanie-tekhnologii-masshtabirovaniya-amd-fsr-4-1-na-arkhitekture-rdna-3 (2026-07-01, summarizing Hardware Unboxed / ComputerBase) (accessed 2026-10-27). Treated as `[D]` from secondary press pending primary AMD statements.

## Open questions (feeding M1–M3)

- [U] Does driver 32.0.32015.7029 expose SM 6.9 **Cooperative Vector** (and/or DirectX Linear Algebra) on gfx1101 via `CheckFeatureSupport`? `[H]` likely yes (Oct 2026 driver + AMD GDC 2026 statements) — must be verified, not assumed.
- [U] Which INT8 operations map to WMMA on RDNA 3 through Cooperative Vector, and what is emulated where hardware doesn't support an op?
- [U] Exact semantics of INT8→INT32 accumulation reachable from HLSL (M2/M3 experiments).
- [U] DXC shipped with the SDK (1.8.2502) predates SM 6.9 retail (DXC 1.9.2602+); M3 will require a newer standalone DXC plus Agility SDK. Not installed yet.
- [U] Role of `amd_fidelityfx_dx12.dll` and driver-store components in the FSR 4 RDNA 3 path (M6).

## Sources

See inline citations. Raw dxdiag export retained at `%TEMP%\opencode\quattro_dxdiag.xml` (not committed).
