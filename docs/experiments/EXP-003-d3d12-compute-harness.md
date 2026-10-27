# EXP-003: Does the minimal D3D12 compute harness execute, validate, and time correctly on the RX 7700 XT?

Date: 2026-10-27
Milestone: M05

## Question

Can a minimal native D3D12 compute harness (explicit adapter selection → HLSL→DXC→CSO → upload → dispatch → readback → fence sync → GPU timestamps → debug diagnostics) run correctly and reproducibly on the gfx1101 target — the foundation all later GPU milestones build on?

## Hypothesis

Plain D3D12 compute (SM 6.0, no vendor extensions) on the explicitly selected physical AMD adapter produces bit-exact results vs. a CPU reference, with stable per-dispatch GPU timestamp measurements and a clean debug/GPU-based-validation log.

## Reference sources

- [D] Microsoft Direct3D 12 docs: buffer heap/state rules (DEFAULT=COMMON, UPLOAD=GENERIC_READ, READBACK=COPY_DEST), timestamp queries via `EndQuery`+`ResolveQueryData`, debug layer + GPU-based validation (`ID3D12Debug1`).
- [D] DXGI adapter enumeration: `DXGI_ADAPTER_DESC1::AdapterLuid`; software adapters flagged `DXGI_ADAPTER_FLAG_SOFTWARE`.
- Project context: [M] SudoMaker virtual display adapter shares PCI IDs + VRAM with the physical card (see [`findings/2026-10-27-environment-baseline.md`](../findings/2026-10-27-environment-baseline.md)).

## Method

1. `src/gpu/` — device module: full adapter enumeration with dump, explicit selection (hardware + "Radeon" in description + highest dedicated VRAM), debug layer + GPU-based validation + info queue, fence sync. Compute module: offline-CSO loading (relative to the executable), root signature (2×root SRV + UAV + 32-bit constant), upload/readback heaps, dispatch, timestamp pairs per dispatch.
2. `src/gpu/shaders/vector_add.hlsl` — float32 vector add, `[numthreads(256)]`, compiled **offline by the project build** with the Windows SDK `dxc.exe` (`-T cs_6_0`, CMake custom command → `build/bin/shaders/vector_add.cso`).
3. `src/reference/` — independent CPU reference (`a[i] + b[i]`, same fp32 op per element → bit-exact expected) + bit-wise comparator.
4. `src/experiments/compute harness` (`rdna3-int8.exe run d3d12-compute`): 1 048 576 elements (4 096 KiB/field, 12 MiB moved per dispatch), 10 warmup + 100 timed dispatches, two readback snapshots (post-warmup and post-timed) compared bit-exactly, per-dispatch GPU timestamps, info-queue verdict. Exit 0 = pass.
5. Methodology note: single command list records warmup → snapshot → timed loop (timestamp stamp before/after each dispatch) → final snapshot → query resolve; one execute + one fence wait — snapshot/timed-loop results are validated separately.

## Result

[M] **PASS on hardware — both Debug and Release configurations.** Exit 0, CTest 4/4 both presets.

```
adapter: AMD Radeon RX 7700 XT (vid=0x1002 did=0x747E, 12080 MiB, luid 0xE7EB)
workload: float32 vector add, 1048576 elements (4096 KiB/field)
dispatches: 10 warmup + 100 timed (4096 groups x 256 threads)
gpu time (Debug):  min 12.080 us, avg 12.980 us, max 70.200 us
gpu time (Release): min 12.200 us, avg 13.013 us, max 73.800 us
correctness: post-warmup mismatches 0, post-timed mismatches 0 (bit-exact)
diagnostics: debug layer + GPU-based validation enabled, no error-severity messages
```

- [M] Explicit adapter selection works: the physical RX 7700 XT is chosen and verified by name/IDs/LUID despite the virtual display adapter being present.
- [M] Offline DXC compilation through CMake produces a working CSO (SM 6.0).
- [M] Correctness is bit-exact on both snapshots — upload, dispatch, and readback are deterministic and the fence synchronization is correct.
- [M] Timing is stable: avg 12.98/13.01 µs (Debug/Release), min ≈ 12.1 µs; the max (~70 µs) is the first timed dispatch (cache/clock warm-up after the snapshot barrier), tail excluded from any future comparison methodology — recorded, not hidden.
- [M] Throughput sanity: 12 MiB moved in ~13 µs ≈ 0.9 TB/s effective for this tiny latency-bound dispatch — plausible; not a bandwidth benchmark (that is M07+).

## Interpretation

- [M] All six M05 gate items hold: dispatch executes; results match an independent CPU reference; synchronization is correct (bit-exact double snapshot is impossible otherwise); diagnostics report no relevant errors; GPU time is measured; repeated executions are consistent (100 timed dispatches + 2 configs + CTest repetitions).
- [S] Dependency boundaries hold in practice: `gpu` knows nothing about the reference comparison (the experiment layer compares `gpu` outputs against `reference`), `reference` never touches GPU APIs.

## Debugging trail (recorded per conventions — failed attempts are evidence)

Built through these real issues, each caught or confirmed by diagnostics:
1. `ComPtr<T>` requires complete types (d3d12.h in the header, not forward declarations).
2. `windows.h`'s `GetMessage` macro mangles `ID3D12InfoQueue::GetMessage` at header-parse time — `#undef GetMessage` must precede `<d3d12.h>`; `NOMINMAX` needed for `std::min/max`.
3. `d3dx12.h` is not in the Windows SDK — replaced with local struct helpers (also keeps third-party headers out).
4. Buffers: `MipLevels` must be 1; READBACK heap requires initial `COPY_DEST` (E_INVALIDARG otherwise); UAV state requires `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` — this one **faulted the device** and was pinpointed by the debug-layer info queue (M05 C07 proven by use).
5. Fence fast-path bug: skipping `SetEventOnCompletion` when `GetCompletedValue()` already satisfies the value, then waiting on the (never set) auto-reset event anyway → guaranteed timeout. Diagnosed via "completed value 1, wait result 258 (timeout)" — the wait must only happen when completion has not already occurred.
6. `runExperiment` compared `const char*` by pointer, not contents — registration lookup always failed; found via `list` vs `run` asymmetry.

## Next

- M06 uses this harness for the INT8 WMMA/CoopVec probes: replace the CSO and root layout, keep upload/dispatch/readback/timestamps/adapter selection as-is.
- Timestamp methodology (tail exclusion, warmup counts) inherits from here.

## Files

- `src/gpu/device.{h,cpp}`, `src/gpu/compute.{h,cpp}`, `src/gpu/shaders/vector_add.hlsl`
- `src/reference/reference.{h,cpp}`, `src/experiments/experiments.{h,cpp}`, `src/main.cpp`
- `CMakeLists.txt` (sources, `DXC_EXECUTABLE` cache var, shader target, `experiment_d3d12_compute` test)
