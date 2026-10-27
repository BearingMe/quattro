# EXP-001: Does the minimal build → run → test loop work reproducibly on the target machine?

Date: 2026-10-27
Milestone: M01

## Question

Can the bootstrap toolchain (CMake + VS 2022 MSVC 14.44, Windows SDK 26100) configure, compile, execute, and test a minimal C++20 project on the gfx1101 target machine, reproducibly from documented commands?

## Method

1. Root `CMakeLists.txt` + `CMakePresets.json` — `debug`/`release` presets, Visual Studio 17 2022 x64 generator (MSVC `cl.exe` is not on PATH; the VS generator locates it itself), binary dirs `build/debug` and `build/release`, executables flattened into `build/bin/`.
2. `src/main.cpp` — experiment CLI skeleton: `list`, `run <name>`, `help`. Exit-code contract: 0 success · 1 experiment failure · 2 usage error. No experiments registered (M01 has none — registration happens as milestones authorize them).
3. Smoke tests wired through CTest directly against the built binary's exit-code contract (no test sources exist yet; a stub test file would violate the no-placeholder rule).
4. All commands run from a clean tree (no pre-existing build directory).

## Result

[M] Every command succeeded on the first attempt; both presets verified.

- `cmake --preset debug` → configure OK. Compiler: MSVC 19.44.35229.0 (`cl.exe` 14.44.35207 toolset via VS 2022 Community generator), Windows SDK 10.0.26100.0. Build files written to `build/debug`.
- `cmake --build --preset debug` → `main.cpp` compiled with `/W4`, zero warnings; produced `build\bin\rdna3-int8.exe`.
- `build/bin/rdna3-int8.exe list` → prints `registered experiments: (none yet — see docs/MILESTONES.md)`, exit 0.
- `ctest --preset debug` → **3/3 passed** (`cli_list`, `cli_help`, `cli_run_unregistered` via `WILL_FAIL`), 0.04 s.
- Exit-code contract verified manually: `run does-not-exist` → stderr message, exit 2; `bogus` → stderr usage, exit 2; `list`/`help` → exit 0.
- `cmake --preset release` + `cmake --build --preset release` → configure and build OK, same output path.
- `git status` after builds: only the three new source files — no generated artifacts escaped `build/` (`.vs/`, intermediate objects all ignored).
- Toolchain versions: CMake 4.4.3, MSBuild 17.14, MSVC 19.44.35229, Windows SDK 10.0.26100.0.

## Interpretation

- [M] M01 gate items "CMake configuration succeeds", "C++ compilation succeeds", "Executable runs successfully", and "Build instructions are reproducible" all hold on the target machine. Combined with the environment baseline ([M] GPU/driver/toolchain in `findings/2026-10-27-environment-baseline.md`), the M01 validation gate **PASSES**.
- [M] Reproducibility: the recorded commands are the full procedure — no manual steps, no vcvars invocation needed (VS generator resolves MSVC), no state outside `build/`.
- [D] Build-system policy from `docs/architecture.md` §2 holds in practice: one CMake, named presets, all output under `build/`.

## Next

- M05 introduces the first D3D12 work — first real use of `dxc.exe` and of explicit physical-adapter selection (SudoMaker virtual adapter shares the AMD PCI IDs).
- Experiments register in `src/main.cpp` explicitly as their milestones authorize them (next: none until M05/M06).
- CMake ≥ 4.x note: `CMakePresets.json` must not carry `subminor` inside `cmakeMinimumRequired` (rejected by CMake 4.4).
