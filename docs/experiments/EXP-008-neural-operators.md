# EXP-008: Source-justified neural operations and explicit precision boundaries

Date: 2026-10-27 · Milestone: M10

## Question and scope

- [H] The non-matrix constituents of the selected FSR 4.0.2 v07 INT8 Quality
  main-pass architecture can be represented by a small typed operator
  collection, validated independently of model weights and convolutions.
- [S] The complete call/bias inventory, manually selected overloads, arithmetic
  anchors and storage contract are in
  [the M10 finding](../findings/2026-10-27-m10-operator-inventory.md).
  Source pin: `.references/sources/FSR-4.0.2-reference` at `c5e34b4`.
  FSR 4.1 behavior and temporal/image pre/postprocessing are not inferred.

## Environment and reproduction

- [M] RX 7700 XT, gfx1101, PCI vendor/device 1002:747e; Windows 11;
  reported AMD driver `32.0.32015.7029`.
- [M] DXC `1.8.2502.11 (239921522)`, SDK `10.0.26100.0`, VS2022/MSVC;
  shader flags `-T cs_6_6 -O3 -HV 2021 -denorm preserve`.
- [S] A single CLI/CTest registration runs the collection. Reports and raw
  tensors live under `build/reports/m10/`, never in tracked source directories.

```powershell
python scripts/m10_inventory.py
cmake --preset debug
cmake --build --preset debug
build/bin/rdna3-int8.exe run tensor-ops
ctest --preset debug --output-on-failure --output-log build/reports/m10/debug-ctest.log
cmake --preset release
cmake --build --preset release
build/bin/rdna3-int8.exe run tensor-ops
ctest --preset release --output-on-failure --output-log build/reports/m10/release-ctest.log
```

- [S] Rebuild immediately before each preset's tests: Debug and Release share
  `build/bin`. Allow >135 seconds for the full suite; a 120-second shell timeout
  interrupted an initial regression run during M09, not an operator failure.

## Method

1. [S] `scripts/m10_inventory.py` enumerates 14 main-pass calls, FP16 bias
   declarations and 13 padding passes. Overload/arithmetic tracing is manual.
2. [S] The GPU runner validates typed NHWC ranges, strides, scales, biases,
   numerical bounds and dispatch limits before upload. It sends a 128-byte
   SRV header, input storage and per-channel transforms; no model-specific GPU
   dependencies, graph compiler or additional executable is introduced.
3. [S] The independent CPU reference uses scalar coordinate loops, explicit
   FP32 rounding points and floor/fraction ties-to-even. The GPU uses packed
   dword ownership, `precise` arithmetic and explicit integer-bit FP16 RNE
   packing. All output bytes, including zero tail padding, are compared.
4. [S] Each case compares the post-warmup and final snapshots after three
   repeated dispatches against the reference. UAV barriers order successive
   writes. Debug layer and GPU-based validation are enabled; no M10 diagnostic
   filtering is allowed. Timing is incidental, not an M10 performance claim.
5. [S] Each case emits `.input.bin`, `.cpu.bin`, `.gpu.bin` and descriptor
   `.txt`, enabling shape/stride/type/transform reconstruction and inspection.
   The input file excludes the runner's generated SRV header; descriptor offsets
   therefore address that file directly.

## Results

- [M] **60 hardware cases pass**, with byte-exact CPU/GPU agreement at both
  snapshots. **24 rejection/boundary checks pass** (18 invalid-operation cases
  plus six accumulation-boundary checks). Detailed stdout:
  `build/reports/m10/debug-operators.log`.
- [M] Regression verification before the final review fixes: Debug **9/9**
  and Release **9/9**, including all previous WMMA/quantization/kernel tests.
  Final post-review regression results are recorded below once complete.

| Coverage | Measured evidence |
|---|---|
| ReLU | All 256 signed-byte encodings; channels 1,7,16,32,64 |
| Strided copy/split, concat and zero pad | Odd logical sizes, pixel and row padding, unaligned INT8 channel offsets, same channel sweep; literal concat/split expected bytes |
| Affine requantization and residual | INT32→FP32, FP16 bias, independently scaled residuals, channel/group output scales, optional ReLU, signed saturation; both INT32 endpoints |
| Floating quantization | FP16 ties and saturation; all 63,488 finite FP16 encodings (nonfinite encodings replaced with zero in the sweep); FP32 input plus FP16 bias |
| Half output | Explicit two-round anchor, subnormal/normal exponent sweep (16,864 values), FP16 overflow |
| Spatial scatter | INT8/FP16/INT32 phase tensors; literal phase-labelled two-pixel anchor; **1920×1080×8 FP16**, 33,177,600 output bytes, crossing the X dispatch limit |
| Contraction boundary | acc=4095, scale=1.0001220703125, bias=-4094 → separately rounded result 1.5 → INT8 **2**; `std::fma` produces <1.5, which would quantize to 1 |
| FP32 denormals | acc=2^30, scale=2^-140, reciprocal=2^111 → **2**; normal-scale residual cancellation producing a subnormal intermediate, then reciprocal 2^127 → **1** |
| Invalid input rejection | NaN/Inf quantization values; type, shape, stride/alignment/range errors; invalid/unsafe scales; invalid concat/pad/phase shapes |
| Accumulation safety | Reusable full-range policy accepts K=131071, rejects K=131072, bias headroom exhaustion, UINT64_MAX terms, zero terms and excessive bias |

### Negative evidence and repairs

1. [M] Initial half-output test failed at acc=65520, scale=1, bias=0:
   the `f32tof16`-based path produced **0x7bff**, while the declared IEEE RNE
   reference expected **0x7c00**. Saved output:
   `build/reports/m10/failed-f32tof16-overflow.gpu.bin`.
2. [M] Adding only an explicit overflow threshold fixed that fixture but the
   exponent/subnormal sweep still failed. Its second element produced
   **0x0310** vs **0x0314**. Saved output:
   `build/reports/m10/failed-f32tof16-rounding.gpu.bin`. The final shader uses
   explicit integer-bit RNE packing for normal, subnormal and overflow cases;
   the same fixtures now pass. These observations do not identify a universal
   intrinsic/driver cause or establish AMD native-half overflow semantics.
3. [S] Adversarial review identified accepted subnormal scales without an
   enforced GPU mode. The shader now uses `-denorm preserve` and both the
   subnormal-scale and subnormal-intermediate fixtures pass on hardware.
4. [S] Review also requested source-derived literal phase/concat/split anchors
   to reduce CPU/GPU shared-formula blind spots; these are included above.

## Interpretation and boundaries

- [M] The selected **non-matrix constituent set** is implemented and passes
  independent hardware checks. Layouts, mixed precision, and inspectable
  outputs are explicit; no speculative neural operators were added.
- [S] Final FP16 rounding and noncontracted FP32 arithmetic are defined by our
  contract. Source formulas justify the operations but are not proof of
  bit-equivalence to every AMD fused shader/association/rounding mode.
- [S] Quantization boundaries include dequantizing INT32/INT8 into FP32 within
  affine/residual operations and converting INT32 results to FP16. There is no
  unnecessary standalone dequantization dispatch.
- [S] This remains **isolated operator validation**, not GPU-resident inference.
  Spatial scatter is not learned convolution. M09 still uses CPU im2col and
  accumulator readback/re-upload; M11 must implement resident execution.
- [U] M09 cbuffer/analyzer and Fill failures remain scoped observations, not
  universal limitations. Its validation-on benchmark is not a clean-path
  performance ceiling. Its contracted requant path is not silently changed
  or certified at rounding boundaries by this experiment.

## Final regression and gate

- [M] **Final rebuilt Debug: 9/9 PASS**, 139.91 s total; M10 test 16.89 s.
  **Final rebuilt Release: 9/9 PASS**, 133.25 s total; M10 test 16.25 s.
  Logs: `build/reports/m10/debug-ctest.log`, `release-ctest.log`.
- [M] Direct Debug and Release operator runs both exit 0, reporting **60
  hardware cases and 24 rejection/boundary checks PASS**. The direct logs are
  `build/reports/m10/debug-operators.log` and `release-operators.log`.
- [S] `@adversary` reviewed before commit, then verified the denormal-mode,
  numerical-bound, literal-layout-anchor and documentation repairs. Follow-up
  verdict: **no remaining substantive blockers**, conditional on the final
  regressions above. `git diff --check` passes.
- [M] M10 gate **PASS on gfx1101**; all six checkpoints and six gate items are
  backed by the source inventory, operator tests and inspectable snapshots.
  M11 is not implemented or claimed by this milestone.

## Next

- [S] Stop at the M10 gate. M11's fixed-network GPU-resident execution remains
  a separate milestone; no network graph or model-specific runtime is hidden
  in this change.
