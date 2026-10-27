# EXP-005: Does native FP16 WMMA execute correctly on gfx1101, and how do FP16/INT8/DP4A compare on equivalent workloads?

Date: 2026-10-27
Milestone: M07

## Question

Can FP16 WMMA (F32 and F16 accumulators) run natively and correctly on the RX 7700 XT through the M06-established AMD intrinsics path — and what are the measured hardware differences between FP16 WMMA, INT8 WMMA, and INT8 DP4A on comparable matrix workloads?

## Hypothesis

The F16 operand formats (`WMMA_F32_16X16X16_F16`, `WMMA_F16_16X16X16_F16`) work exactly like the INT8 modes validated in EXP-004; on equal GEMM workloads WMMA outperforms DP4A because one wave-instruction computes 4096 MACs.

## Reference sources

- [S] FSR-4.0.2-reference @ `c5e34b4` — vendored `AmdExtD3D` headers (`src/gpu/amd/`), FSR4's own WMMA usage (`pre_wmma.hlsl` L212–243).
- [D] RDNA 3 ISA reference: WMMA cycle counts and operand formats (theoretical-peak formula); IEEE-754 exactness of fp16/fp32 arithmetic.
- [M] EXP-004 (M06): the validated harness, extension runtime, and analyzer evidence path.

## Method

1. **FP16 tile probes** (`src/gpu/shaders/fp16_wmma.hlsl`, `-T cs_6_6 -enable-16bit-types`): one 16×16×16 tile per Wave32 group; F16 inputs (row-major 32 B/row), F32 or F16 accumulator. Probe values: A ∈ [−1, 1], B ∈ [−0.75, 0.75] (multiples of 2⁻³), initC ∈ [−4, 4]; nonzero accumulator init included.
2. **CPU reference**: fp16→fp32 exact unpack; per-product exact in fp32 (11-bit mantissas → 22-bit products); sequential accumulation. Hardware intra-instruction accumulation structure is undocumented ([U]) → tolerance compare with documented bounds (2e-3 fp32-acc; 0.06 fp16-acc ≈ order-swap budget of 16 adds at magnitude ≤ 16).
3. **Multi-tile comparison** (`gemm_compare.hlsl`): 64 independent 16×16×K GEMMs per dispatch (the dispatch geometry carries the GEMM count) — INT8 WMMA (K-loop of chained `AmdWaveMatrixMultiply`), FP16 WMMA (F32 acc), INT8 DP4A control — all from the same staged bytes, same initC, K ∈ {64, 256}. Fixed-graph CSOs (K is a `-D` define, one CSO per kernel × K; FSR4/ML2Code codegen style) because the driver's analyzer rejects runtime-bounded K loops over magic-code sequences (E_INVALIDARG — recorded in the debugging trail).
4. **Experiment** (`rdna3-int8.exe run wmma-fp16`): analyzer PSOs + ISA dumps for every (kernel, K) configuration to `build/reports/m06/`, 10 warmup + 100 timed dispatches with timestamp pairs, correctness per kernel, info-queue verdict.

## Result

[M] **PASS on hardware — both Debug and Release configurations.** Exit 0, CTest 6/6 both presets. Canonical record (Release):

```
fp16 wmma ->f32 acc  | maxdiff 1.19e-07 (tol 0.002) OK | v_wmma_f32_16x16x16_f16 x1 | vgpr 33
fp16 wmma ->f16 acc  | maxdiff 0        (tol 0.06)  OK | v_wmma_f16_16x16x16_f16 x1 | vgpr 33
multi-tile: 64 gemms of 16x16xK, 10 warmup + 100 timed dispatches
kernel       vgpr   min@K64    avg@K64   min@K256   avg@K256  scale
int8 wmma      52   74.04us    74.42us   179.24us   253.68us  2.42x
fp16 wmma      83   71.16us    76.07us   218.00us   322.55us  3.06x
int8 dp4a      83   15.68us    21.48us    80.68us    85.21us  5.15x
(Debug run mins agree: 69.7-74 / 71 / 15.8 us at K64; K-scale 2.4-2.5x / 2.4-3.1x / 4.9-5.2x)
```

### Findings

- [M] **FP16 WMMA correctness (C01/C02): F32-accumulator maxdiff 2⁻²³ (1 ULP); F16-accumulator bit-exact (maxdiff 0).** The probe values are multiples of 2⁻³: products are exact multiples of 2⁻⁶ (exact in fp16 *and* fp32), and F16-acc partial sums (≤ 16) need ≤ 11 significant bits — exactly the fp16 mantissa — hence order-independent bit-exactness. The F32-acc result shows a 1-ULP deviation from the sequential reference at magnitude ~1: the hardware's internal accumulation structure is *not* plain left-to-right fp32 adds ([U] — candidates: different add order, fused C-add stage, or internal rounding placement), but the deviation is 4 orders of magnitude inside the tolerance. Exactness reasoning documents *why* F16-acc must be bit-exact and bounds F32-acc; the measured 1 ULP is the first concrete evidence of the hardware's internal structure.
- [M] **Native FP16 WMMA instructions confirmed (C03):** `v_wmma_f32_16x16x16_f16` (F32 acc) and `v_wmma_f16_16x16x16_f16` (F16 acc) — present in every WMMA configuration's ISA, absent from the DP4A control (`build/reports/m06/gemm_*.isa`, all 6 dumped).
- [M] **Accumulator/operand differences (C05):** single-tile probes use 24 VGPRs (INT8) vs 33 (FP16); the equivalent GEMM kernels 52 (INT8) vs 83 (FP16) — wider fragments and fp32 staging cost registers. F16-acc doubles effective accumulator capacity at fp16 precision.
- [M] **Timing is reproducible (gate):** per-run min-of-100 is stable; Debug/Release agree within ~5% (69.7–74.0 µs WMMA; 15.7–16.5 µs DP4A at K=64). Cross-run K256 values show clock-state variance (fp16 gemm min 170–218 µs across process runs) — within-run comparisons (the K-scaling) are the reliable metric.
- [M] **Arithmetic vs memory (C07):** INT8 and FP16 WMMA scale **2.4–3.1×** for 4× the K — overhead-bound: the driver-generated per-tile operand gather (fragment loads + `ds_bpermute` cross-lane redistribution) dominates, not the matrix math. DP4A scales **4.9–5.2×** ≈ compute-bound. DP4A wins **4.2–4.7× at K=64** (15.7–16.5 vs 71–74 µs) but **2.2–2.7× at K=256** (81 vs 179–218 µs) — the WMMA prologue amortizes with K.
- [M] **Theoretical vs observed (gate):** one WMMA wave-instruction = 4096 MACs vs DP4A's 128 MACs/lane-instruction (32 lanes × 4) — comparable per-lane-instruction arithmetic; the WMMA advantage exists only when operand staging is amortized (LDS staging, tile reuse — FSR4's own optimized path stages through `groupshared`, see `pre_wmma.hlsl`). Our first-cut memory-per-tile path wastes that advantage. This matches the M04 report's memory-overhead sensitivity (F3) and the lhl harness lesson (F6): memory movement dominates this workload class.
- [H] A LDS-staged WMMA kernel (stage tiles once, chain WMMA from `groupshared`) would close much of the gap — that is M15 optimization territory, recorded as the direction, not implemented here (M07 boundary: no full-network optimization).

### Occupancy/register investigation (C06)

- [M] Analyzer stats per kernel: VGPRs 52 (int8 WMMA GEMM), 83 (fp16 WMMA GEMM), 83 (dp4a GEMM), 24/33 (single-tile probes); LDS 0 B/group everywhere (no `groupshared` in these kernels); physical budget 1536 VGPRs, 2048 SGPRs per the analyzer.
- [M] All kernels dispatch exactly one wave per group with `numthreads(32)` (Wave32 mandate of the WMMA ISA); occupancy is group-count-driven here — no LDS-limited or VGPR-limited occupancy cliffs at these sizes (52–83 VGPRs ≪ 1536 budget; the per-SIMD wave slots, not VGPRs, bound this shape).
- [H] The theoretical-peak formula from the ROCm guide (ops/instruction ÷ cycles × frequency) would predict WMMA ≈ 2× DP4A on int8 — the observed 0.21–0.24× at K=64 is entirely prologue; separating the two is only meaningful with staged operands.

## Debugging trail (recorded per conventions)

1. **Analyzer rejects runtime-K-loop GEMM kernels (E_INVALIDARG 0x80070057)** — magic-code atomic sequences must be straight-line/fully-unrolled with compile-time constants; a `[unroll]` loop with a runtime `break` also fails. Fix: fixed-graph CSOs per (kernel, K) via `-D GEMM_K_TILES` — exactly FSR4's codegen philosophy (one pass per tile group, no runtime K loops).
2. **My DP4A GEMM kernel indexed A/B with the global K index inside each 16×16 tile** (`kk = t*16 + local` instead of the tile-local `kk`) — reads for t≥1 ran off the tile into the initC region; results were element-dependently short by a fixed set of contributions. Caught because the (correct) WMMA GEMM passed while the "simple" control failed; the WMMA kernels' ISA was verified instruction-by-instruction before trusting that asymmetry.
3. **Unsigned-underflow in the fp16 probe data generation** — `float((i % 17) - 8)` evaluates `(i%17) - 8` in uint32, wrapping to ~4.29e9 for i%17 < 8 (×0.125 = 5.37e8 → fp16 +inf → NaN through the products). The GPU and the CPU reference agreed **bit-exactly through the inf/NaN propagation** (correct behavior on both sides — same garbage in, same garbage out), which is why the first "maxdiff 0" result was meaningless. Found only after adding a NaN guard to the comparator: the old `std::max(maxDiff, fabs(diff))` silently skipped NaN diffs. Fix: all data arithmetic in float (`float(i % 17) * 0.125f - 1.0f`); comparator now propagates NaN as a hard failure.
4. The temporary GPU-side input-buffer verification (readback + memcmp) ruled out the upload path during the investigation — removed after the root cause was found (it was M07 debug scaffolding, not a gate check; the gate checks — CPU-reference compare, ISA counts, info queue — were never removed).

## Interpretation

- [M] All seven M07 gate items hold: FP16 WMMA valid (≤ 1 ULP / bit-exact with documented reasoning), native FP16 WMMA confirmed in ISA, comparable matrix workloads across precisions (same GEMM shapes), accumulator differences documented, repeated timing (2 configs × 100 dispatches × 2 K values), warm-up/measurement methodology documented (10 warmup + 100 timed, min-based, tail excluded), theoretical vs observed throughput distinguished.
- [M] The compute-path decision from M06 stands and gains nuance: native WMMA works and is correct, but the driver-generated operand path needs LDS staging to be competitive — DP4A remains the throughput baseline to beat at current shapes.

## Next

- M08 (quantization) is independent of this result; the LDS-staged WMMA kernel and the scalar-vs-packed question on gfx1101 land in M09/M15 where they belong.

## Files

- `src/gpu/shaders/fp16_wmma.hlsl`, `src/gpu/shaders/gemm_compare.hlsl`
- `src/reference/reference.{h,cpp}` (fp16 conversion + matmul references, NaN-guarded comparators), `src/experiments/experiments.{h,cpp}`, `src/main.cpp`
- `CMakeLists.txt` (6 new CSO targets, `experiment_wmma_fp16` test)
- ISA evidence: `build/reports/m06/{fp16_wmma_*,gemm_*}.isa` (generated; not committed)
