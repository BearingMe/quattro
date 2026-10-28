// experiments — independently runnable experiment implementations and
// benchmarks. Nothing may depend on this module (docs/architecture.md §3/§4).

#pragma once

namespace experiments {

// M05 — D3D12 compute harness validation. Returns the CLI exit code
// (0 pass, 1 experiment failure).
int RunComputeHarness();

// M06 — Native INT8 WMMA probe (AMD shader intrinsics extension) with DP4A
// control. Returns the CLI exit code (0 pass, 1 experiment failure).
int RunWmmaInt8();

// M07 — FP16 WMMA (F32 + F16 accumulators) and the INT8/FP16/DP4A multi-tile
// hardware comparison. Returns the CLI exit code.
int RunWmmaFp16();

// M08 — FP8 E4M3/E5M2 + symmetric INT8 quantization: exhaustive CPU
// validation vs independent oracles, quantization error metrics over
// representative distributions, and AMD's shipped QuantizeLinear operator
// executed on gfx1101 as the C09 hardware reference. Returns the CLI exit code.
int RunQuantization();

// M09 — generalized INT8 GEMM (runtime M/N, fixed-graph K, boundary tiles
// via padded packing), AMD FNB_CT2D requantization at store, 3x3p1
// convolution via im2col+GEMM, overflow guard, ISA/stats evidence and
// benchmarks. Returns the CLI exit code.
int RunInt8Kernels();

// M10 — source-justified non-matrix operations and explicit tensor contracts.
int RunTensorOps();

// M11 preliminary CPU oracle check, not the milestone's GPU inference gate.
int RunFixedNetworkReference();

// M11 hardware gate: the fixed two-layer INT8 network executed end-to-end on
// the GPU (native WMMA + requantization, GPU-resident intermediates, CPU
// comparison, repeated execution, end-to-end timing).
int RunFixedNetworkGpu();

// M12 reference oracle: executes the pinned FSR 4 v07 INT8 (1080p) model passes
// from the read-only research tree and captures the tensors our reproduction
// must match. Not the M12 gate.
int RunFsr4Oracle();

// M12 incremental reproduction validation (only when model data is available).
int RunFsr4Weights();
int RunFsr4Pass0();
int RunFsr4Cnb();
int RunFsr4Chain();
int RunFsr4Graph();
int RunFsr4Temporal();
int RunFsr4Downscale();
int RunFsr4Fnb();

} // namespace experiments
