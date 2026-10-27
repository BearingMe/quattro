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

} // namespace experiments
