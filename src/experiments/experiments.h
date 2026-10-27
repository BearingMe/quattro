// experiments — independently runnable experiment implementations and
// benchmarks. Nothing may depend on this module (docs/architecture.md §3/§4).

#pragma once

namespace experiments {

// M05 — D3D12 compute harness validation. Returns the CLI exit code
// (0 pass, 1 experiment failure).
int RunComputeHarness();

} // namespace experiments
