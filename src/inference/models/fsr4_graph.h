#pragma once

#include "gpu/amdext.h"
#include "gpu/graph.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

// Concrete M12 dispatch assembly, not a second execution engine. Initial data
// pointers must remain alive until RunBufferPasses returns; PSOs live here.
// The caller creates the signature (nativeWmma=true when including a CNB).
struct DispatchPlan {
    gpu::ComPtr<ID3D12RootSignature> signature;
    std::vector<gpu::BufferInit> buffers;
    std::vector<gpu::ComPtr<ID3D12PipelineState>> psos;
    std::vector<gpu::BufferPass> passes;
    std::vector<std::string> convolutionIsa;
    std::vector<std::string> convolutionStats;
};

// Append only: no uploads, submissions, synchronization or CPU readback.
// Inputs are buffer indices; the returned index names the new output tensor.
// Discard the plan on failure (assembly may already have appended workspace).
// AppendConvNextBlock requires a native-WMMA signature and validated weights.
bool AppendInputConvolution(gpu::DeviceContext& context, DispatchPlan& plan,
                            size_t input, size_t weights, size_t& output,
                            std::string& error);
bool AppendConvNextBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                         DispatchPlan& plan, uint32_t pass, size_t input, size_t weights,
                         size_t& output, std::string& error);
bool AppendDownscale(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                      DispatchPlan& plan, uint32_t pass, size_t input, size_t weights,
                      size_t& output, std::string& error);
bool AppendFasterNetBlock(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                          DispatchPlan& plan, uint32_t pass, size_t input, size_t skip,
                          size_t weights, size_t& output, std::string& error);
bool AppendCnbCt2d(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                   DispatchPlan& plan, size_t input, size_t weights, size_t& output,
                   std::string& error);
bool AppendModelGraph(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                       DispatchPlan& plan, size_t input, size_t weights, uint32_t lastPass,
                       size_t& output, std::string& error,
                       std::vector<size_t>* boundaries = nullptr);

// M15 equivalent-workload comparison: replace ONLY the eleven WMMA products
// with native F16/F32 products + exact I32 conversion. DP4A and epilogues stay
// identical. This is not a fully FP16 network or a different trained model.
bool UseFp16MatrixComparison(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                             DispatchPlan& plan, std::vector<size_t>& boundaries,
                             std::vector<std::string>& stats, std::string& error);

// Fixed pinned graph prefix, inclusive lastPass (0..13). Separate live tensors;
// optional logical snapshots at every model boundary, not every micro-dispatch.
bool RunModelGraph(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                    std::span<const uint8_t> input, std::span<const uint8_t> weights,
                    uint32_t lastPass, std::vector<uint8_t>& output,
                    std::vector<std::vector<uint8_t>>* snapshots,
                    std::vector<std::string>& convolutionIsa, std::string& error);

// Incremental integration checkpoint: passes 0 -> 1, ten GPU dispatches in
// one submission. Optional pass-0 snapshot is diagnostic only and never fed
// back to the GPU. Normal execution reads back only the pass-1 output.
bool RunInputChain(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                    std::span<const uint8_t> input, std::span<const uint8_t> weights,
                    std::vector<uint8_t>& output, std::vector<uint8_t>* pass0Snapshot,
                    std::vector<std::string>& convolutionIsa, std::string& error);

} // namespace inference::fsr4
