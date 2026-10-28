#include "fsr4_graph.h"
#include "fsr4_weights.h"

namespace inference::fsr4 {

bool RunModelGraph(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                    std::span<const uint8_t> input, std::span<const uint8_t> weights,
                    uint32_t lastPass, std::vector<uint8_t>& output,
                    std::vector<std::vector<uint8_t>>* snapshots,
                    std::vector<std::string>& convolutionIsa, std::string& error) {
    output.clear();
    convolutionIsa.clear();
    if (snapshots)
        snapshots->clear();
    if (lastPass > 13) {
        error = "FSR4 graph prefix must end at pass 0..13";
        return false;
    }
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error, true))
        return false;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()}};
    size_t tensors[14]{};
    std::vector<bool> snapshotAfter;
    for (uint32_t pass = 0; pass <= lastPass; ++pass) {
        bool appended = false;
        const size_t source = pass == 0 ? 0 : tensors[pass - 1];
        if (pass == 0)
            appended = AppendInputConvolution(context, plan, source, 1, tensors[pass], error);
        else if (pass == 1 || pass == 2 || pass == 12)
            appended = AppendConvNextBlock(context, extensions, plan, pass, source, 1, tensors[pass], error);
        else if (pass == 3 || pass == 6)
            appended = AppendDownscale(context, extensions, plan, pass, source, 1, tensors[pass], error);
        else if (pass == 13)
            appended = AppendCnbCt2d(context, extensions, plan, source, 1, tensors[pass], error);
        else {
            // Decoder skips are outputs of passes 5 and 2, not the downscale
            // outputs. Separate allocations retain them without scratch aliasing.
            const size_t skip = pass == 9 ? tensors[5] : pass == 11 ? tensors[2] : SIZE_MAX;
            appended = AppendFasterNetBlock(context, extensions, plan, pass, source, skip, 1,
                                             tensors[pass], error);
        }
        if (!appended) {
            error = "assembling model pass " + std::to_string(pass) + ": " + error;
            return false;
        }
        snapshotAfter.resize(plan.passes.size(), false);
        snapshotAfter.back() = snapshots != nullptr;
        plan.passes.back().snapshotBuffer = tensors[pass];
    }
    convolutionIsa = std::move(plan.convolutionIsa);
    std::vector<std::vector<uint8_t>> captured;
    const size_t destination = tensors[lastPass];
    if (!gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes, destination,
                              snapshotAfter, captured, destination, plan.buffers[destination].bytes,
                              output, error, true))
        return false;
    if (snapshots)
        *snapshots = std::move(captured);
    return true;
}

bool RunInputChain(gpu::DeviceContext& context, gpu::AmdExtensions& extensions,
                    std::span<const uint8_t> input, std::span<const uint8_t> weights,
                    std::vector<uint8_t>& output, std::vector<uint8_t>* pass0Snapshot,
                    std::vector<std::string>& convolutionIsa, std::string& error) {
    output.clear();
    convolutionIsa.clear();
    if (pass0Snapshot)
        pass0Snapshot->clear();
    if (!ValidateWeights(weights, error))
        return false;
    DispatchPlan plan;
    if (!gpu::CreateBufferPassRootSignature(context, plan.signature, error, true))
        return false;
    plan.buffers = {{input.size(), input.data()}, {weights.size(), weights.data()}};
    size_t pass0 = 0, pass1 = 0;
    if (!AppendInputConvolution(context, plan, 0, 1, pass0, error) ||
        !AppendConvNextBlock(context, extensions, plan, 1, pass0, 1, pass1, error))
        return false;
    convolutionIsa = std::move(plan.convolutionIsa);
    std::vector<bool> snapshotAfter(plan.passes.size(), false);
    snapshotAfter[0] = pass0Snapshot != nullptr;
    std::vector<std::vector<uint8_t>> snapshots;
    if (!gpu::RunBufferPasses(context, plan.signature, plan.buffers, plan.passes, pass0,
                              snapshotAfter, snapshots, pass1, plan.buffers[pass1].bytes,
                              output, error, true))
        return false;
    if (pass0Snapshot)
        *pass0Snapshot = std::move(snapshots.front());
    return true;
}

} // namespace inference::fsr4
