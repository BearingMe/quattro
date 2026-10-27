#pragma once

#include "core/tensor.h"
#include "matmul.h"

namespace gpu {

bool ValidateTensorOp(const core::TensorOpDesc& desc, const std::vector<uint8_t>& storage,
                      std::string& error);

// Isolated M10 operator validation runner, not a graph executor. Reuses the
// existing D3D12 upload/dispatch/readback harness; both snapshots are exposed.
bool RunTensorOp(DeviceContext& ctx, const core::TensorOpDesc& desc,
                  const std::vector<uint8_t>& storage,
                  std::vector<uint8_t>& warm, std::vector<uint8_t>& repeated,
                  std::string& error);

} // namespace gpu
