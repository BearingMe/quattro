#include "fsr4_weights.h"
#include "weights.generated.h"

#include <cmath>
#include <fstream>

namespace inference::fsr4 {

std::span<const WeightTensor> WeightTable() { return generated::kWeights; }
std::span<const PassScalars> ScalarTable() { return generated::kScalars; }
size_t WeightPayloadBytes() { return generated::kPayloadBytes; }
const char* WeightPayloadSha256() { return generated::kPayloadSha256; }

bool ValidateWeights(std::span<const uint8_t> bytes, std::string& error) {
    error.clear();
    if (bytes.size() != generated::kPayloadBytes) {
        error = "FSR4 weight payload size mismatch";
        return false;
    }
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t value : bytes) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    if (hash != generated::kPayloadFnv1a) {
        error = "FSR4 weight payload integrity mismatch";
        return false;
    }
    for (const WeightTensor& tensor : WeightTable()) {
        if (tensor.pass >= 14 || (tensor.rank != 1 && tensor.rank != 3 && tensor.rank != 4) ||
            (tensor.itemBytes != 1 && tensor.itemBytes != 2) ||
            !std::isfinite(tensor.scale) || tensor.scale <= 0 ||
            tensor.offset > bytes.size() || tensor.bytes > bytes.size() - tensor.offset) {
            error = "invalid FSR4 weight descriptor: " + std::string(tensor.name);
            return false;
        }
        uint64_t extent = tensor.itemBytes;
        for (uint32_t axis = 0; axis < tensor.rank; ++axis) {
            if (tensor.logical[axis] == 0 || tensor.physical[axis] < tensor.logical[axis] ||
                tensor.strides[axis] == 0) {
                error = "invalid FSR4 weight shape: " + std::string(tensor.name);
                return false;
            }
            extent += uint64_t(tensor.physical[axis] - 1) * tensor.strides[axis];
        }
        if (extent != tensor.bytes) {
            error = "invalid FSR4 weight extent: " + std::string(tensor.name);
            return false;
        }
    }
    for (const PassScalars& scalars : ScalarTable()) {
        if (scalars.count > scalars.values.size()) {
            error = "invalid FSR4 scalar count";
            return false;
        }
        for (uint32_t i = 0; i < scalars.count; ++i) {
            if (!std::isfinite(scalars.values[i]) || scalars.values[i] <= 0) {
                error = "invalid FSR4 operator scalar";
                return false;
            }
        }
    }
    return true;
}

bool LoadWeights(const std::filesystem::path& path, std::vector<uint8_t>& bytes,
                 std::string& error) {
    bytes.clear();
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot open FSR4 weight payload";
        return false;
    }
    if (file.tellg() != std::streamoff(generated::kPayloadBytes)) {
        error = "FSR4 weight payload file size mismatch";
        return false;
    }
    std::vector<uint8_t> data(generated::kPayloadBytes);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()))) {
        error = "cannot read complete FSR4 weight payload";
        return false;
    }
    if (!ValidateWeights(data, error))
        return false;
    bytes = std::move(data);
    return true;
}

} // namespace inference::fsr4
