// Fixed FSR 4.0.2 v07 INT8 Quality weight contract. No graph/runtime framework.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace inference::fsr4 {

struct WeightTensor {
    uint32_t pass;
    const char* name;
    uint32_t rank;
    uint32_t itemBytes; // 1 = signed INT8; 2 = IEEE FP16
    uint32_t offset;
    uint32_t bytes; // physical bounding extent, including padding
    std::array<uint32_t, 4> logical;
    std::array<uint32_t, 4> physical;
    std::array<uint32_t, 4> strides;
    float scale;
};

struct PassScalars {
    uint32_t count;
    std::array<float, 6> values; // numeric prefix of the pinned operator call
};

std::span<const WeightTensor> WeightTable();
std::span<const PassScalars> ScalarTable();
size_t WeightPayloadBytes();
const char* WeightPayloadSha256(); // offline identity, not computed by the loader

// Full-payload FNV-1a catches accidental corruption, not adversarial tampering.
// Offline packing additionally checks the pinned source SHA-256 identities.
bool ValidateWeights(std::span<const uint8_t> bytes, std::string& error);
bool LoadWeights(const std::filesystem::path& path, std::vector<uint8_t>& bytes,
                 std::string& error);

} // namespace inference::fsr4
