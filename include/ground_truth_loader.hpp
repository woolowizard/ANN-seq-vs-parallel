#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

// Read rowCount SIFT ground truth rows, each containing 100 neighbor indices.
std::vector<std::vector<std::int32_t>> loadGroundTruth(
    const std::filesystem::path& path, size_t rowCount
);
