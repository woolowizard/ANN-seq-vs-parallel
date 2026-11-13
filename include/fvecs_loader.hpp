#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

std::vector<std::vector<float>> loadFvecs(
    const std::filesystem::path& path,
    std::int32_t expectedDim
);
