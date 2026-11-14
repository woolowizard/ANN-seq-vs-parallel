#pragma once

#include <cstdint>
#include <filesystem>

#include "flat_matrix.hpp"

FlatMatrix loadFvecs(
    const std::filesystem::path& path,
    std::int32_t expectedDim
);
