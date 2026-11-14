#include "fvecs_loader.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

FlatMatrix loadFvecs(
    const std::filesystem::path& path,
    std::int32_t expectedDim
) {
    if (expectedDim <= 0) {
        throw std::invalid_argument("expectedDim must be positive");
    }

    static_assert(sizeof(float) == 4, "fvecs requires 32-bit floats");

    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Cannot open dataset: " + path.string());
    }

    const auto endPosition = file.tellg();
    if (endPosition < 0) {
        throw std::runtime_error("Cannot determine dataset size: " + path.string());
    }

    const auto fileBytes = static_cast<std::uintmax_t>(endPosition);
    const auto rowBytes = static_cast<std::uintmax_t>(expectedDim) * sizeof(float);
    const auto recordBytes = sizeof(std::int32_t) + rowBytes;

    if (fileBytes % recordBytes != 0) {
        throw std::runtime_error("Truncated fvecs record in: " + path.string());
    }
    const auto rowCount = fileBytes / recordBytes;
    FlatMatrix vectors;
    vectors.dim = static_cast<std::size_t>(expectedDim);
    if (rowCount > vectors.data.max_size() / vectors.dim ||
        rowBytes > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("Dataset is too large: " + path.string());
    }

    // Allocate all components once and read each record directly into its row.
    vectors.rows = static_cast<std::size_t>(rowCount);
    vectors.data.resize(vectors.rows * vectors.dim);
    file.seekg(0, std::ios::beg);

    for (std::size_t i = 0; i < vectors.rows; ++i) {
        std::int32_t vectorDim{};

        if (!file.read(reinterpret_cast<char*>(&vectorDim), sizeof(vectorDim))) {
            throw std::runtime_error("Cannot read vector dimension in: " + path.string());
        }

        if (vectorDim != expectedDim) {
            throw std::runtime_error("Unexpected vector dimension in: " + path.string());
        }

        if (!file.read(
                reinterpret_cast<char*>(vectors.data.data() + i * vectors.dim),
                static_cast<std::streamsize>(rowBytes)
            )) {
            throw std::runtime_error("Truncated vector data in: " + path.string());
        }
    }

    return vectors;
}
