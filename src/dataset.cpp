#include "dataset.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

FlatMatrix loadFvecs(const std::filesystem::path& path, std::int32_t expectedDim) {
    if (expectedDim <= 0) {
        throw std::invalid_argument("expectedDim must be positive");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open dataset: " + path.string());
    }

    // Ogni record su disco contiene: dimensione int32, poi dim componenti float.
    const size_t rowBytes = static_cast<size_t>(expectedDim) * sizeof(float);
    const size_t recordBytes = sizeof(std::int32_t) + rowBytes;
    const auto fileBytes = std::filesystem::file_size(path);
    if (fileBytes % recordBytes != 0) {
        throw std::runtime_error("Truncated fvecs record: " + path.string());
    }

    FlatMatrix vectors;
    vectors.dim = static_cast<size_t>(expectedDim);
    vectors.rows = fileBytes / recordBytes;
    vectors.data.resize(vectors.rows * vectors.dim);
    std::vector<float> row(vectors.dim); // Una sola riga temporanea, riutilizzata.

    for (size_t i = 0; i < vectors.rows; ++i) {
        std::int32_t dim = 0;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        file.read(reinterpret_cast<char*>(row.data()), rowBytes);
        if (!file || dim != expectedDim) {
            throw std::runtime_error("Invalid fvecs record: " + path.string());
        }

        for (size_t d = 0; d < vectors.dim; ++d) {
            const float value = row[d];
            // Conversione esatta, senza quantizzazione. Il controllo esclude anche NaN/inf.
            if (!(value >= 0.0f && value <= 255.0f) || std::trunc(value) != value) {
                throw std::runtime_error("SIFT component must be an integer in [0, 255]");
            }
            vectors.data[i * vectors.dim + d] = static_cast<std::uint8_t>(value);
        }
    }
    return vectors;
}

std::vector<std::vector<std::int32_t>> loadGroundTruth(
    const std::filesystem::path& path, size_t rowCount
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open ground truth");
    }

    std::vector<std::vector<std::int32_t>> groundTruth(rowCount);
    for (auto& neighbors : groundTruth) {
        std::int32_t neighborCount;
        if (!file.read(reinterpret_cast<char*>(&neighborCount), sizeof(neighborCount))) {
            throw std::runtime_error("Cannot read ground truth header");
        }
        if (neighborCount != 100) {
            throw std::runtime_error("Unexpected ground truth dimensions");
        }

        neighbors.resize(neighborCount);
        if (!file.read(
                reinterpret_cast<char*>(neighbors.data()),
                neighborCount * sizeof(std::int32_t))) {
            throw std::runtime_error("Cannot read ground truth neighbors");
        }
    }

    return groundTruth;
}
