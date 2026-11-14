#include "ground_truth_loader.hpp"

#include <fstream>
#include <stdexcept>

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
