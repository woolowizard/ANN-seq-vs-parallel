#include "fvecs_loader.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

std::vector<std::vector<float>> loadFvecs(
    const std::filesystem::path& path,
    std::int32_t expectedDim
) {
    if (expectedDim <= 0) {
        throw std::invalid_argument("expectedDim must be positive");
    }

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open dataset: " + path.string());
    }

    std::vector<std::vector<float>> vectors;

    while (true) {
        std::int32_t vectorDim{};

        if (!file.read(reinterpret_cast<char*>(&vectorDim), sizeof(vectorDim))) {
            if (file.eof() && file.gcount() == 0) {
                break;
            }

            throw std::runtime_error("Truncated vector dimension in: " + path.string());
        }

        if (vectorDim != expectedDim) {
            throw std::runtime_error("Unexpected vector dimension in: " + path.string());
        }

        std::vector<float> vector(static_cast<std::size_t>(vectorDim));

        if (!file.read(
                reinterpret_cast<char*>(vector.data()),
                static_cast<std::streamsize>(vector.size() * sizeof(float))
            )) {
            throw std::runtime_error("Truncated vector data in: " + path.string());
        }

        vectors.push_back(std::move(vector));
    }

    return vectors;
}
