#include "dataset.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Action>
void requireThrows(Action action) {
    try {
        action();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("Expected exception was not thrown");
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ann-fvecs-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));

    TemporaryDirectory() {
        require(std::filesystem::create_directory(path), "Cannot create test directory");
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

template <typename Value>
void writeValue(std::ofstream& file, Value value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void runTests(const std::filesystem::path& directory) {
    const auto valid = directory / "valid.fvecs";
    {
        std::ofstream file(valid, std::ios::binary);
        writeValue<std::int32_t>(file, 2);
        writeValue<float>(file, 0.0f);
        writeValue<float>(file, 255.0f);
        writeValue<std::int32_t>(file, 2);
        writeValue<float>(file, 7.0f);
        writeValue<float>(file, 8.0f);
    }
    const auto matrix = loadFvecs(valid, 2);
    require(matrix.rows == 2 && matrix.dim == 2, "Wrong matrix shape");
    require(matrix.row(0)[0] == 0 && matrix.row(0)[1] == 255 &&
            matrix.row(1)[0] == 7 && matrix.row(1)[1] == 8,
            "Row boundaries or components were not preserved");
    require(matrix.row(1) == matrix.data.data() + 2,
            "Rows must be views into contiguous storage");
    requireThrows<std::invalid_argument>([&] { loadFvecs(valid, 0); });
    requireThrows<std::runtime_error>([&] { loadFvecs(valid, 3); });
    requireThrows<std::runtime_error>([&] {
        loadFvecs(directory / "missing.fvecs", 2);
    });

    // File size is valid for two rows, but the second header has a wrong dimension.
    const auto wrongDimension = directory / "wrong-dimension.fvecs";
    std::filesystem::copy_file(valid, wrongDimension);
    {
        std::fstream file(wrongDimension, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(sizeof(std::int32_t) + 2 * sizeof(float));
        const std::int32_t dim = 3;
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    requireThrows<std::runtime_error>([&] { loadFvecs(wrongDimension, 2); });

    const auto truncated = directory / "truncated.fvecs";
    std::filesystem::copy_file(valid, truncated);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
    requireThrows<std::runtime_error>([&] { loadFvecs(truncated, 2); });

    // Non arrotondiamo o tronchiamo componenti che non sono rappresentabili esattamente.
    for (float value : {1.5f, -1.0f, 256.0f,
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
        const auto invalid = directory / "invalid-component.fvecs";
        {
            std::ofstream file(invalid, std::ios::binary);
            writeValue<std::int32_t>(file, 1);
            writeValue<float>(file, value);
        }
        requireThrows<std::runtime_error>([&] { loadFvecs(invalid, 1); });
    }

    const auto empty = directory / "empty.fvecs";
    { std::ofstream file(empty, std::ios::binary); }
    const auto emptyMatrix = loadFvecs(empty, 128);
    require(emptyMatrix.rows == 0 && emptyMatrix.dim == 128 && emptyMatrix.data.empty(),
            "Wrong empty matrix");
}

} // namespace

int main() {
    try {
        TemporaryDirectory directory;
        runTests(directory.path);
        std::cout << "Flat matrix and fvecs loader tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
