#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

struct FlatMatrix {
    // Le componenti SIFT sono interi in [0, 255]: basta un byte per componente.
    std::vector<std::uint8_t> data;
    std::size_t rows = 0;
    std::size_t dim = 0;

    // No copy: index must be < rows, and the matrix must outlive the pointer.
    const std::uint8_t* row(std::size_t index) const {
        return data.data() + index * dim;
    }
};

// Legge SIFT in formato fvecs e converte gli interi [0, 255] in byte senza perdita.
FlatMatrix loadFvecs(const std::filesystem::path& path, std::int32_t expectedDim);

// Legge rowCount righe di ground truth: ogni riga contiene 100 indici di vicini esatti.
std::vector<std::vector<std::int32_t>> loadGroundTruth(
    const std::filesystem::path& path, size_t rowCount
);
