#pragma once

#include <cstddef>
#include <vector>

struct FlatMatrix {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t dim = 0;

    // No copy: index must be < rows, and the matrix must outlive the pointer.
    const float* row(std::size_t index) const {
        return data.data() + index * dim;
    }
};
