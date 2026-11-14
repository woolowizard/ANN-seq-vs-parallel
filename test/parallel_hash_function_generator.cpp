#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ann.hpp"
#include "fvecs_loader.hpp"
#include "ground_truth_loader.hpp"

int main() {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    // Configuration: SIFT1M baseline, recall@5 = 87.738%, seed 42.
    const std::uint32_t SEED = 42;
    std::mt19937 gen{SEED};
    const float w = 512.0f;
    const int dim = 128;
    const int M = 8;
    const int L = 60;
    const size_t K = 5;
    const std::array<int, 1> thread_number = {10};
    const bool CHECKS = false;
    const std::filesystem::path datasetDirectory =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "data" / "sift";

    // Dataset and ground truth loading, outside search timing.
    const auto vectors = loadFvecs(datasetDirectory / "sift_base.fvecs", dim);
    const auto queries = loadFvecs(datasetDirectory / "sift_query.fvecs", dim);
    if (queries.rows == 0) {
        throw std::runtime_error("Query dataset is empty");
    }
    std::cout << "Base vectors: " << vectors.rows << '\n';
    std::cout << "Query vectors: " << queries.rows << '\n';
    const auto groundTruth = loadGroundTruth(
        datasetDirectory / "sift_groundtruth.ivecs",
        std::min<size_t>(10, queries.rows)
    );

    // Index construction, shared by all thread configurations.
    const auto indexingStart = Clock::now();
    const auto hashTables = generateHashTable(L, M, dim, w, gen);
    const auto buckets = buildBuckets(vectors, hashTables, w);
    const double indexingMs = Milliseconds(Clock::now() - indexingStart).count();
    std::cout << "Index construction (ms): " << indexingMs << std::endl;

    // Parallel search: each iteration writes only its own result row.
    for (int t : thread_number) {
        size_t q = 0;
        const size_t queryCount = queries.rows;
        std::vector<std::vector<size_t>> neighbours_structure(queryCount);
        const auto searchStart = Clock::now();

        #pragma omp parallel shared(vectors, queries, queryCount, hashTables, buckets, neighbours_structure) private(q) num_threads(t)
        {
            #pragma omp for schedule(dynamic) nowait
            for (q = 0; q < queryCount; ++q) {
                neighbours_structure[q] = searchQuery(
                    vectors, queries.row(q), queries.dim, hashTables, buckets, w, K
                );
            }
        }
        const double searchMs = Milliseconds(Clock::now() - searchStart).count();
        std::cout << "Threads: " << t << '\n';
        std::cout << "Search time (ms): " << searchMs << '\n';

        // Optional inspection of the first 10 queries, outside search timing.
        if (CHECKS) {
            for (size_t q = 0; q < std::min<size_t>(10, neighbours_structure.size()); ++q) {
                std::cout << "Query " << q << ": ";
                for (size_t index : neighbours_structure[q]) {
                    std::cout << index << ' ';
                }
                std::cout << '\n';
            }

            for (size_t q = 0; q < std::min<size_t>(10, groundTruth.size()); ++q) {
                std::cout << "Ground truth query " << q << ": ";
                for (size_t j = 0; j < std::min<size_t>(5, groundTruth[q].size()); ++j) {
                    std::cout << groundTruth[q][j] << ' ';
                }
                std::cout << '\n';
            }
        }
    }

    return 0;
}
