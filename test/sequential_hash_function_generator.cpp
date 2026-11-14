#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ann.hpp"
#include "fvecs_loader.hpp"

int main() {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const auto totalStart = Clock::now();

    // Configuration: SIFT1M baseline, recall@5 = 87.738%, seed 42.
    const std::uint32_t SEED = 42;
    std::mt19937 gen{SEED};
    const float w = 512.0f;
    const int dim = 128;
    const int M = 8;
    const int L = 60;
    const size_t K = 5;
    const std::filesystem::path datasetDirectory =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "data" / "sift";

    // Dataset loading.
    const auto loadingStart = Clock::now();
    const auto vectors = loadFvecs(datasetDirectory / "sift_base.fvecs", dim);
    const auto queries = loadFvecs(datasetDirectory / "sift_query.fvecs", dim);
    const double loadingMs = Milliseconds(Clock::now() - loadingStart).count();
    if (queries.rows == 0) {
        throw std::runtime_error("Query dataset is empty");
    }
    std::cout << "Base vectors: " << vectors.rows << '\n';
    std::cout << "Query vectors: " << queries.rows << '\n';

    // Index construction.
    const auto indexingStart = Clock::now();
    const auto hashTables = generateHashTable(L, M, dim, w, gen);
    const auto buckets = buildBuckets(vectors, hashTables, w);
    const double indexingMs = Milliseconds(Clock::now() - indexingStart).count();
    std::cout << "Index construction (ms): " << indexingMs << std::endl;

    // Sequential search: one result row per query, as in the parallel version.
    const size_t queryCount = queries.rows;
    std::vector<std::vector<size_t>> neighbours_structure(queryCount);
    const auto searchStart = Clock::now();
        for (size_t q = 0; q < queryCount; ++q) {
        neighbours_structure[q] = searchQuery(
            vectors, queries.row(q), queries.dim, hashTables, buckets, w, K
        );
    }
    const double searchMs = Milliseconds(Clock::now() - searchStart).count();

    // Timing summary, outside the search measurement.
    const double totalMs = Milliseconds(Clock::now() - totalStart).count();
    std::cout << std::fixed << std::setprecision(3)
              << "\nSeed: " << SEED << '\n'
              << "Queries processed: " << queryCount << '\n'
              << "Loading (ms): " << loadingMs << '\n'
              << "Index construction (ms): " << indexingMs << '\n'
              << "Search, all queries (ms): " << searchMs << '\n'
              << "Average per query (ms): " << searchMs / queryCount << '\n'
              << "Throughput (queries/s): " << queryCount * 1000.0 / searchMs << '\n'
              << "Elapsed until results (ms): " << totalMs << '\n';

    return 0;
}
