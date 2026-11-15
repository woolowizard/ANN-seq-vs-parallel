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
    const std::array<int, 9> thread_number = {2,3,4,5,6,7,8,9,10};
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
    queries.rows
    );

    // Index construction, shared by all thread configurations.
    const auto indexingStart = Clock::now();
    const auto hashTables = generateHashTable(L, M, dim, w, gen);
    const auto buckets = buildBuckets(vectors, hashTables, w);
    const double indexingMs = Milliseconds(Clock::now() - indexingStart).count();
    std::cout << "Index construction (ms): " << indexingMs << std::endl;

    /*****************************************************/
    /****************** START SEARCHING ******************/
    /*****************************************************/

    // Sequential search
    const size_t queryCount = queries.rows;
    std::vector<std::vector<size_t>> sequentialResults(queryCount);
    const auto SSearchStart = Clock::now();
    for (size_t q = 0; q < queryCount; ++q) {
        sequentialResults[q] = searchQuery(
            vectors, queries.row(q), queries.dim, hashTables, buckets, w, K
        );
    }
    const double SSearchMs = Milliseconds(Clock::now() - SSearchStart).count();
    std::cout << "Sequential search time (ms): " << SSearchMs << std::endl;


    // Parallel search
    for (int t : thread_number) {
        size_t q = 0;
        std::vector<std::vector<size_t>> parallelResults(queryCount);
        const auto PSearchStart = Clock::now();

        #pragma omp parallel shared(vectors, queries, queryCount, hashTables, buckets, parallelResults) private(q) num_threads(t)
        {
            #pragma omp for schedule(dynamic) nowait
            for (q = 0; q < queryCount; ++q) {
                parallelResults[q] = searchQuery(
                    vectors, queries.row(q), queries.dim, hashTables, buckets, w, K
                );
            }
        }
        const double PSearchMs = Milliseconds(Clock::now() - PSearchStart).count();
        std::cout << "Threads: " << t << '\n';
        std::cout << "Parallel search time (ms): " << PSearchMs << '\n';

        if (bool identical = (sequentialResults == parallelResults)) {
            std::cout << "Parallel results are identical compared to sequential results with thread_number: " << t << std::endl;
        } else {std::cout << "Parallel results are not identical compared to sequential results with thread_number: " << t << std::endl;}
    }

    double recall = calculateRecall(sequentialResults, groundTruth, K);
    std::cout << "Global recall: " << recall << std::endl;

    return 0;
}
