#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ann.hpp"
#include "dataset.hpp"

int main() {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    const std::uint32_t SEED = 42;
    std::mt19937 gen{SEED};
    const float w = 512.0f;
    const int dim = 128;
    const int M = 8;
    const int L = 60;
    const size_t K = 5;
    const int INDEX_THREADS = 10;
    const size_t QUERIES_PER_THREAD = 1000;
    const std::array<int, 10> thread_number = {1,2,3,4,5,6,7,8,9,10}; // Il primo misura T1.
    const auto datasetDirectory = std::filesystem::path(PROJECT_SOURCE_DIR) / "data" / "sift";

    // Caricamento e costruzione dell'indice restano fuori dal tempo di ricerca.
    const auto vectors = loadFvecs(datasetDirectory / "sift_base.fvecs", dim);
    const auto queries = loadFvecs(datasetDirectory / "sift_query.fvecs", dim);
    const int maxThreads = *std::max_element(thread_number.begin(), thread_number.end());
    const size_t requiredQueries = QUERIES_PER_THREAD * maxThreads;
    if (queries.rows < requiredQueries) {
        throw std::runtime_error("Not enough queries: weak scaling needs " +
                                 std::to_string(requiredQueries));
    }
    const auto groundTruth = loadGroundTruth(datasetDirectory / "sift_groundtruth.ivecs", requiredQueries);

    const auto indexingStart = Clock::now();
    const auto hashTables = generateHashTable(L, M, dim, w, gen);
    const auto buckets = buildBuckets(vectors, hashTables, w, INDEX_THREADS);
    const double indexingMs = Milliseconds(Clock::now() - indexingStart).count();
    std::cout << std::fixed << std::setprecision(3)
              << "Base vectors: " << vectors.rows << '\n'
              << "Query vectors: " << queries.rows << '\n'
              << "Queries per thread: " << QUERIES_PER_THREAD << '\n'
              << "Index construction threads: " << INDEX_THREADS << '\n'
              << "Index construction (ms), excluded from search: " << indexingMs << '\n';

    // Anche T1 usa questo ciclo, con un solo thread e 1000 query.
    // L'indice e l'ordine delle query rimangono gli stessi per ogni configurazione.
    omp_set_dynamic(0);
    double T1 = 0.0;
    std::vector<std::vector<size_t>> sequentialResults;
    for (int t : thread_number) {
        const size_t queryCount = QUERIES_PER_THREAD * t;
        std::vector<std::vector<size_t>> results(queryCount);
        SearchStats stats;
        int actualThreads = 0;
        const auto searchStart = Clock::now();

        #pragma omp parallel num_threads(t)
        {
            #pragma omp single
            { actualThreads = omp_get_num_threads(); }

            // Ogni thread alloca il proprio buffer una volta e lo riutilizza.
            SearchBuffers buffers(vectors.rows, hashTables.M, K);
            SearchStats localStats;
            #pragma omp for schedule(dynamic)
            for (size_t q = 0; q < queryCount; ++q) {
                results[q] = searchQuery(
                    vectors, queries.row(q), queries.dim, hashTables, buckets, w, K, buffers
                );
                localStats += buffers.stats;
            }
            #pragma omp critical
            { stats += localStats; }
        }
        const double searchMs = Milliseconds(Clock::now() - searchStart).count();
        if (actualThreads != t) {
            throw std::runtime_error("OpenMP did not create the requested number of threads");
        }

        // Controlli e recall fuori dal tempo misurato.
        if (t == 1) {
            T1 = searchMs;
            sequentialResults = results;
        } else {
            // Solo queste query sono presenti anche nel riferimento a un thread.
            if (!std::equal(sequentialResults.begin(), sequentialResults.end(), results.begin())) {
                std::cerr << "Results differ on the first " << QUERIES_PER_THREAD
                          << " queries with " << t << " threads\n";
                return 1;
            }
            std::cout << "First " << QUERIES_PER_THREAD << " query results match T1\n";
        }

        std::cout << "Threads: " << actualThreads << '\n'
                  << "Queries processed: " << queryCount << '\n'
                  << "T1 (ms): " << T1 << '\n'
                  << "Search time (ms): " << searchMs << '\n';
        printWeakScalingStats(std::cout, stats, queryCount, vectors.rows, vectors.dim, T1, searchMs);
        std::cout << "Recall@" << K << " (%): " << 100.0 * calculateRecall(results, groundTruth, K)
                  << '\n';
    }
    return 0;
}
