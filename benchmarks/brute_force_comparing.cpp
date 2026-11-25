#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "ann.hpp"
#include "dataset.hpp"

int main() {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const int DIM = 128;
    const size_t K = 20;
    const int THREADS = 1; // Cambia qui il numero di thread.
    const std::string which_distance = "squared"; // squared, manhattan oppure cosine.
    const auto directory = std::filesystem::path(PROJECT_SOURCE_DIR) / "data" / "sift";
    const auto vectors = loadFvecs(directory / "sift_base.fvecs", DIM);
    const auto queries = loadFvecs(directory / "sift_query.fvecs", DIM);
    const auto vectorInverseNorms = which_distance == "cosine"
        ? precomputeInverseNorms(vectors) : std::vector<double>{};
    const auto queryInverseNorms = which_distance == "cosine"
        ? precomputeInverseNorms(queries) : std::vector<double>{};
    const size_t queryCount = queries.rows; // Tutte le query del file.
    std::vector<std::vector<size_t>> results(queryCount);

    // Parallelizziamo le query: ogni thread scorre tutta la base per le proprie query.
    const auto start = Clock::now();
    #pragma omp parallel num_threads(THREADS)
    {
        std::vector<std::pair<float, size_t>> nearest;
        nearest.reserve(K);
        #pragma omp for schedule(static)
        for (size_t q = 0; q < queryCount; ++q) {
            results[q] = bruteForceQuery(
                vectors, queries.row(q), K, nearest, which_distance,
                which_distance == "cosine" ? queryInverseNorms[q] : 0.0,
                vectorInverseNorms
            );
        }
    }
    const double searchMs = Milliseconds(Clock::now() - start).count();
    std::cout << std::fixed << std::setprecision(3)
              << "Base vectors: " << vectors.rows << '\n'
              << "Queries processed: " << queryCount << " / " << queries.rows << '\n'
              << "K: " << K << '\n'
              << "Threads: " << THREADS << '\n'
              << "Parallel time: " << searchMs << " ms\n"
              << "Parallel throughput: " << queryCount * 1000.0 / searchMs << " queries/s\n";

    // Nel brute force ogni distanza e' completa: nessun accesso all'indice o early exit.
    SearchStats stats;
    stats.distanceComputations = static_cast<std::uint64_t>(queryCount) * vectors.rows;
    stats.candidateVisits = stats.distanceComputations;
    stats.distanceComponents = stats.distanceComputations * vectors.dim;
    printSearchStats(std::cout, stats, queryCount, vectors.rows, vectors.dim);
    return 0;
}
