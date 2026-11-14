#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <array>
#include <utility>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include "fvecs_loader.hpp"

using Signature = std::vector<std::int64_t>;

// Generate a random vector from a Gaussian distribution a ~ N(mean, std)
std::vector<float> normalVector(
    int dim,
    float mean,
    float stddev,
    std::mt19937& generator
) {
    std::vector<float> vec(dim);

    std::normal_distribution<float> distribution(mean, stddev);

    for (int i = 0; i < dim; i++) {
        vec[i] = distribution(generator);
    }

    return vec;
}

struct SignatureHash {
    size_t operator()(
        const Signature& signature
    ) const noexcept {
        size_t seed = 0;

        for (std::int64_t value : signature) {
            const size_t valueHash =
                std::hash<std::int64_t>{}(value);

            seed ^= valueHash + 0x9e3779b9
                    + (seed << 6)
                    + (seed >> 2);
        }

        return seed;
    }
};

using BucketTable = std::unordered_map<Signature, std::vector<size_t>, SignatureHash>;

// Generate the random offset b ~ U(0, w)
float biasParameter(float w, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(0.0f, w);
    float bias = distribution(generator);

    return bias;
}

// Compute h(x) = floor((a^T x + b) / w)
std::int64_t computeHash(
    const float* x,
    size_t dim,
    const std::vector<float>& a,
    float b,
    float w
) {
    if (dim != a.size() || w <= 0.0f) {
        throw std::invalid_argument("Invalid dimensions or w");
    }

    double projection = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        projection += static_cast<double>(a[i]) * x[i];
    }

    return static_cast<std::int64_t>(
        std::floor((projection + b) / w)
    );
}

struct HashFunction {
    std::vector<float> a;
    float b;

    HashFunction(int dim, float w, std::mt19937& generator)
        : a(normalVector(dim, 0.0f, 1.0f, generator)),
          b(biasParameter(w, generator))
    {}
};

// Generate M fixed hash functions for each of the L tables.
std::vector<std::vector<HashFunction>> generateHashTable(
    int L,
    int M,
    int dim,
    float w,
    std::mt19937& generator
) {
    std::vector<std::vector<HashFunction>> hashTables(L);

    for (auto& table : hashTables) {
        table.reserve(M);
        for (int m = 0; m < M; ++m) {
            table.emplace_back(dim, w, generator);
        }
    }

    return hashTables;
}

Signature computeSignature(
    const float* vector,
    size_t dim,
    const std::vector<HashFunction>& functions,
    float w
) {
    Signature signature;
    signature.reserve(functions.size());

    for (const auto& function : functions) {
        signature.push_back(computeHash(vector, dim, function.a, function.b, w));
    }

    return signature;
}

float calculate_squared_distance(const float* v, const float* q, size_t dim) {
    double squaredDistance = 0.0;

    for (size_t i = 0; i < dim; ++i) {
        const double difference = static_cast<double>(v[i]) - q[i];
        squaredDistance += difference * difference;
    }

    return static_cast<float>(squaredDistance);
}

int main() {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const auto totalStart = Clock::now();

    const std::uint32_t SEED = 42;
    std::mt19937 gen{SEED};

    // SIFT1M baseline: recall@5 = 87.738% over all 10,000 queries, seed 42.
    float w = 512.0f;
    int dim = 128;
    int M = 8;
    int L = 60;
    const size_t K = 5;

    const std::filesystem::path datasetDirectory =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "data" / "sift";

    const auto loadingStart = Clock::now();
    const auto vectors = loadFvecs(datasetDirectory / "sift_base.fvecs", dim);
    const auto queries = loadFvecs(datasetDirectory / "sift_query.fvecs", dim);
    const double loadingMs = Milliseconds(Clock::now() - loadingStart).count();

    if (queries.rows == 0) {
        throw std::runtime_error("Query dataset is empty");
    }

    std::cout << "Base vectors: " << vectors.rows << '\n';
    std::cout << "Query vectors: " << queries.rows << '\n';

    // Load the first 10 ground truth rows for comparison outside search timing.
    std::ifstream groundTruthFile(
        datasetDirectory / "sift_groundtruth.ivecs",
        std::ios::binary
    );
    if (!groundTruthFile) {
        throw std::runtime_error("Cannot open ground truth");
    }

    std::vector<std::vector<std::int32_t>> groundTruth(
        std::min<size_t>(10, queries.rows)
    );
    for (auto& neighbors : groundTruth) {
        std::int32_t neighborCount;
        if (!groundTruthFile.read(
                reinterpret_cast<char*>(&neighborCount),
                sizeof(neighborCount))) {
            throw std::runtime_error("Cannot read ground truth header");
        }
        if (neighborCount != 100) {
            throw std::runtime_error("Unexpected ground truth dimensions");
        }

        neighbors.resize(neighborCount);
        if (!groundTruthFile.read(
                reinterpret_cast<char*>(neighbors.data()),
                neighborCount * sizeof(std::int32_t))) {
            throw std::runtime_error("Cannot read ground truth neighbors");
        }
    }

    // Build the LSH index.
    const auto indexingStart = Clock::now();
    const auto hashTables = generateHashTable(L, M, dim, w, gen);
    std::vector<BucketTable> buckets(L);


    for (size_t i = 0; i < vectors.rows; ++i) {
        const float* vector = vectors.row(i);
        for (int l = 0; l < L; l++) {
            const auto signature = computeSignature(vector, vectors.dim, hashTables[l], w);
            buckets[l][signature].push_back(i);
        }
    }
    const double indexingMs = Milliseconds(Clock::now() - indexingStart).count();

    std::cout << "Index construction (ms): "
          << indexingMs << std::endl;

    /***************************/
    // BEGIN Parallel Version //
    /***************************/

    // std::array<int, 10> thread_number = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::array<int, 1> thread_number = {10};

    for (int t: thread_number)  {

        // Parallel version declarations
        size_t q = 0;
        const size_t queryCount = queries.rows;

        std::vector<std::vector<size_t>> neighbours_structure(queryCount);
        // In position neighbours_structure[q] --> pointer to vector of neighbours index of q
        // In position neighbours_structure[q] --> pointer to j-th neighbours index of q

        const auto searchStart = Clock::now();

        // BEGIN Parallel Region
        #pragma omp parallel shared(vectors, queries, queryCount, hashTables, buckets, neighbours_structure) private(q) num_threads(t)
        {
            #pragma omp for schedule(dynamic) nowait
                for (q = 0; q < queryCount; ++q) {
                    const float* query = queries.row(q);
                    std::unordered_set<size_t> candidateIndexes;

                    for (int l = 0; l < L; ++l) {
                        const auto signature = computeSignature(query, queries.dim, hashTables[l], w);
                        const auto it = buckets[l].find(signature);

                        if (it != buckets[l].end()) {
                            candidateIndexes.insert(
                                it->second.begin(),
                                it->second.end()
                            );
                        }
                    }

                    // Score each distinct candidate once, then select the closest K.
                    std::vector<std::pair<float, size_t>> scoredCandidates; // Pair vector of (distance_from_q, index_baseline_vector)
                    scoredCandidates.reserve(candidateIndexes.size());

                    for (size_t index : candidateIndexes) {
                        const float distance = calculate_squared_distance(vectors.row(index), query, vectors.dim);

                        scoredCandidates.emplace_back(distance, index); // Append pair (distance_from_q, index_baseline_vector)
                    }

                    const size_t resultCount = std::min(K, scoredCandidates.size());

                    std::partial_sort(
                        scoredCandidates.begin(),
                        scoredCandidates.begin() + resultCount,
                        scoredCandidates.end(),
                        [](const auto& left, const auto& right) {
                            return left.first < right.first;
                        }
                    );


                    neighbours_structure[q].reserve(resultCount);
                    for (size_t j = 0; j < resultCount; ++j) {
                        neighbours_structure[q].push_back(scoredCandidates[j].second);
                    }

                }
            } // END Parallel Region
            const double searchMs = Milliseconds(Clock::now() - searchStart).count();

            std::cout << "Threads: " << t <<  '\n';
            std::cout << "Search time (ms): " << searchMs << '\n';


            /******************************************/
            /************ INTEGRITY CHECKS ************/
            /******************************************/

            bool CHECKS = false;
            if (CHECKS == true) {
                // Print first K=5 ANN neighbours for first 10 queries
                for (size_t q = 0; q < std::min<size_t>(10, neighbours_structure.size()); ++q) {
                    std::cout << "Query " << q << ": ";

                    for (size_t index : neighbours_structure[q]) {
                        std::cout << index << ' ';
                    }

                    std::cout << '\n';
                }

                // Print first K=5 GT neighbours for first 10 queries
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
