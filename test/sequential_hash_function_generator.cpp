#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <filesystem>
#include <utility>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <iomanip>

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

    // Search the complete query set. Keep output outside the timed loop.
    const size_t queryCount = queries.rows;
    size_t totalCandidates = 0;
    size_t totalResults = 0;
    std::uint64_t resultIndexSum = 0;
    double resultDistanceSum = 0.0;

    const auto searchStart = Clock::now();
    for (size_t q = 0; q < queryCount; ++q) {
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
        std::vector<std::pair<float, size_t>> scoredCandidates;
        scoredCandidates.reserve(candidateIndexes.size());

        for (size_t index : candidateIndexes) {
            const float distance = calculate_squared_distance(vectors.row(index), query, vectors.dim);

            scoredCandidates.emplace_back(distance, index);
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
        totalCandidates += candidateIndexes.size();
        totalResults += resultCount;
        // Consume every top-K result without printing or retaining per-query buffers.
        for (size_t i = 0; i < resultCount; ++i) {
            resultIndexSum += scoredCandidates[i].second;
            resultDistanceSum += scoredCandidates[i].first;
        }
    }
    const double searchMs = Milliseconds(Clock::now() - searchStart).count();

    // Overall elapsed time stops before the summary and final object destruction.
    const double totalMs = Milliseconds(Clock::now() - totalStart).count();
    std::cout << std::fixed << std::setprecision(3)
              << "\nSeed: " << SEED << '\n'
              << "Queries processed: " << queryCount << '\n'
              << "Total candidates: " << totalCandidates << '\n'
              << "Total neighbors returned: " << totalResults << '\n'
              << "Top-K index sum: " << resultIndexSum << '\n'
              << "Top-K squared-distance sum: " << resultDistanceSum << '\n'
              << "Loading (ms): " << loadingMs << '\n'
              << "Index construction (ms): " << indexingMs << '\n'
              << "Search, all queries (ms): " << searchMs << '\n'
              << "Average per query (ms): " << searchMs / queryCount << '\n'
              << "Throughput (queries/s): " << queryCount * 1000.0 / searchMs << '\n'
              << "Elapsed until results (ms): " << totalMs << '\n';

    return 0;
}
