#include "ann.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace {

// Generate a random vector a ~ N(mean, stddev).
std::vector<float> normalVector(
    int dim, float mean, float stddev, std::mt19937& generator
) {
    std::vector<float> vec(dim);
    std::normal_distribution<float> distribution(mean, stddev);
    for (int i = 0; i < dim; ++i) {
        vec[i] = distribution(generator);
    }
    return vec;
}

// Generate the random offset b ~ U(0, w).
float biasParameter(float w, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(0.0f, w);
    return distribution(generator);
}

} // namespace

size_t SignatureHash::operator()(const Signature& signature) const noexcept {
    size_t seed = 0;
    for (std::int64_t value : signature) {
        const size_t valueHash = std::hash<std::int64_t>{}(value);
        seed ^= valueHash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

HashFunction::HashFunction(int dim, float w, std::mt19937& generator)
    : a(normalVector(dim, 0.0f, 1.0f, generator)),
      b(biasParameter(w, generator))
{}

HashTables generateHashTable(
    int L, int M, int dim, float w, std::mt19937& generator
) {
    HashTables hashTables(L);
    for (auto& table : hashTables) {
        table.reserve(M);
        for (int m = 0; m < M; ++m) {
            table.emplace_back(dim, w, generator);
        }
    }
    return hashTables;
}

// Compute h(x) = floor((a^T x + b) / w).
std::int64_t computeHash(
    const float* x, size_t dim, const std::vector<float>& a, float b, float w
) {
    if (dim != a.size() || w <= 0.0f) {
        throw std::invalid_argument("Invalid dimensions or w");
    }

    double projection = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        projection += static_cast<double>(a[i]) * x[i];
    }
    return static_cast<std::int64_t>(std::floor((projection + b) / w));
}

Signature computeSignature(
    const float* vector, size_t dim,
    const std::vector<HashFunction>& functions, float w
) {
    Signature signature;
    signature.reserve(functions.size());
    for (const auto& function : functions) {
        signature.push_back(computeHash(vector, dim, function.a, function.b, w));
    }
    return signature;
}

std::vector<BucketTable> buildBuckets(
    const FlatMatrix& vectors, const HashTables& hashTables, float w
) {
    std::vector<BucketTable> buckets(hashTables.size());
    for (size_t i = 0; i < vectors.rows; ++i) {
        const float* vector = vectors.row(i);
        for (size_t l = 0; l < hashTables.size(); ++l) {
            const auto signature = computeSignature(vector, vectors.dim, hashTables[l], w);
            buckets[l][signature].push_back(i);
        }
    }
    return buckets;
}

float calculate_squared_distance(const float* v, const float* q, size_t dim) {
    double squaredDistance = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double difference = static_cast<double>(v[i]) - q[i];
        squaredDistance += difference * difference;
    }
    return static_cast<float>(squaredDistance);
}

std::vector<size_t> searchQuery(
    const FlatMatrix& vectors, const float* query, size_t queryDim,
    const HashTables& hashTables, const std::vector<BucketTable>& buckets,
    float w, size_t K
) {
    std::unordered_set<size_t> candidateIndexes;
    for (size_t l = 0; l < hashTables.size(); ++l) {
        const auto signature = computeSignature(query, queryDim, hashTables[l], w);
        const auto it = buckets[l].find(signature);
        if (it != buckets[l].end()) {
            candidateIndexes.insert(it->second.begin(), it->second.end());
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

    std::vector<size_t> neighbors;
    neighbors.reserve(resultCount);
    for (size_t j = 0; j < resultCount; ++j) {
        neighbors.push_back(scoredCandidates[j].second);
    }
    return neighbors;
}

double calculateRecall(
    const std::vector<std::vector<size_t>>& results,
    const std::vector<std::vector<std::int32_t>>& groundTruth,
    size_t K
) {
    size_t correctNeighbors = 0;
    const size_t queryCount = results.size();

    for (size_t q = 0; q < queryCount; ++q) {
        const auto& retrievedNeighbors = results[q]; // Retrieve ANN results for query q
        const auto& exactNeighbors = groundTruth[q]; // Retrieve GT results for query q

        for (size_t j = 0; j < K; ++j) { // Iterating the vector position
            const size_t expectedIndex = exactNeighbors[j]; // GT index

            // Searching if GT index are in ANN
            for (size_t retrievedIndex : retrievedNeighbors) {
                if (retrievedIndex == expectedIndex) {
                    ++correctNeighbors;
                    break; // NB founded --> go to the next j
                }
            }
        }
    }

    const double expectedNeighbors = static_cast<double>(queryCount) * K;
    return correctNeighbors / expectedNeighbors;
}
