#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "flat_matrix.hpp"

// Index structures.
using Signature = std::vector<std::int64_t>;

struct SignatureHash {
    size_t operator()(const Signature& signature) const noexcept;
};

using BucketTable = std::unordered_map<Signature, std::vector<size_t>, SignatureHash>;

struct HashFunction {
    std::vector<float> a;
    float b;

    HashFunction(int dim, float w, std::mt19937& generator);
};

using HashTables = std::vector<std::vector<HashFunction>>;

// Hash generation and evaluation.
HashTables generateHashTable(int L, int M, int dim, float w, std::mt19937& generator);

std::int64_t computeHash(
    const float* x, size_t dim, const std::vector<float>& a, float b, float w
);

Signature computeSignature(
    const float* vector, size_t dim,
    const std::vector<HashFunction>& functions, float w
);

// Sequential index construction and search for one query.
std::vector<BucketTable> buildBuckets(
    const FlatMatrix& vectors, const HashTables& hashTables, float w
);

float calculate_squared_distance(const float* v, const float* q, size_t dim);

std::vector<size_t> searchQuery(
    const FlatMatrix& vectors, const float* query, size_t queryDim,
    const HashTables& hashTables, const std::vector<BucketTable>& buckets,
    float w, size_t K
);

double calculateRecall(
    const std::vector<std::vector<size_t>>& results,
    const std::vector<std::vector<std::int32_t>>& groundTruth,
    size_t K
);