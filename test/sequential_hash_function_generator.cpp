
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

#include "fvecs_loader.hpp"

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
        const std::vector<std::int64_t>& signature
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

// Generate the random offset b ~ U(0, w)
float biasParameter(float w, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(0.0f, w);
    float bias = distribution(generator);

    return bias;
}

// Compute h(x) = floor((a^T x + b) / w)
std::int64_t computeHash(
    const std::vector<float>& x,
    const std::vector<float>& a,
    float b,
    float w
) {
    if (x.size() != a.size() || w <= 0.0f) {
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

// Generate M hash function
std::vector<HashFunction> generateSignature(
    int dim,
    int M,
    float w,
    std::mt19937& generator
) {
    std::vector<HashFunction> signature;
    signature.reserve(M);

    for (int i = 0; i < M; i++) {
        signature.emplace_back(dim, w, generator);
    }

    return signature;
}

// Generate complete HashTable function
std::vector<std::vector<HashFunction>> generateHashTable(
    int L,
    int M,
    int dim,
    float w,
    std::mt19937& generator
) {
    std::vector<std::vector<HashFunction>> hashTables;
    hashTables.reserve(L);

    for (int l = 0; l < L; l++) {
        hashTables.push_back(
            generateSignature(dim, M, w, generator)
        );
    }

    return hashTables;
}

float calculate_squared_distance(
    const std::vector<float>& v,
    const std::vector<float>& q
) {
    if (v.size() != q.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    double squaredDistance = 0.0;

    for (size_t i = 0; i < v.size(); ++i) {
        const double difference = static_cast<double>(v[i]) - q[i];
        squaredDistance += difference * difference;
    }

    return static_cast<float>(squaredDistance);
}

/**************************************************************/
/************************* BEGIN MAIN *************************/
/**************************************************************/

int main() {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    float w = 15.0f;
    int dim = 128;
    int M = 2;
    int L = 10;
    const size_t K = 5;

    auto hashTables = generateHashTable(L, M, dim, w, gen);

    // BEGIN Vector Loading //
    const std::filesystem::path datasetDirectory =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "data" / "sift";

    const auto vectors = loadFvecs(datasetDirectory / "sift_base.fvecs", dim);
    const auto queries = loadFvecs(datasetDirectory / "sift_query.fvecs", dim);

    if (queries.empty()) {
        throw std::runtime_error("Query dataset is empty");
    }

    std::cout << "Base vectors: " << vectors.size() << '\n';
    std::cout << "Query vectors: " << queries.size() << '\n';

    // END Vector Loading //

    std::vector< std::unordered_map<std::vector<std::int64_t>, std::vector<size_t>, SignatureHash> > buckets(L);

    for (size_t i = 0; i < vectors.size(); ++i) {
        for (int l = 0; l < L; l++) {
            std::vector<std::int64_t> signature; // Create signature vector for every table(l)
            signature.reserve(M);
            // Compute signature g_l(x_i)
            for (int m = 0; m < M; m++) {
                std::int64_t hash_value = computeHash(vectors[i], hashTables[l][m].a, hashTables[l][m].b, w);
                signature.push_back(hash_value); // A signature is easily a vector so i push el by el inside
            }
            // Insert vector index into the corresponding bucket
            buckets[l][signature].push_back(i); // Acceding into key signature and put vector index(i) inside, at table l
        }
    }

    /* START Searching */

    for (int q=0; q < 1; ++q) {

        std::vector<std::vector<std::int64_t>> querySignatures(
           L,
           std::vector<std::int64_t>(M)
        );

        for (int l = 0; l < L; ++l) {
            for (int m = 0; m < M; ++m) {
                querySignatures[l][m] = computeHash(
                    queries[q],
                    hashTables[l][m].a,
                    hashTables[l][m].b,
                    w
                );
            }
        }

        std::unordered_set<size_t> candidateIndexes;

        for (int l = 0; l < L; ++l) {
            auto it = buckets[l].find(querySignatures[l]);

            if (it != buckets[l].end()) {
                candidateIndexes.insert(
                    it->second.begin(),
                    it->second.end()
                );
            }
        }

        /* Search neighbors in the candidate space */
        std::vector<std::pair<float, size_t>> scoredCandidates;
        scoredCandidates.reserve(candidateIndexes.size());

        for (size_t index : candidateIndexes) {
            float distance = calculate_squared_distance(vectors[index], queries[q]);

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

        std::cout << "Candidates: " << candidateIndexes.size() << '\n';
        std::cout << "Top " << resultCount << " nearest neighbors:\n";

        for (size_t i = 0; i < resultCount; ++i) {
            const auto& [squaredDistance, index] = scoredCandidates[i];

            std::cout << "Index: " << index
                      << ", squared distance: " << squaredDistance
                      << '\n';
        }

    }

    /* END Searching */

    /* BEGIN PRINT Statistics */
    /*
    *bool LOG = false;
    *if (LOG) {
    *    for (size_t l = 0; l < buckets.size(); l++) {
    *        size_t total = 0;
    *        size_t singletons = 0;
    *        size_t maxSize = 0;
    *
    *        for (const auto& [signature, indices] : buckets[l]) {
    *            total += indices.size();
    *
    *            if (indices.size() == 1)
    *                singletons++;
    *
    *            maxSize = std::max(maxSize, indices.size());
    *        }
    *
    *        std::cout << "Table " << l << "\n";
    *        std::cout << "Buckets: " << buckets[l].size() << "\n";
    *        std::cout << "Total indices: " << total << "\n";
    *        std::cout << "Singletons: " << singletons << "\n";
    *        std::cout << "Max bucket size: " << maxSize << "\n";
    *        std::cout << "Average bucket size: "
    *                  << static_cast<double>(total) / buckets[l].size()
    *                  << "\n";
    *    }
    *}
    */
    /*END PRINT Statistics */

    // END TESTS //

    return 0;
}
