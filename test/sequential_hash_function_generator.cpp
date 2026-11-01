
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <map>
#include <filesystem>
#include <utility>

#define FILE_PATH "/Users/andreacommodi/Desktop/github/reps/ANN-seq-vs-parallel/data/sift/sift_base.fvecs"

// Generate a random vector from a Gaussian distribution
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

int main() {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    float w = 13.0f;
    int dim = 128;
    int M = 2;
    int L = 10;

    auto hashTables = generateHashTable(L, M, dim, w, gen);

    // BEGIN TESTS //
    // BEGIN Vector Loading //

    auto byteTotali = std::filesystem::file_size(FILE_PATH);
    std::cout << "Bytes totali del file: " << byteTotali << std::endl;

    // Each vector has 4 bytes for dimension + 128 floats
    int32_t vec_tot = byteTotali / (4 + (128 * 4));

    std::cout << "Vettori totali: " << vec_tot << std::endl;

    std::ifstream file(FILE_PATH, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open dataset");
    }

    std::vector<std::vector<float>> vectors;
    vectors.reserve(vec_tot);

    // Load all vectors from the .fvecs file
    for (int i = 0; i < vec_tot; i++) {

        // Read dimension for each vector
        int32_t vec_dim;
        if (!file.read(reinterpret_cast<char*>(&vec_dim), sizeof(vec_dim))) {
            throw std::runtime_error("Cannot read vector dimension");
        }

        if (vec_dim != 128) {
            throw std::runtime_error("Invalid vector dimension");
        }

        // Allocate space for 128 float components
        std::vector<float> x(vec_dim);

        // Read all vector components
        if (!file.read(
            reinterpret_cast<char*>(x.data()),
            vec_dim * sizeof(float)
        )) {
            throw std::runtime_error("Cannot read vector components");
        }

        vectors.push_back(std::move(x));
    }

    file.close();

    // END Vector Loading //

    // HashTable for mapping
    std::vector<std::map< std::vector<std::int64_t>, std::vector<size_t> >> buckets(L);

    /* After the cycle i will have:
     * buckets = {
     *              {
     *                  (0,1,29,...-19): [x1, ... , xp],
     *                  ...                                 # This is l = 1
     *                  (0,10,9,...-1): [xi | i |= (1,...,p)],
     *              },
     *              {
     *              ...                                     # This is l = 2
     *              },
     *              ...
     *              {
     *              ...                                        # This is l = L
     *              }
     * }
    */

    for (int i = 0; i < vectors.size(); i++) {
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


    /* BEGIN PRINT Statistics */
    for (size_t l = 0; l < buckets.size(); l++) {
        size_t total = 0;
        size_t singletons = 0;
        size_t maxSize = 0;

        for (const auto& [signature, indices] : buckets[l]) {
            total += indices.size();

            if (indices.size() == 1)
                singletons++;

            maxSize = std::max(maxSize, indices.size());
        }

        std::cout << "Table " << l << "\n";
        std::cout << "Buckets: " << buckets[l].size() << "\n";
        std::cout << "Total indices: " << total << "\n";
        std::cout << "Singletons: " << singletons << "\n";
        std::cout << "Max bucket size: " << maxSize << "\n";
        std::cout << "Average bucket size: "
                  << static_cast<double>(total) / buckets[l].size()
                  << "\n";
    }
    /* END PRINT Statistics */

    // END TESTS //

    return 0;
}
