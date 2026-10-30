
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <map>

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

    float w = 4.0f;
    int dim = 128;
    int M = 5;
    int L = 2;

    auto hashTables = generateHashTable(L, M, dim, w, gen);

    // BEGIN TESTS //
    // Compute every hash function for one vectore (after that generalized it into a function)

    std::vector<std::vector<float>> vectors;
    vectors.push_back(normalVector(dim, 0.0f, 1.5f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));
    vectors.push_back(normalVector(dim, 0.0f, 1.0f, gen));

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


    /* BEGIN PRINT BUCKETS */
    for (size_t l = 0; l < buckets.size(); l++) {
        std::cout << "\n===== TABLE " << l << " =====\n";

        for (const auto& [signature, indices] : buckets[l]) {

            std::cout << "Signature: [";
            for (auto value : signature) {
                std::cout << value << " ";
            }
            std::cout << "] -> Indices: [";

            for (size_t index : indices) {
                std::cout << index << " ";
            }

            std::cout << "]\n";
        }
    }
    /* END PRINT BUCKETS */

    // END TESTS //

    return 0;
}
