#include "ann.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Riferimento indipendente: distanza scalare completa e ordinamento di tutti i punti.
std::vector<size_t> exactNeighbors(const FlatMatrix& vectors, const std::uint8_t* query, size_t K) {
    std::vector<std::pair<float, size_t>> all;
    for (size_t i = 0; i < vectors.rows; ++i) {
        std::uint64_t sum = 0;
        for (size_t d = 0; d < vectors.dim; ++d) {
            const int difference = static_cast<int>(vectors.row(i)[d]) - query[d];
            sum += difference * difference;
        }
        require(calculate_squared_distance(vectors.row(i), query, vectors.dim) == sum,
                "SIMD distance differs from scalar distance");
        all.emplace_back(static_cast<float>(sum), i);
    }
    std::sort(all.begin(), all.end());
    std::vector<size_t> result;
    for (size_t i = 0; i < std::min(K, all.size()); ++i) {
        result.push_back(all[i].second);
    }
    return result;
}

void checkDistanceMetrics() {
    const FlatMatrix vectors{{4, 1, 3, 3, 1, 6}, 3, 2};
    const FlatMatrix queries{{1, 1, 1, 0}, 2, 2};
    const auto vectorNorms = precomputeInverseNorms(vectors);
    const auto queryNorms = precomputeInverseNorms(queries);
    std::vector<std::pair<float, size_t>> nearest;

    require(bruteForceQuery(vectors, queries.row(0), 3, nearest, "squared") ==
                std::vector<size_t>({1, 0, 2}),
            "Squared brute force returned incorrect neighbors");
    require(bruteForceQuery(vectors, queries.row(0), 3, nearest, "manhattan") ==
                std::vector<size_t>({0, 1, 2}),
            "Manhattan brute force returned incorrect neighbors");
    require(nearest[0].first == 3.0f && nearest[1].first == 4.0f && nearest[2].first == 5.0f,
            "Manhattan brute force returned incorrect distances");

    HashTables hashes;
    hashes.L = 1;
    hashes.M = 1;
    hashes.dim = vectors.dim;
    hashes.data.resize(vectors.dim + 1, 0.0f); // Tutti i punti nello stesso bucket.
    const auto buckets = buildBuckets(vectors, hashes, 1.0f, 1);
    SearchBuffers buffers(vectors.rows, hashes.M, vectors.rows);
    const std::vector<size_t> cosineOrders[] = {{1, 0, 2}, {0, 1, 2}};
    for (size_t q = 0; q < queries.rows; ++q) {
        for (size_t K : {size_t{0}, size_t{1}, size_t{2}, size_t{5}}) {
            const std::vector<size_t> expected(
                cosineOrders[q].begin(),
                cosineOrders[q].begin() + std::min(K, vectors.rows)
            );
            require(bruteForceQuery(vectors, queries.row(q), K, nearest,
                                    "cosine", queryNorms[q], vectorNorms) == expected,
                    "Cosine brute force returned incorrect neighbors");
            require(searchQuery(vectors, queries.row(q), queries.dim, hashes, buckets,
                                1.0f, K, buffers, "cosine", queryNorms[q], vectorNorms) == expected,
                    "Cosine ANN differs from brute force");
            require(buffers.stats.earlyExits == 0 &&
                    buffers.stats.distanceComponents == (K == 0 ? 0 : vectors.rows * vectors.dim),
                    "Cosine search must compute complete distances");
        }
    }
    require(std::abs(nearest[0].first - (1.0 - 4.0 / std::sqrt(17.0))) < 1e-6,
            "Cosine brute force used incorrect inverse norms");
}

int main() {
    try {
        checkDistanceMetrics();
        FlatMatrix vectors;
        vectors.rows = 130; // Attraversa piu' parole del bitset.
        vectors.dim = 33;   // Include la coda dopo due blocchi SIMD da 16.
        vectors.data.resize(vectors.rows * vectors.dim);
        for (size_t i = 0; i < vectors.rows; ++i) {
            for (size_t d = 0; d < vectors.dim; ++d) {
                vectors.data[i * vectors.dim + d] = (i * 19 + d * 7) % 256;
            }
        }
        // Due punti uguali: incontriamo prima l'indice maggiore, ma deve vincere lo 0.
        std::fill_n(vectors.data.begin(), vectors.dim, 0);
        std::fill_n(vectors.data.end() - vectors.dim, vectors.dim, 0);

        // a=0 e b=0: tutti i punti nello stesso bucket, in entrambe le tabelle.
        HashTables hashes;
        hashes.L = 2;
        hashes.M = 1;
        hashes.dim = vectors.dim;
        hashes.data.resize(hashes.L * hashes.M * (hashes.dim + 1), 0.0f);
        auto buckets = buildBuckets(vectors, hashes, 1.0f, 1);
        require(buckets == buildBuckets(vectors, hashes, 1.0f, 2),
                "Parallel index differs from sequential index");
        auto& firstBucket = buckets[0].begin()->second;
        std::reverse(firstBucket.begin(), firstBucket.end());

        SearchBuffers buffers(vectors.rows, hashes.M, vectors.rows + 1);
        const auto bitsetMemory = buffers.visited.data();
        const auto signatureMemory = buffers.signature.data();
        const auto nearestMemory = buffers.nearest.data();
        std::vector<std::pair<float, size_t>> bruteBuffer;
        bruteBuffer.reserve(vectors.rows + 1);
        const auto bruteMemory = bruteBuffer.data();
        std::vector<std::uint8_t> query(vectors.dim);

        // Cambiare query riusando gli stessi buffer deve azzerare bitset, K coppie e contatori.
        for (int value : {0, 255, 0}) {
            std::fill(query.begin(), query.end(), value);
            for (size_t K : {size_t{1}, size_t{5}, vectors.rows + 1, size_t{0}}) {
                const auto expected = exactNeighbors(vectors, query.data(), K);
                require(searchQuery(vectors, query.data(), vectors.dim,
                                    hashes, buckets, 1.0f, K, buffers) == expected,
                        "Bounded search differs from exact search");
                require(bruteForceQuery(vectors, query.data(), K, bruteBuffer) == expected,
                        "Brute force differs from exact search");
                if (K != 0) {
                    require(buffers.stats.distanceComputations == vectors.rows,
                            "Duplicate candidates were scored more than once");
                    require(buffers.stats.candidateVisits == vectors.rows * hashes.L &&
                            buffers.stats.bucketLookups == hashes.L &&
                            buffers.stats.hashEvaluations == hashes.L * hashes.M,
                            "Incorrect work counters");
                    if (value == 0 && K == 1) {
                        require(buffers.stats.earlyExits > 0 &&
                                buffers.stats.distanceComponents < vectors.rows * vectors.dim,
                                "Early exit did not reduce evaluated components");
                    }
                } else {
                    require(buffers.stats.distanceComputations == 0,
                            "K=0 must not compute distances");
                }
            }
        }
        require(buffers.visited.data() == bitsetMemory &&
                buffers.signature.data() == signatureMemory &&
                buffers.nearest.data() == nearestMemory && bruteBuffer.data() == bruteMemory,
                "Search buffers were reallocated between queries");

        // Nessun bucket corrispondente: risultato vuoto e nessuna distanza iniziata.
        for (size_t l = 0; l < hashes.L; ++l) {
            hashes.data[l * (hashes.dim + 1) + hashes.dim] = 1.0f;
        }
        require(searchQuery(vectors, query.data(), vectors.dim,
                            hashes, buckets, 1.0f, 5, buffers).empty(),
                "Missing bucket must return no neighbors");
        require(buffers.stats.distanceComputations == 0, "Statistics were not reset");
        std::cout << "Optimized search and brute force tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
