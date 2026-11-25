#include "ann.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace {

// Genero un vettore distribuito normalmente per la funzione hash
void normalVector(float* a, size_t dim, std::mt19937& generator) {
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t d = 0; d < dim; ++d) {
        a[d] = distribution(generator);
    }
}

// Sommiamo blocchi di 16 componenti: il SIMD è dentro il blocco, controllo tra i blocchi.
float ManhattanDistanceWithinLimit(
    const std::uint8_t* v, const std::uint8_t* q, size_t dim,
    float limit, SearchStats& stats
) {
    std::uint64_t sum = 0;
    size_t processed = 0;
    while (processed < dim) {
        const size_t end = std::min(processed + 16, dim);
        #pragma omp simd reduction(+:sum)
        for (size_t d = processed; d < end; ++d) {
            // Convertire PRIMA della sottrazione: la differenza puo' essere negativa.
            const int difference = static_cast<int>(v[d]) - static_cast<int>(q[d]);
            sum += std::abs(difference);
        }
        processed = end;
        if (static_cast<float>(sum) > limit) {
            break;
        }
    }
    // Un solo aggiornamento, sia per la distanza completa sia per quella interrotta.
    stats.distanceComponents += processed;
    if (processed < dim) {
        ++stats.earlyExits;
    }
    return static_cast<float>(sum);
}

// Sommiamo blocchi di 16 componenti: il SIMD è dentro il blocco, controllo tra i blocchi.
float SquaredDistanceWithinLimit(
    const std::uint8_t* v, const std::uint8_t* q, size_t dim,
    float limit, SearchStats& stats
) {
    std::uint64_t sum = 0;
    size_t processed = 0;
    while (processed < dim) {
        const size_t end = std::min(processed + 16, dim);
        #pragma omp simd reduction(+:sum)
        for (size_t d = processed; d < end; ++d) {
            // Convertire PRIMA della sottrazione: la differenza puo' essere negativa.
            const int difference = static_cast<int>(v[d]) - static_cast<int>(q[d]);
            sum += difference * difference;
        }
        processed = end;
        if (static_cast<float>(sum) > limit) {
            break;
        }
    }
    // Un solo aggiornamento, sia per la distanza completa sia per quella interrotta.
    stats.distanceComponents += processed;
    if (processed < dim) {
        ++stats.earlyExits;
    }
    return static_cast<float>(sum);
}

} // namespace

HashTables generateHashTable(
    int L, int M, int dim, float w, std::mt19937& generator
) {
    if (L <= 0 || M <= 0 || dim <= 0 || !(w > 0.0f) || !std::isfinite(w)) {
        throw std::invalid_argument("Invalid hash parameters");
    }
    HashTables hashTables;
    hashTables.L = L;
    hashTables.M = M;
    hashTables.dim = dim;
    hashTables.data.resize(hashTables.L * hashTables.M * (hashTables.dim + 1));
    std::uniform_real_distribution<float> bias(0.0f, w);
    for (size_t l = 0; l < hashTables.L; ++l) {
        for (size_t m = 0; m < hashTables.M; ++m) {
            const size_t offset = (l * hashTables.M + m) * (hashTables.dim + 1);
            float* function = hashTables.data.data() + offset;
            normalVector(function, hashTables.dim, generator);
            function[hashTables.dim] = bias(generator);
        }
    }
    return hashTables;
}

// h(x) = floor((a^T x + b) / w). I coefficienti restano float, x usa byte.
std::int64_t computeHash(
    const std::uint8_t* x, size_t dim, const float* a, float b, float w
) {
    double projection = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        projection += static_cast<double>(a[d]) * x[d];
    }
    return static_cast<std::int64_t>(std::floor((projection + b) / w));
}

// Precompute 1/||x||_2 for query & vectors
std::vector<double> precomputeInverseNorms(const FlatMatrix& vectors) {
    std::vector<double> inverseNorms(vectors.rows);

    for (size_t i = 0; i < vectors.rows; ++i) {
        const std::uint8_t* v = vectors.row(i);
        std::uint64_t sumSquares = 0;

    #pragma omp simd reduction(+:sumSquares)
        for (size_t pos = 0; pos < vectors.dim; ++pos) {
            const auto value = static_cast<std::uint64_t>(v[pos]);
            sumSquares += value * value;
        }

        if (sumSquares == 0) {
            throw std::invalid_argument(
                "Cosine distance is undefined for zero vectors"
            );
        }

        inverseNorms[i] = 1.0 / std::sqrt(static_cast<double>(sumSquares));
    }

    return inverseNorms;
}

void computeSignature(
    const std::uint8_t* vector, const HashTables& hashTables,
    size_t table, float w, Signature& signature
) {
    for (size_t m = 0; m < hashTables.M; ++m) {
        const size_t offset = (table * hashTables.M + m) * (hashTables.dim + 1);
        const float* function = hashTables.data.data() + offset;
        signature[m] = computeHash(vector, hashTables.dim, function, function[hashTables.dim], w);
    }
}

std::vector<BucketTable> buildBuckets(
    const FlatMatrix& vectors, const HashTables& hashTables, float w, int threads
) {
    if (vectors.dim != hashTables.dim || threads <= 0) {
        throw std::invalid_argument("Invalid index dimensions or thread count");
    }
    // Controlliamo una sola volta che tutti gli indici siano rappresentabili a 32 bit.
    if (vectors.rows > 0 && vectors.rows - 1 > std::numeric_limits<VectorId>::max()) {
        throw std::length_error("Dataset exceeds the range of 32-bit bucket indices");
    }
    std::vector<BucketTable> buckets(hashTables.L);
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (size_t l = 0; l < hashTables.L; ++l) {
        Signature signature(hashTables.M);
        for (size_t i = 0; i < vectors.rows; ++i) {
            computeSignature(vectors.row(i), hashTables, l, w, signature);
            buckets[l][signature].push_back(static_cast<VectorId>(i));
        }
    }
    return buckets;
}

float calculate_cosine_distance(
        const std::uint8_t* v,
        const std::uint8_t* q,
        size_t dim,
        double queryInverseNorm,
        double vectorInverseNorm
    ) {
    std::uint64_t dot = 0;
    #pragma omp simd reduction(+:dot)
        for (size_t d = 0; d < dim; ++d) {
            dot += static_cast<std::uint64_t>(v[d]) * q[d];
        }

        const double distance = 1.0 - static_cast<double>(dot) * queryInverseNorm * vectorInverseNorm;

        return static_cast<float>(distance);
}

float calculate_manhattan_distance(const std::uint8_t* v, const std::uint8_t* q, size_t dim) {
    std::uint64_t sum = 0;
    // SIMD usa piu' corsie dello stesso core: questa direttiva non crea altri thread.
    #pragma omp simd reduction(+:sum)
    for (size_t d = 0; d < dim; ++d) {
        const int difference = static_cast<int>(v[d]) - static_cast<int>(q[d]);
        sum += std::abs(difference);
    }
    return static_cast<float>(sum);
}


float calculate_squared_distance(const std::uint8_t* v, const std::uint8_t* q, size_t dim) {
    std::uint64_t sum = 0;
    // SIMD usa piu' corsie dello stesso core: questa direttiva non crea altri thread.
    #pragma omp simd reduction(+:sum)
    for (size_t d = 0; d < dim; ++d) {
        const int difference = static_cast<int>(v[d]) - static_cast<int>(q[d]);
        sum += difference * difference;
    }
    return static_cast<float>(sum);
}

SearchBuffers::SearchBuffers(size_t vectorCount, size_t M, size_t K)
    : signature(M), visited(vectorCount / 32 + (vectorCount % 32 != 0)) {
    nearest.reserve(K);
}

std::vector<size_t> searchQuery(
    const FlatMatrix& vectors, const std::uint8_t* query, size_t queryDim,
    const HashTables& hashTables, const std::vector<BucketTable>& buckets,
    float w, size_t K, SearchBuffers& buffers, const std::string& which_distance,
    double queryInverseNorm, const std::vector<double>& vectorInverseNorms
) {
    if (queryDim != vectors.dim || queryDim != hashTables.dim) {
        throw std::invalid_argument("Query and index dimensions differ");
    }
    if (which_distance != "squared" && which_distance != "manhattan" && which_distance != "cosine") {
        throw std::invalid_argument("Unknown distance: " + which_distance);
    }
    if (which_distance == "cosine" &&
        (vectorInverseNorms.size() != vectors.rows ||
         !(queryInverseNorm > 0.0) || !std::isfinite(queryInverseNorm))) {
        throw std::invalid_argument("Cosine distance requires precomputed inverse norms");
    }
    // Azzeriamo il contenuto, conservando la memoria allocata per la query successiva.
    std::fill(buffers.visited.begin(), buffers.visited.end(), 0);
    buffers.nearest.clear();
    buffers.worst = 0;
    buffers.stats = {};
    K = std::min(K, vectors.rows);
    if (K == 0) {
        return {};
    }

    auto& nearest = buffers.nearest;
    auto& stats = buffers.stats;
    for (size_t l = 0; l < hashTables.L; ++l) {
        computeSignature(query, hashTables, l, w, buffers.signature);
        stats.hashEvaluations += hashTables.M;
        ++stats.bucketLookups;
        const auto bucket = buckets[l].find(buffers.signature); // Cerco nella hash table l --> bucket con la firma della query corrente
        if (bucket == buckets[l].end()) {
            continue;
        }
        stats.candidateVisits += bucket->second.size();
        for (size_t index : bucket->second) { // Inizio il ciclo sui candidati in questo bucket
            // Il bit numero index indica se abbiamo gia' valutato questo candidato --> Qui la logica è recuperata dalla versione di LSHKit
            const size_t word = index / 32; // Mi dice a quale posizione del vettore visited cercare il bit di segnalazione per il vettore i-esimo
            const std::uint32_t bit = std::uint32_t{1} << (index % 32); // Mi dice quale bit è quello che mi interessa: 1 visitato 0 non visitato
            if (buffers.visited[word] & bit) {
                continue; // Qui skippo il compute della distanza
            }
            buffers.visited[word] |= bit; // Incremento il bit del vettore candidato
            ++stats.distanceComputations;

            float distance;
            if (which_distance == "squared") {
                if (nearest.size() < K) {
                    distance = calculate_squared_distance(vectors.row(index), query, vectors.dim);
                    stats.distanceComponents += vectors.dim;
                } else {
                    // Qui usiamo la funzione che compara la distanza calcolata su un blocco parziale e vede se è già > della peggiore, in quel caso usciamo prima senza calcolarla tutta
                    distance = SquaredDistanceWithinLimit(
                        vectors.row(index), query, vectors.dim, nearest[buffers.worst].first, stats
                    );
                }
            } else if (which_distance == "manhattan") {
                if (nearest.size() < K) {
                    distance = calculate_manhattan_distance(vectors.row(index), query, vectors.dim);
                    stats.distanceComponents += vectors.dim;
                } else {
                    // Qui usiamo la funzione che compara la distanza calcolata su un blocco parziale e vede se è già > della peggiore, in quel caso usciamo prima senza calcolarla tutta
                    distance = ManhattanDistanceWithinLimit(
                        vectors.row(index), query, vectors.dim, nearest[buffers.worst].first, stats
                    );
                }
            } else if (which_distance == "cosine") {
                distance = calculate_cosine_distance(
                    vectors.row(index), query, vectors.dim,
                    queryInverseNorm, vectorInverseNorms[index]
                );
                stats.distanceComponents += vectors.dim;
            } else {
                throw std::invalid_argument("Unknown distance: " + which_distance);
            }

            const std::pair<float, size_t> candidate{distance, index};
            if (nearest.size() < K) { // Se non ho ancora K candidati questo lo inserisco senza confronto
                nearest.push_back(candidate);
            } else if (candidate < nearest[buffers.worst]) { // Se ho K candidati allora vedo se current candidate < del worst candidate dei candidati registrati
                nearest[buffers.worst] = candidate; // Se lo è sostituisco il peggiore con il current
            } else {
                continue;
            }
            // Riassegno il peggiore per poi ricominciare con la prossima query
            buffers.worst = std::max_element(nearest.begin(), nearest.end()) - nearest.begin();
        }
    }

    // Ordiniamo solo le K coppie conservate. Se ho distanze pari allora vince l'indice minore.
    std::sort(nearest.begin(), nearest.end());
    std::vector<size_t> neighbors;
    neighbors.reserve(nearest.size());
    for (const auto& neighbor : nearest) {
        neighbors.push_back(neighbor.second);
    }
    return neighbors;
}

// Esamina tutta la base, ma conserva soltanto K coppie (distanza, indice).
// Come nell'ANN, aggiorniamo la posizione della peggiore solo quando cambia il buffer.
std::vector<size_t> bruteForceQuery(
    const FlatMatrix& vectors, const std::uint8_t* query, size_t K,
    std::vector<std::pair<float, size_t>>& nearest, const std::string& which_distance,
    double queryInverseNorm, const std::vector<double>& vectorInverseNorms
) {
    if (which_distance != "squared" && which_distance != "manhattan" && which_distance != "cosine") {
        throw std::invalid_argument("Unknown distance: " + which_distance);
    }
    if (which_distance == "cosine" &&
        (vectorInverseNorms.size() != vectors.rows ||
         !(queryInverseNorm > 0.0) || !std::isfinite(queryInverseNorm))) {
        throw std::invalid_argument("Cosine distance requires precomputed inverse norms");
    }
    nearest.clear(); // La capacita' viene riutilizzata per la prossima query.
    K = std::min(K, vectors.rows);
    if (K == 0) {
        return {};
    }
    size_t worst = 0;
    for (size_t i = 0; i < vectors.rows; ++i) {
        float distance;
        if (which_distance == "squared") {
            distance = calculate_squared_distance(vectors.row(i), query, vectors.dim);
        } else if (which_distance == "manhattan") {
            distance = calculate_manhattan_distance(vectors.row(i), query, vectors.dim);
        } else if (which_distance == "cosine") {
            distance = calculate_cosine_distance(
                vectors.row(i), query, vectors.dim,
                queryInverseNorm, vectorInverseNorms[i]
            );
        } else {
            throw std::invalid_argument("Unknown distance: " + which_distance);
        }

        const std::pair<float, size_t> candidate{distance, i};
        if (nearest.size() < K) {
            nearest.push_back(candidate);
        } else if (candidate < nearest[worst]) {
            nearest[worst] = candidate;
        } else {
            continue;
        }
        worst = std::max_element(nearest.begin(), nearest.end()) - nearest.begin();
    }
    std::sort(nearest.begin(), nearest.end());
    std::vector<size_t> neighbors;
    neighbors.reserve(nearest.size());
    for (const auto& neighbor : nearest) {
        neighbors.push_back(neighbor.second);
    }
    return neighbors;
}

SearchStats& SearchStats::operator+=(const SearchStats& other) {
    hashEvaluations += other.hashEvaluations;
    bucketLookups += other.bucketLookups;
    candidateVisits += other.candidateVisits;
    distanceComputations += other.distanceComputations;
    distanceComponents += other.distanceComponents;
    earlyExits += other.earlyExits;
    return *this;
}


void printWeakScalingStats(
    std::ostream& output, const SearchStats& stats,
    size_t queryCount, size_t baseCount, size_t dim,
    const double SeqSearchTime, const double ParSearchTimeP
) {
    const double allDistances = static_cast<double>(queryCount) * baseCount;
    const double allComponents = static_cast<double>(stats.distanceComputations) * dim;
    output << "Query hash evaluations: " << stats.hashEvaluations << '\n'
           << "Bucket lookups: " << stats.bucketLookups << '\n'
           << "Candidate visits (including duplicates): " << stats.candidateVisits << '\n'
           << "Distance computations (started): " << stats.distanceComputations << '\n'
           << "Distance early exits: " << stats.earlyExits << '\n'
           << "Distance components evaluated: " << stats.distanceComponents << '\n'
           << "Distances per query: "
           << (queryCount ? static_cast<double>(stats.distanceComputations) / queryCount : 0.0) << '\n'
           << "Distances vs brute force (%): "
           << (allDistances ? 100.0 * stats.distanceComputations / allDistances : 0.0) << '\n'
           << "Distance components skipped (%): "
           << (allComponents ? 100.0 * (1.0 - stats.distanceComponents / allComponents) : 0.0) << '\n'
           << "Weak scaling (T1/Tp): "
           << (SeqSearchTime/ParSearchTimeP) << '\n';
}

void printSearchStats(
    std::ostream& output, const SearchStats& stats,
    size_t queryCount, size_t baseCount, size_t dim
) {
    const double allDistances = static_cast<double>(queryCount) * baseCount;
    const double allComponents = static_cast<double>(stats.distanceComputations) * dim;
    output << "Query hash evaluations: " << stats.hashEvaluations << '\n'
           << "Bucket lookups: " << stats.bucketLookups << '\n'
           << "Candidate visits (including duplicates): " << stats.candidateVisits << '\n'
           << "Distance computations (started): " << stats.distanceComputations << '\n'
           << "Distance early exits: " << stats.earlyExits << '\n'
           << "Distance components evaluated: " << stats.distanceComponents << '\n'
           << "Distances per query: "
           << (queryCount ? static_cast<double>(stats.distanceComputations) / queryCount : 0.0) << '\n'
           << "Distances vs brute force (%): "
           << (allDistances ? 100.0 * stats.distanceComputations / allDistances : 0.0) << '\n'
           << "Distance components skipped (%): "
           << (allComponents ? 100.0 * (1.0 - stats.distanceComponents / allComponents) : 0.0) << '\n';
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
