#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "dataset.hpp"

using Signature = std::vector<std::int64_t>;
using VectorId = std::uint32_t;

// Come in multi-probe: signature -> lista di indici, con 4 byte per candidato.
using BucketTable = std::map<Signature, std::vector<VectorId>>;

struct HashTables {
    size_t L = 0;
    size_t M = 0;
    size_t dim = 0;
    // Per ogni funzione: dim coefficienti a, seguiti dal bias b. w resta condiviso.
    std::vector<float> data;
};

struct SearchStats {
    std::uint64_t hashEvaluations = 0;
    std::uint64_t bucketLookups = 0;
    std::uint64_t candidateVisits = 0;       // Include le visite duplicate.
    std::uint64_t distanceComputations = 0;  // Distanze iniziate su candidati distinti.
    std::uint64_t distanceComponents = 0;    // Componenti effettivamente confrontate.
    std::uint64_t earlyExits = 0;            // Distanze fermate prima dell'ultima componente.

    SearchStats& operator+=(const SearchStats& other);
};

struct SearchBuffers {
    Signature signature; // firma della query che sto usando per cercare un bucket
    std::vector<std::uint32_t> visited; // Bitset: 32 candidati per parola, un bit per candidato.
    std::vector<std::pair<float, size_t>> nearest; // Coppia (distanza, indice)
    size_t worst = 0;                 // Posizione della peggiore delle K coppie.
    SearchStats stats;                // Statistiche dell'ultima query.

    // La creo una volta per thread poi riutilizzo la stessa per ogni query
    SearchBuffers(size_t vectorCount, size_t M, size_t K);
};

HashTables generateHashTable(int L, int M, int dim, float w, std::mt19937& generator);

std::int64_t computeHash(
    const std::uint8_t* x, size_t dim, const float* a, float b, float w
);

// Scrive nel buffer ricevuto, senza creare una nuova signature.
void computeSignature(
    const std::uint8_t* vector, const HashTables& hashTables,
    size_t table, float w, Signature& signature
);

// Ogni thread costruisce tabelle diverse, senza scritture sulla stessa mappa.
std::vector<BucketTable> buildBuckets(
    const FlatMatrix& vectors, const HashTables& hashTables, float w, int threads = 1
);

float calculate_squared_distance(const std::uint8_t* v, const std::uint8_t* q, size_t dim);

// Il buffer deve essere dimensionato per vectors.rows, hashTables.M e K.
// Restituisce al massimo K indici, ordinati per distanza e poi per indice.
std::vector<size_t> searchQuery(
    const FlatMatrix& vectors, const std::uint8_t* query, size_t queryDim,
    const HashTables& hashTables, const std::vector<BucketTable>& buckets,
    float w, size_t K, SearchBuffers& buffers
);

// Ricerca esatta: tutte le distanze complete, buffer di sole K coppie riutilizzabile.
std::vector<size_t> bruteForceQuery(
    const FlatMatrix& vectors, const std::uint8_t* query, size_t K,
    std::vector<std::pair<float, size_t>>& nearest
);

void printSearchStats(
    std::ostream& output, const SearchStats& stats,
    size_t queryCount, size_t baseCount, size_t dim
);

void printWeakScalingStats(
    std::ostream& output, const SearchStats& stats,
    size_t queryCount, size_t baseCount, size_t dim,
    double SeqSearchTime, double ParSearchTimeP
);

double calculateRecall(
    const std::vector<std::vector<size_t>>& results,
    const std::vector<std::vector<std::int32_t>>& groundTruth,
    size_t K
);
