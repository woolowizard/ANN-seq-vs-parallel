# ANN: sequential and parallel search

This project compares sequential and parallel implementations of Locality-Sensitive
Hashing (LSH) for approximate nearest neighbor search, using SIFT vectors.

An exact brute force implementation provides a reference by comparing each query
with every vector in the base dataset. Its distance computation cost is
O(base_count * query_count * dimension).

Dataset page: http://corpus-texmex.irisa.fr/

Run from the data directory:

```bash
curl -s ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz | tar -xz
```

`curl -s` downloads the archive without showing progress. The pipe passes it directly
to `tar -xz`, which decompresses and extracts it into the `sift/` directory.

On the `optimizations` branch, SIFT components are stored in a contiguous `uint8_t`
array. The loader reads one row of floats at a time and converts only integer values
in [0, 255], without loss. Hash function coefficients are also stored in a single
array: each function occupies `dim + 1` floats for its coefficients and bias.

Search reuses one `SearchBuffers` instance per thread: a signature, a bitset of
visited candidates, and at most K (distance, index) pairs. The bitset uses 32-bit
words and is cleared between queries. Once K results are available, distances are
computed in blocks of 16 components with `omp simd`. Computation stops when the
partial sum exceeds the worst retained squared distance. Index construction
distributes tables across threads.

Buckets use `std::map<Signature, std::vector<uint32_t>>`, as in the `multi-probe`
branch. Candidate indices occupy 4 bytes each instead of 8, halving the storage
for the indices themselves, excluding vector and map overhead.

The programs report hash evaluations, bucket lookups, candidate visits, distances
started, early exits, and distance components evaluated. Distances started include
those stopped early, so skipped components also matter when assessing work saved.
The distance count relative to brute force uses `query_count * base_count` as its
reference; it does not compare execution times.

Brute force uses the same data representation and SIMD squared Euclidean distance.
It processes all queries, parallelizing across queries, and retains only K pairs
per thread. Every distance is computed in full to provide an exact reference.

`INDEX_THREADS` in the ANN benchmarks controls index construction.
`thread_number` controls the thread counts used for ANN search;
`THREADS` in `benchmarks/brute_force_comparing.cpp` controls brute force threads.

```bash
cmake -S . -B cmake-build-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cmake-build-relwithdebinfo --parallel
```

The code is organized as follows:

```text
include/
    ann.hpp                 # Algorithm structures and function declarations
    dataset.hpp             # FlatMatrix and loader declarations
src/
    ann.cpp                 # Hashing, index construction, search, and statistics
    dataset.cpp             # fvecs and ground truth loaders
benchmarks/
    brute_force_comparing.cpp
    strong_scaling.cpp
    weak_scaling.cpp
test/
    fvecs_loader_test.cpp
    optimized_search_test.cpp
```

The three benchmark executables are `brute_force_comparing`, `strong_scaling`,
and `weak_scaling`. Strong scaling runs sequential and parallel searches on the
same queries, compares results, and reports recall. Weak scaling keeps 1000 queries
per thread and reports T1/Tp and recall for each configuration. Dataset loading and
index construction are excluded from search timing.

`fvecs_loader_test` checks data loading and conversion.
`optimized_search_test` uses small datasets to check the bitset, buffer reuse,
early distance termination, and brute force against a scalar reference.
CTest registers only these two correctness tests; benchmarks are run separately.
