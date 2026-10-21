Some info:

1) The purpose of this project is to implement a sequential and parallel version of ANN (Approximate Nearest Neighboor) method for vectors similarity search.

2) First, was implemented a brute force version of ANN that is equal to a KNN algorithm that compute similarity over all the vectors in baseline. The complexity of this implementation should be O(|b|*|q|) where |b|, |q| are respectively the cardinality of the baseline and query set.

Dataset page: http://corpus-texmex.irisa.fr/

Run from the data directory:

```bash
curl -s ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz | tar -xz
```

`curl -s` downloads the archive without showing progress. The pipe passes it directly to `tar -xz`, which decompresses and extracts it into the `sift/` directory.
