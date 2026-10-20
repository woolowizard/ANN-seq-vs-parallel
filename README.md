Some info:

1) The purpose of this project is to implement a sequential and parallel version of ANN (Approximate Nearest Neighboor) method for vectors similarity search. 

Dataset page: http://corpus-texmex.irisa.fr/

Run from the data directory:

```bash
curl -s ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz | tar -xz
```

`curl -s` downloads the archive without showing progress. The pipe passes it directly to `tar -xz`, which decompresses and extracts it into the `sift/` directory.
