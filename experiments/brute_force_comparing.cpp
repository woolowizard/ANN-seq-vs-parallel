#include <fstream>
#include <filesystem>
#include <cstdint>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <set>

#define FILE_PATH "/Users/andreacommodi/Desktop/github/reps/ANN-seq-vs-parallel/data/sift/sift_base.fvecs"

struct Neighbor {
     float distance; // distance from query vector
     int id;          // index of node

     Neighbor(float d, int i) {
          distance = d;
          id = i;
     }
};

void sort_vector(Results, *r) {

};

float calculate_distance(const std::vector<float>& v, const std::vector<float>& q, const bool flag) {
     /*
      * Function that compute the L2 (euclidean) distance between two vectors
      * v: baseline vector
      * q: query vector
     */

     float distance = 0.000f;

     for (int i = 0; i < v.size(); i++) {
         distance = distance + sqrt(std::pow((v[i] - q[i]), 2));
     }

     if (flag == 1) {
          std::cout << std::fixed << std::setprecision(3) << distance << std::endl;
     }

     return distance;
}

int main() {

     auto byteTotali = std::filesystem::file_size(FILE_PATH);
     std::cout << "Bytes totali del file: " << byteTotali << std::endl;

     int32_t vec_dim;

     std::ifstream file(FILE_PATH, std::ios::in | std::ios::binary);
     file.read(reinterpret_cast<char*>(&vec_dim), sizeof(vec_dim));

     /* Those dim are fixed because i know the dataset */
     int32_t vec_tot = byteTotali / (4 + (128 * 4));

     std::cout << "Vettori totali: " << vec_tot << std::endl;

     /* After calculate how many rows there are in the file, i know how iteration i will do */

     std::vector<std::vector<float>> my_vector;

     /* Full [v1, ... , vn] with vi \in R^128 */
     for (int i = 0; i < vec_tot; i++) {
          my_vector.emplace_back(); // Aggiunge la riga i
          for (int j = 0; j < 128; j++) {
               float temp;
               file.read((char*)&temp, sizeof(temp));
               my_vector[i].push_back(temp);
          }
     }

     /* Begin distance calculation in sequential mode */
     std::vector<float> query_vector;
     std::vector<Neighbor> neighbor_vector;

     query_vector = my_vector[100]; // Simulate the query vector

     double start_seq = omp_get_wtime();
     for (int i = 0; i < vec_tot; i++) {
          float distance = calculate_distance(my_vector[i], query_vector, 0);
          Neighbor node = Neighbor(distance, i);
          neighbor_vector.push_back(node);
     }
     double time_seq = omp_get_wtime() - start_seq;

     /* Begin parallel version */
     int i = 0;
     std::vector<Neighbor> neighbor_vector_P;

     double start_par = omp_get_wtime();
     #pragma omp parallel shared(my_vector, vec_tot) private(i, neighbor_vector_P)
     {
          int tid = omp_get_thread_num();
          printf("Hello World from thread = %d\n", tid);

          #pragma omp for schedule(static) nowait
               for (i = 0; i < vec_tot; i++) {
                    float distance = calculate_distance(my_vector[i], query_vector, 0);
                    Neighbor node = Neighbor(distance, i);
                    neighbor_vector_P.push_back(node);
          }

          #pragma omp critical {

          }

     }
     double time_par = omp_get_wtime() - start_par;


     std::cout << "Tempo sequenziale: " << time_seq * 1000 << " ms\n";
     std::cout << "Tempo parallelo: " << time_par * 1000 << " ms\n";
     std::cout << "Speedup: " << time_seq / time_par << "x\n";

     return 0;
}
