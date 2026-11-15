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

int test_solution(
    const std::vector<Neighbor>& neighbor_vector,
    const std::vector<Neighbor>& final_structure,
    std::size_t k
) {
     bool invalid = neighbor_vector.size() < k ||
                    final_structure.size() < k;

     std::size_t i = 0;

     while (!invalid && i < k) {
          invalid = neighbor_vector[i].distance !=
                    final_structure[i].distance;
          ++i;
     }

     return invalid; // 0 = valida, 1 = non valida
}

bool compare_distance(const Neighbor& a, const Neighbor& b) {
     return a.distance < b.distance;
}

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

     int n_similar = 5;
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

     /* ********************** */
     /* Begin SEQUENTIAL VERSION */
     /* ********************** */

     std::vector<float> query_vector;
     std::vector<Neighbor> neighbor_vector;

     query_vector = my_vector[100]; // Simulate the query vector

     double start_seq = omp_get_wtime();
     for (int i = 0; i < vec_tot; i++) {
          float distance = calculate_distance(my_vector[i], query_vector, 0);
          Neighbor node = Neighbor(distance, i);
          neighbor_vector.push_back(node);
     }

     // The idea is that every thread sort the private version of struct
     std::partial_sort(
        neighbor_vector.begin(),
        neighbor_vector.begin() + n_similar,
        neighbor_vector.end(),
        compare_distance);

     double time_seq = omp_get_wtime() - start_seq;



     /* ********************** */
     /* Begin PARALLEL VERSION */
     /* ********************** */
     int i = 0;
     int k = 0;
     std::vector<Neighbor> neighbor_vector_P;
     std::vector<Neighbor> final_structure;

     double start_par = omp_get_wtime();
     #pragma omp parallel shared(my_vector, vec_tot, n_similar) private(i, neighbor_vector_P)
     {
          int tid = omp_get_thread_num();

          #pragma omp for schedule(static) nowait
               for (i = 0; i < vec_tot; i++) {
                    float distance = calculate_distance(my_vector[i], query_vector, 0);
                    Neighbor node = Neighbor(distance, i);
                    neighbor_vector_P.push_back(node);
          }

          // The idea is that every thread sort the private version of struct
          std::partial_sort(
             neighbor_vector_P.begin(),
             neighbor_vector_P.begin() + n_similar,
             neighbor_vector_P.end(),
             compare_distance);

          // In this line i will allow thread one by one to write in a common struct
          #pragma omp critical
               for (int k = 0; k < n_similar; k++) {
                    final_structure.push_back(neighbor_vector_P[k]);
               }
     }

     // Sorting merged structure
     std::sort(
              final_structure.begin(),
              final_structure.end(),
              compare_distance
          );

     double time_par = omp_get_wtime() - start_par;

     int invalid = test_solution(
         neighbor_vector,
         final_structure,
         n_similar
     );

     std::cout << (invalid ? "Test KO\n" : "Test OK\n");

     std::cout << "Sequential time: " << time_seq * 1000 << " ms\n";
     std::cout << "Parallel time: " << time_par * 1000 << " ms\n";
     std::cout << "Speedup (Ts/Tp): " << time_seq / time_par << "x\n";

     return 0;
}
