// Testing std::partial_sort method for sorting vectors

#include <iostream>     // std::cout
#include <algorithm>    // std::partial_sort
#include <vector>       // std::vector

struct Neighbor {
    float distance; // distance from query vector
    int id;          // index of node

    Neighbor(float d, int i) {
        distance = d;
        id = i;
    }
};

bool compare_distance(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance;
}

int main () {
    int myints[] = {9,8,7,6,5,4,3,2,1};
    std::vector<int> myvector (myints, myints+9);

    Neighbor node1(1.0, 1);
    Neighbor node2(2.0, 2);
    Neighbor node3(1.5, 2);

    std::vector<Neighbor> neighbor_vector;

    neighbor_vector.push_back(node1);
    neighbor_vector.push_back(node2);
    neighbor_vector.push_back(node3);

    std::partial_sort(
        neighbor_vector.begin(),
        neighbor_vector.begin() + 2,
        neighbor_vector.end(),
        compare_distance
    );

    for (int i=0; i < 3; i++){
        std::cout << neighbor_vector[i].distance << std::endl;
    }
    return 0;
}
