#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono> // For timing

#include "../src/nnd.h"

using namespace nndescent;
using namespace std::chrono;

int main()
{
    // Load dataset from file
    std::ifstream infile("100k_dataset.txt");

    if (!infile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }

    // Read the dataset into a vector
    std::vector<float> values;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream ss(line);
        float value;
        while (ss >> value) {
            values.push_back(value);
        }
    }

    infile.close();

    int num_points = 100000;

    // Create the data matrix
    Matrix<float> data(num_points, values);

    // Define algorithm parameters
    Parms parms;
    parms.n_neighbors = 4;

    // Start timing graph construction
    auto start = high_resolution_clock::now();

    // Run nearest neighbor descent algorithm.
    NNDescent nnd = NNDescent(data, parms);

    // End timing graph construction
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "Graph construction time for 100K points: " << duration.count() << " milliseconds\n";

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_indices = nnd.neighbor_indices;
    Matrix<float> nn_distances = nnd.neighbor_distances;

    // // Use brute force algorithm for exact values.
    // parms.algorithm = "bf";
    // NNDescent nnd_bf = NNDescent(data, parms);

    // // Return exact NN-Matrix.
    // Matrix<int> nn_indices_ect = nnd_bf.neighbor_indices;
    // Matrix<float> nn_distances_ect = nnd_bf.neighbor_distances;

    // Print NN-graphs
    // std::cout << "\nNearest neighbor graph indices\n" << nn_indices
    //     << "\nExact nearest neighbor graph indices\n" << nn_indices_ect
    //     << "\nNearest neighbor graph distances\n" << nn_distances
    //     << "\nExact nearest neighbor graph distances\n" << nn_distances_ect;

    return 0;
}
