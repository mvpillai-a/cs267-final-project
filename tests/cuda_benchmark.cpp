#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cfloat>
#include "../src/nnd.h"
#include "../src/nnd_cuda.h"

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
    parms.metric = "euclidean";
    parms.seed = 42;
    parms.verbose = false;
    
    // Run CPU implementation
    std::cout << "Running CPU implementation...\n";
    auto cpu_start = high_resolution_clock::now();
    
    NNDescent cpu_nnd(data, parms);
    
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU time: " << cpu_duration.count() << " ms\n\n";
    
    // Run CUDA implementation
    std::cout << "Running CUDA implementation...\n";
    
    // Initialize the graph with NONE values (as the CPU version does)
    HeapList<float> cuda_graph(num_points, parms.n_neighbors, FLT_MAX, NEW);
    RandomState rng_state;
    seed_state(rng_state, parms.seed);
    
    std::cout << "Graph initialized with " << cuda_graph.nheaps() << " nodes and " 
              << cuda_graph.nnodes() << " neighbors per node" << std::endl;
    
    auto cuda_start = high_resolution_clock::now();
    
    // Run CUDA NN-Descent
    nndescent_cuda::nn_descent_cuda(
        data, cuda_graph, parms.n_neighbors, rng_state,
        60,  // max_candidates
        10,  // n_iters
        parms.delta, 1, parms.verbose, parms.metric
    );
    
    cuda_graph.heapsort();
    
    auto cuda_end = high_resolution_clock::now();
    auto cuda_duration = duration_cast<milliseconds>(cuda_end - cuda_start);
    
    std::cout << "CUDA time: " << cuda_duration.count() << " ms\n\n";
    
    // Calculate speedup
    double speedup = static_cast<double>(cpu_duration.count()) / cuda_duration.count();
    std::cout << "Speedup: " << speedup << "x\n";
    
    return 0;
}