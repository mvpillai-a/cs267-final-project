// cuda_benchmark.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cfloat>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/nnd.h"
#include "../src/nnd_cuda.h"

using namespace nndescent;
using namespace std::chrono;

// Function to check CUDA errors
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Function to calculate recall (accuracy) between CPU and CUDA results
double calculate_recall(const Matrix<int>& cpu_indices, const Matrix<int>& cuda_indices) {
    int total_neighbors = cpu_indices.nrows() * cpu_indices.ncols();
    int matches = 0;
    
    for (size_t i = 0; i < cpu_indices.nrows(); ++i) {
        for (size_t j = 0; j < cpu_indices.ncols(); ++j) {
            for (size_t k = 0; k < cuda_indices.ncols(); ++k) {
                if (cpu_indices(i, j) == cuda_indices(i, k)) {
                    matches++;
                    break;
                }
            }
        }
    }
    
    return static_cast<double>(matches) / total_neighbors;
}

// Function to check if the graph has valid neighbors
int count_valid_neighbors(const HeapList<float>& graph) {
    int valid_count = 0;
    for (size_t i = 0; i < graph.nheaps(); ++i) {
        for (size_t j = 0; j < graph.nnodes(); ++j) {
            if (graph.indices(i, j) != NONE && graph.indices(i, j) >= 0) {
                valid_count++;
            }
        }
    }
    return valid_count;
}

// Function to get CUDA device information
void printCudaInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(1);
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "  Device Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;
}

int main()
{
    // Print CUDA device information
    printCudaInfo();
    
    // Load dataset from file
    std::ifstream infile("1M_dataset.txt");
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

    int num_points = 1000000;
    
    // Create the data matrix
    Matrix<float> data(num_points, values);
    
    // Define algorithm parameters
    Parms parms;
    parms.n_neighbors = 10;
    parms.metric = "euclidean";
    parms.seed = 42;
    parms.verbose = true;
    parms.tree_init = false;
    parms.n_iters = 6;
    parms.max_candidates = 15;
    
    std::cout << "Dataset size: " << num_points << " points" << std::endl;
    std::cout << "Dimensions: " << data.ncols() << std::endl;
    std::cout << "Number of neighbors: " << parms.n_neighbors << std::endl;
    std::cout << "Iterations: " << parms.n_iters << std::endl;
    std::cout << std::endl;
    
    // Warm up CUDA
    std::cout << "Warming up CUDA..." << std::endl;
    cudaFree(0);
    
    // Run CPU implementation
    std::cout << "Running CPU implementation..." << std::endl;
    auto cpu_start = high_resolution_clock::now();
    
    NNDescent cpu_nnd(data, parms);
    
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU valid neighbors: " << count_valid_neighbors(cpu_nnd.current_graph) << std::endl;
    std::cout << std::endl;
    
    // Run CUDA implementation
    std::cout << "Running CUDA implementation..." << std::endl;

    // Initialize the graph
    HeapList<float> cuda_graph(num_points, parms.n_neighbors, FLT_MAX, NEW);
    RandomState rng_state;
    seed_state(rng_state, parms.seed);

    // Time the CUDA implementation
    auto cuda_start = high_resolution_clock::now();

    // Run CUDA NN-Descent
    nndescent_cuda::nn_descent_cuda(
        data, cuda_graph, parms.n_neighbors, rng_state,
        parms.max_candidates, parms.n_iters,
        parms.delta, parms.n_threads, parms.verbose, parms.metric
    );

    // Ensure all CUDA operations are complete
    cudaDeviceSynchronize();
    checkCudaError("After CUDA NN-Descent");
    
    auto cuda_end = high_resolution_clock::now();
    auto cuda_duration = duration_cast<milliseconds>(cuda_end - cuda_start);

    std::cout << "CUDA time: " << cuda_duration.count() << " ms" << std::endl;
    std::cout << "CUDA valid neighbors: " << count_valid_neighbors(cuda_graph) << std::endl;
    std::cout << std::endl;

    // Sort the results for proper comparison (heapsort)
    cpu_nnd.current_graph.heapsort();
    cuda_graph.heapsort();
    
    // Calculate speedup
    double speedup = static_cast<double>(cpu_duration.count()) / cuda_duration.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Calculate recall (accuracy)
    Matrix<int> cpu_indices = cpu_nnd.current_graph.indices;
    Matrix<int> cuda_indices = cuda_graph.indices;
    
    double recall = calculate_recall(cpu_indices, cuda_indices);
    std::cout << "Recall (accuracy): " << recall * 100 << "%" << std::endl;
    std::cout << std::endl;
    
    // Sample some results to manually verify
    std::cout << "Sample results comparison (first 5 nodes):" << std::endl;
    std::cout << "Node\tCPU neighbors\t\t\tCUDA neighbors" << std::endl;
    for (int i = 0; i < std::min(5, num_points); ++i) {
        std::cout << i << "\t";
        
        // CPU neighbors
        for (int j = 0; j < parms.n_neighbors; ++j) {
            std::cout << cpu_indices(i, j) << " ";
        }
        std::cout << "\t\t";
        
        // CUDA neighbors
        for (int j = 0; j < parms.n_neighbors; ++j) {
            std::cout << cuda_indices(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    // Check distances
    std::cout << "\nSample distance comparison (first 5 nodes):" << std::endl;
    std::cout << "Node\tCPU distances\t\t\tCUDA distances" << std::endl;
    for (int i = 0; i < std::min(5, num_points); ++i) {
        std::cout << i << "\t";
        
        // CPU distances
        for (int j = 0; j < parms.n_neighbors; ++j) {
            printf("%.4f ", cpu_nnd.current_graph.keys(i, j));
        }
        std::cout << "\t";
        
        // CUDA distances
        for (int j = 0; j < parms.n_neighbors; ++j) {
            printf("%.4f ", cuda_graph.keys(i, j));
        }
        std::cout << std::endl;
    }
    
    // Additional performance metrics
    std::cout << "\nDetailed Performance Metrics:" << std::endl;
    std::cout << "CPU throughput: " << (num_points * parms.n_neighbors) / (cpu_duration.count() / 1000.0) << " edges/second" << std::endl;
    std::cout << "CUDA throughput: " << (num_points * parms.n_neighbors) / (cuda_duration.count() / 1000.0) << " edges/second" << std::endl;
    
    return 0;
}