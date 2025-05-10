// cuda/test_cuda.cpp
#include "hnswlib.h"        // This defines the namespace
#include "hnsw_cuda.h"
#include "space_l2.h"       // This has L2Space
#include "cuda_distance.cuh"

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    // Parameters
    int dim = 128;
    int num_elements = 10000;
    int num_queries = 100;
    
    std::cout << "Testing HNSW with CUDA acceleration" << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Number of elements: " << num_elements << std::endl;
    
    // Create random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<std::vector<float>> data(num_elements, std::vector<float>(dim));
    for (int i = 0; i < num_elements; i++) {
        for (int j = 0; j < dim; j++) {
            data[i][j] = dis(gen);
        }
    }
    
    // Build index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* index = 
        new hnswlib::HierarchicalNSW<float>(&space, num_elements, 16, 200);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Add points to index
    for (int i = 0; i < num_elements; i++) {
        index->addPoint(data[i].data(), i);
        if (i % 1000 == 0) {
            std::cout << "Added " << i << " points" << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index construction time: " << build_time.count() << " ms" << std::endl;
    
    // Perform some searches
    std::cout << "\nPerforming " << num_queries << " queries..." << std::endl;
    
    std::vector<float> query(dim);
    double total_search_time = 0;
    
    for (int i = 0; i < num_queries; i++) {
        // Generate random query
        for (int j = 0; j < dim; j++) {
            query[j] = dis(gen);
        }
        
        start = std::chrono::high_resolution_clock::now();
        auto result = index->searchKnn(query.data(), 10);
        end = std::chrono::high_resolution_clock::now();
        
        auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_search_time += search_time.count();
    }
    
    std::cout << "Average search time: " << total_search_time / num_queries << " μs" << std::endl;
    
    delete index;
    return 0;
}