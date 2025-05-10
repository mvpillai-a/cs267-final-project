// cuda/test_comparison.cpp
#include "hnswlib.h"
#include "cuda_l2_space_simple.h"  // Use the simple version
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    int dim = 128;
    int num_elements = 50000;
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<std::vector<float>> data(num_elements, std::vector<float>(dim));
    for (int i = 0; i < num_elements; i++) {
        for (int j = 0; j < dim; j++) {
            data[i][j] = dis(gen);
        }
    }
    
    // Test CPU version
    {
        std::cout << "Testing CPU version..." << std::endl;
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* index = 
            new hnswlib::HierarchicalNSW<float>(&space, num_elements, 16, 200);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_elements; i++) {
            index->addPoint(data[i].data(), i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "CPU construction time: " << cpu_time.count() << " sec" << std::endl;
        
        delete index;
    }
    
    // Test CUDA version
    {
        std::cout << "\nTesting CUDA version..." << std::endl;
        hnswlib::L2SpaceCUDASimple space(dim);  // Use simple CUDA space
        hnswlib::HierarchicalNSW<float>* index = 
            new hnswlib::HierarchicalNSW<float>(&space, num_elements, 16, 200);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_elements; i++) {
            index->addPoint(data[i].data(), i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto cuda_time = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "CUDA construction time: " << cuda_time.count() << " sec" << std::endl;
        
        delete index;
    }
    
    return 0;
}