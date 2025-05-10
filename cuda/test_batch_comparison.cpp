// cuda/test_batch_comparison.cpp
#include "hnswlib.h"
#include "cuda_l2_space_batch.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Custom HNSW that processes multiple points at once
template<typename dist_t>
class BatchHNSW : public hnswlib::HierarchicalNSW<dist_t> {
    using Base = hnswlib::HierarchicalNSW<dist_t>;
    
public:
    using Base::Base;  // Inherit constructors
    
    void addPointBatch(const std::vector<const void*>& data_points, 
                       const std::vector<hnswlib::labeltype>& labels) {
        // Add points in batches to maximize CUDA efficiency
        for (size_t i = 0; i < data_points.size(); i++) {
            this->addPoint(data_points[i], labels[i]);
            
            // Force batch processing every N points
            if (i % 100 == 99) {
                auto& manager = hnswlib::CudaBatchManager::getInstance();
                manager.cleanup();
            }
        }
    }
};

int main() {
    int dim = 128;
    int num_elements = 50000;
    int batch_size = 1000;
    
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
        auto* index = new hnswlib::HierarchicalNSW<float>(&space, num_elements, 16, 200);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_elements; i++) {
            index->addPoint(data[i].data(), i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "CPU construction time: " << cpu_time.count() << " ms" << std::endl;
        
        delete index;
    }
    
    // Test CUDA version with batching
    {
        std::cout << "\nTesting CUDA version with batching..." << std::endl;
        hnswlib::L2SpaceCUDABatched space(dim);
        auto* index = new BatchHNSW<float>(&space, num_elements, 16, 200);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Add points in batches
        for (int i = 0; i < num_elements; i += batch_size) {
            std::vector<const void*> batch_data;
            std::vector<hnswlib::labeltype> batch_labels;
            
            for (int j = i; j < std::min(i + batch_size, num_elements); j++) {
                batch_data.push_back(data[j].data());
                batch_labels.push_back(j);
            }
            
            index->addPointBatch(batch_data, batch_labels);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "CUDA construction time: " << cuda_time.count() << " ms" << std::endl;
        
        delete index;
    }
    
    return 0;
}