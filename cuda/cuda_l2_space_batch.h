// cuda/cuda_l2_space_batch.h
#pragma once
#include "space_l2.h"
#include "cuda_distance.cuh"
#include "cuda_hnsw_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace hnswlib {

// Batch manager for CUDA distance calculations
class CudaBatchManager {
private:
    struct BatchContext {
        const float* query;
        std::vector<const float*> vectors;
        std::vector<float> results;
        std::unordered_map<const float*, size_t> vector_to_index;
        bool processed = false;
    };
    
    // CUDA resources
    float* d_vectors = nullptr;
    float* d_query = nullptr;
    float* d_distances = nullptr;
    size_t max_batch_size = 512;
    size_t allocated_dim = 0;
    bool initialized = false;
    
    // Batch tracking
    std::unordered_map<const float*, BatchContext> active_batches;
    const float* current_query = nullptr;
    
    // Statistics
    std::atomic<size_t> cuda_kernel_calls{0};
    std::atomic<size_t> total_distances{0};

public:
    static CudaBatchManager& getInstance() {
        static CudaBatchManager instance;
        return instance;
    }
    
    void init(size_t dim) {
        if (!initialized || allocated_dim != dim) {
            if (initialized) {
                cudaFree(d_vectors);
                cudaFree(d_query);
                cudaFree(d_distances);
            }
            
            cudaMalloc(&d_vectors, max_batch_size * dim * sizeof(float));
            cudaMalloc(&d_query, dim * sizeof(float));
            cudaMalloc(&d_distances, max_batch_size * sizeof(float));
            
            allocated_dim = dim;
            initialized = true;
        }
    }
    
    void processBatch(BatchContext& batch, size_t dim) {
        if (batch.processed || batch.vectors.empty()) return;
        
        size_t batch_size = batch.vectors.size();
        
        // Prepare batch data
        std::vector<float> vectors_data;
        vectors_data.reserve(batch_size * dim);
        
        for (const float* vec : batch.vectors) {
            vectors_data.insert(vectors_data.end(), vec, vec + dim);
        }
        
        // Process in chunks if needed
        for (size_t offset = 0; offset < batch_size; offset += max_batch_size) {
            size_t chunk_size = std::min(max_batch_size, batch_size - offset);
            
            // Copy to GPU
            cudaMemcpy(d_query, batch.query, dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vectors, vectors_data.data() + offset * dim, 
                      chunk_size * dim * sizeof(float), cudaMemcpyHostToDevice);
            
            // Launch kernel
            launch_l2_distance_kernel(d_vectors, d_query, d_distances, chunk_size, dim);
            
            // Copy results back
            cudaMemcpy(batch.results.data() + offset, d_distances, 
                      chunk_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            cuda_kernel_calls++;
        }
        
        batch.processed = true;
        total_distances += batch_size;
    }
    
    float computeDistance(const float* query, const float* vector, size_t dim) {
        // Check if we're starting a new query
        if (query != current_query) {
            // Process previous batch if exists
            if (current_query != nullptr) {
                auto it = active_batches.find(current_query);
                if (it != active_batches.end() && !it->second.processed) {
                    processBatch(it->second, dim);
                }
            }
            current_query = query;
        }
        
        // Get or create batch for this query
        auto& batch = active_batches[query];
        if (batch.query != query) {
            batch.query = query;
            batch.vectors.clear();
            batch.results.clear();
            batch.vector_to_index.clear();
            batch.processed = false;
        }
        
        // Check if we already have this vector in the batch
        auto vec_it = batch.vector_to_index.find(vector);
        if (vec_it != batch.vector_to_index.end()) {
            if (batch.processed) {
                return batch.results[vec_it->second];
            }
        } else {
            // Add to batch
            size_t index = batch.vectors.size();
            batch.vectors.push_back(vector);
            batch.results.resize(batch.vectors.size());
            batch.vector_to_index[vector] = index;
            
            // Process batch if it's large enough
            if (batch.vectors.size() >= max_batch_size && !batch.processed) {
                processBatch(batch, dim);
            }
        }
        
        // If not processed yet, compute on CPU as fallback
        if (!batch.processed) {
            float sum = 0;
            for (size_t i = 0; i < dim; i++) {
                float diff = query[i] - vector[i];
                sum += diff * diff;
            }
            return sum;
        }
        
        return batch.results[batch.vector_to_index[vector]];
    }
    
    void cleanup() {
        // Clean up old batches to prevent memory growth
        if (active_batches.size() > 100) {
            active_batches.clear();
            current_query = nullptr;
        }
    }
    
    ~CudaBatchManager() {
        if (initialized) {
            std::cout << "CUDA batch statistics:" << std::endl;
            std::cout << "  Kernel calls: " << cuda_kernel_calls << std::endl;
            std::cout << "  Total distances: " << total_distances << std::endl;
            std::cout << "  Average batch size: " 
                      << (cuda_kernel_calls > 0 ? total_distances / cuda_kernel_calls : 0) 
                      << std::endl;
            
            cudaFree(d_vectors);
            cudaFree(d_query);
            cudaFree(d_distances);
        }
    }
};

// Batched CUDA distance function
float L2SqrCUDABatched(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    const float *pVect1 = (const float *) pVect1v;
    const float *pVect2 = (const float *) pVect2v;
    
    auto& manager = CudaBatchManager::getInstance();
    manager.init(qty);
    
    float result = manager.computeDistance(pVect1, pVect2, qty);
    
    // Periodically clean up
    static thread_local size_t call_count = 0;
    if (++call_count % 100000 == 0) {
        manager.cleanup();
    }
    
    return result;
}

class L2SpaceCUDABatched : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    
public:
    L2SpaceCUDABatched(size_t dim) {
        fstdistfunc_ = L2SqrCUDABatched;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
        std::cout << "L2SpaceCUDABatched initialized for dimension " << dim << std::endl;
    }
    
    size_t get_data_size() {
        return data_size_;
    }
    
    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }
    
    void *get_dist_func_param() {
        return &dim_;
    }
    
    ~L2SpaceCUDABatched() {}
};

} // namespace hnswlib