// cuda/cuda_l2_space.h
#pragma once
#include "space_l2.h"
#include "cuda_distance.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <deque>

namespace hnswlib {

// Global CUDA memory manager and batch processor
class CudaBatchProcessor {
public:
    static CudaBatchProcessor& getInstance() {
        static CudaBatchProcessor instance;
        return instance;
    }
    
    struct BatchData {
        const float* query;
        std::vector<const float*> vectors;
        std::vector<float> results;
        bool processed = false;
    };
    
    // CUDA memory
    float* d_vectors = nullptr;
    float* d_query = nullptr;
    float* d_distances = nullptr;
    size_t max_batch_size = 256;  // Optimal batch size for GPU
    size_t allocated_dim = 0;
    bool initialized = false;
    size_t cuda_calls = 0;
    size_t total_distance_calcs = 0;
    
    // Batch management
    std::unordered_map<const float*, BatchData> batches;
    
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
    
    void processBatch(const float* query, BatchData& batch, size_t dim) {
        if (batch.processed || batch.vectors.empty()) return;
        
        size_t batch_size = batch.vectors.size();
        if (batch_size > max_batch_size) {
            // Process in chunks
            for (size_t i = 0; i < batch_size; i += max_batch_size) {
                size_t chunk_size = std::min(max_batch_size, batch_size - i);
                processBatchChunk(query, batch, i, chunk_size, dim);
            }
        } else {
            processBatchChunk(query, batch, 0, batch_size, dim);
        }
        
        batch.processed = true;
        cuda_calls++;
        total_distance_calcs += batch_size;
    }
    
    void processBatchChunk(const float* query, BatchData& batch, size_t start_idx, size_t chunk_size, size_t dim) {
        // Prepare data for GPU
        std::vector<float> vectors_data;
        vectors_data.reserve(chunk_size * dim);
        
        for (size_t i = start_idx; i < start_idx + chunk_size; i++) {
            const float* vec = batch.vectors[i];
            vectors_data.insert(vectors_data.end(), vec, vec + dim);
        }
        
        // Copy to GPU
        cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vectors, vectors_data.data(), chunk_size * dim * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int num_blocks = (chunk_size + block_size - 1) / block_size;
        l2_distance_kernel<<<num_blocks, block_size>>>(d_vectors, d_query, d_distances, chunk_size, dim);
        
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(&batch.results[start_idx], d_distances, chunk_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    float addToBatch(const float* query, const float* vector, size_t dim) {
        auto& batch = batches[query];
        
        if (batch.query != query) {
            // New batch
            batch.query = query;
            batch.vectors.clear();
            batch.results.clear();
            batch.processed = false;
        }
        
        size_t idx = batch.vectors.size();
        batch.vectors.push_back(vector);
        batch.results.resize(batch.vectors.size());
        
        // Process batch when it reaches optimal size
        if (batch.vectors.size() >= max_batch_size && !batch.processed) {
            processBatch(query, batch, dim);
        }
        
        // If batch is processed, return result; otherwise return placeholder
        if (batch.processed) {
            return batch.results[idx];
        } else {
            return -1.0f;  // Placeholder
        }
    }
    
    void flushBatch(const float* query, size_t dim) {
        auto it = batches.find(query);
        if (it != batches.end() && !it->second.processed) {
            processBatch(query, it->second, dim);
        }
    }
    
    float getResult(const float* query, const float* vector) {
        auto it = batches.find(query);
        if (it != batches.end()) {
            for (size_t i = 0; i < it->second.vectors.size(); i++) {
                if (it->second.vectors[i] == vector) {
                    return it->second.results[i];
                }
            }
        }
        return -1.0f;
    }
    
    void clearOldBatches() {
        // Keep only recent batches to avoid memory growth
        if (batches.size() > 1000) {
            batches.clear();
        }
    }
    
    ~CudaBatchProcessor() {
        if (initialized) {
            std::cout << "CUDA batch calls: " << cuda_calls 
                      << ", Total distance calculations: " << total_distance_calcs 
                      << ", Efficiency: " << (cuda_calls > 0 ? total_distance_calcs / cuda_calls : 0) 
                      << " distances per CUDA call" << std::endl;
            cudaFree(d_vectors);
            cudaFree(d_query);
            cudaFree(d_distances);
        }
    }
};

// CUDA-accelerated L2 distance function
float L2SqrCUDA(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    const float *pVect1 = (const float *) pVect1v;
    const float *pVect2 = (const float *) pVect2v;
    
    auto& processor = CudaBatchProcessor::getInstance();
    processor.init(qty);
    
    // Add to batch
    float result = processor.addToBatch(pVect1, pVect2, qty);
    
    if (result < 0) {
        // Not yet processed, compute CPU result as fallback
        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = pVect1[i] - pVect2[i];
            res += t * t;
        }
        return res;
    }
    
    return result;
}

// Function to force batch processing (called periodically)
void flushCudaBatches(const void* query_data, size_t dim) {
    auto& processor = CudaBatchProcessor::getInstance();
    processor.flushBatch((const float*)query_data, dim);
}

class L2SpaceCUDA : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    
public:
    L2SpaceCUDA(size_t dim) {
        fstdistfunc_ = L2SqrCUDA;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
        std::cout << "L2SpaceCUDA initialized for dimension " << dim << std::endl;
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
    
    ~L2SpaceCUDA() {}
};

} // namespace hnswlib