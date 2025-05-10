// cuda/cuda_l2_space_simple.h
#pragma once
#include "space_l2.h"
#include "cuda_distance.cuh"
#include "cuda_hnsw_kernels.cuh"  // Include the kernel wrapper
#include <cuda_runtime.h>
#include <iostream>
#include <atomic>

namespace hnswlib {

// Thread-local CUDA resources
struct CudaResources {
    float* d_vector1 = nullptr;
    float* d_vector2 = nullptr;
    float* d_result = nullptr;
    size_t allocated_dim = 0;
    bool initialized = false;
    
    void init(size_t dim) {
        if (!initialized || allocated_dim != dim) {
            if (initialized) {
                cudaFree(d_vector1);
                cudaFree(d_vector2);
                cudaFree(d_result);
            }
            
            cudaMalloc(&d_vector1, dim * sizeof(float));
            cudaMalloc(&d_vector2, dim * sizeof(float));
            cudaMalloc(&d_result, sizeof(float));
            
            allocated_dim = dim;
            initialized = true;
        }
    }
    
    ~CudaResources() {
        if (initialized) {
            cudaFree(d_vector1);
            cudaFree(d_vector2);
            cudaFree(d_result);
        }
    }
};

// Global counter for statistics
static std::atomic<size_t> cuda_distance_calls{0};

// CUDA-accelerated L2 distance function (simple version)
float L2SqrCUDASimple(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    static thread_local CudaResources cuda_res;
    
    size_t qty = *((size_t *) qty_ptr);
    const float *pVect1 = (const float *) pVect1v;
    const float *pVect2 = (const float *) pVect2v;
    
    cuda_res.init(qty);
    
    // Copy vectors to GPU
    cudaMemcpy(cuda_res.d_vector1, pVect1, qty * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_res.d_vector2, pVect2, qty * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use the wrapper function to launch kernel
    launch_l2_distance_kernel(cuda_res.d_vector2, cuda_res.d_vector1, cuda_res.d_result, 1, qty);
    
    // Get result
    float result;
    cudaMemcpy(&result, cuda_res.d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    cuda_distance_calls++;
    if (cuda_distance_calls % 1000000 == 0) {
        std::cout << "CUDA distance calculations: " << cuda_distance_calls << std::endl;
    }
    
    return result;
}

class L2SpaceCUDASimple : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    
public:
    L2SpaceCUDASimple(size_t dim) {
        fstdistfunc_ = L2SqrCUDASimple;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
        std::cout << "L2SpaceCUDASimple initialized for dimension " << dim << std::endl;
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
    
    ~L2SpaceCUDASimple() {
        std::cout << "Total CUDA distance calculations: " << cuda_distance_calls << std::endl;
    }
};

} // namespace hnswlib