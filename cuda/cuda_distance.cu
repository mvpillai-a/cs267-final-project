// cuda/cuda_distance.cu
#include "cuda_distance.cuh"
#include <stdio.h>

#define TILE_SIZE 16

__global__ void l2_distance_kernel_optimized(
    const float* vectors,
    const float* query,
    float* distances,
    int num_vectors,
    int dim)
{
    extern __shared__ float shared_query[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load query into shared memory cooperatively
    for (int i = local_tid; i < dim; i += blockDim.x) {
        shared_query[i] = query[i];
    }
    __syncthreads();
    
    if (tid < num_vectors) {
        float sum = 0.0f;
        const float* vector = vectors + tid * dim;
        
        // Use shared memory for query access
        for (int i = 0; i < dim; i++) {
            float diff = vector[i] - shared_query[i];
            sum += diff * diff;
        }
        distances[tid] = sum;
    }
}

__global__ void l2_distance_kernel(
    const float* vectors,
    const float* query,
    float* distances,
    int num_vectors,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_vectors) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = vectors[idx * dim + i] - query[i];
            sum += diff * diff;
        }
        distances[idx] = sum;
    }
}

void compute_distances_cuda(
    const float* vectors,
    const float* query,
    float* distances,
    int num_vectors,
    int dim)
{
    float *d_vectors, *d_query, *d_distances;
    
    cudaMalloc(&d_vectors, num_vectors * dim * sizeof(float));
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_distances, num_vectors * sizeof(float));
    
    cudaMemcpy(d_vectors, vectors, num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (num_vectors + block_size - 1) / block_size;
    int shared_mem_size = dim * sizeof(float);
    
    // Use optimized kernel with shared memory
    l2_distance_kernel_optimized<<<num_blocks, block_size, shared_mem_size>>>(
        d_vectors, d_query, d_distances, num_vectors, dim);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(distances, d_distances, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_vectors);
    cudaFree(d_query);
    cudaFree(d_distances);
}