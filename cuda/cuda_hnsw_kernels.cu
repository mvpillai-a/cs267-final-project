// cuda/cuda_hnsw_kernels.cu
#include "cuda_hnsw_kernels.cuh"
#include "cuda_distance.cuh"

void launch_l2_distance_kernel(
    const float* d_vectors,
    const float* d_query,
    float* d_distances,
    int num_vectors,
    int dim)
{
    int block_size = 256;
    int num_blocks = (num_vectors + block_size - 1) / block_size;
    l2_distance_kernel<<<num_blocks, block_size>>>(d_vectors, d_query, d_distances, num_vectors, dim);
    cudaDeviceSynchronize();
}