// cuda/cuda_hnsw_kernels.cuh
#pragma once

void launch_l2_distance_kernel(
    const float* d_vectors,
    const float* d_query,
    float* d_distances,
    int num_vectors,
    int dim);