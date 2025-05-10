#pragma once
#include <cuda_runtime.h>

// Simple L2 distance calculation on GPU
__global__ void l2_distance_kernel(
    const float* vectors,    // All vectors in dataset
    const float* query,      // Query vector
    float* distances,        // Output distances
    int num_vectors,         // Number of vectors
    int dim                  // Dimension of each vector
);

// Wrapper function to call from CPU
void compute_distances_cuda(
    const float* vectors,
    const float* query,
    float* distances,
    int num_vectors,
    int dim
);