// nnd_cuda.cuh
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cfloat>

#include "nnd.h"
#include "distances.h"
#include "dtypes.h"
#include "utils.h"

namespace nndescent_cuda {

// GPU kernel for batch distance calculation
__global__ void euclidean_distance_kernel(
    const float* d_data,
    const int* d_idx_pairs,
    float* d_distances,
    int num_pairs,
    int dim
);

// GPU kernel for initializing random neighbors
__global__ void init_random_kernel(
    int* d_indices,
    float* d_keys,
    int num_points,
    int n_neighbors,
    unsigned int seed
);

// GPU kernel for updating the graph
__global__ void update_graph_kernel(
    int* d_indices,
    float* d_keys,
    const int* d_pairs,
    const float* d_distances,
    int num_pairs,
    int num_points,
    int n_neighbors
);

// Main CUDA NN-Descent function
void nn_descent_cuda(
    const nndescent::Matrix<float>& data,
    nndescent::HeapList<float>& current_graph,
    int n_neighbors,
    nndescent::RandomState& rng_state,
    int max_candidates,
    int n_iters,
    float delta,
    int n_threads,
    bool verbose,
    const std::string& metric
);

} // namespace nndescent_cuda