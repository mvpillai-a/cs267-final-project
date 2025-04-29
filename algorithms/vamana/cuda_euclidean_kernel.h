#pragma once

#include <cuda_runtime.h>

namespace parlayANN {

// Check CUDA availability
bool isCudaAvailable();

// Print CUDA device information
void printCudaDeviceInfo();

/**
 * Host wrapper function for the Euclidean distance kernel
 * 
 * @param query Pointer to query point coordinates on host
 * @param points Pointer to dataset points coordinates on host
 * @param candidates Array of indices of candidate points to compute distances for
 * @param numCandidates Number of candidates to process
 * @param dims Dimensionality of the points
 * @param distances Output array for computed distances
 */
template<typename T>
void computeEuclideanDistances(
    const T* query,
    const T* points,
    const int* candidates,
    int numCandidates,
    int dims,
    float* distances
);

} // namespace parlayANN