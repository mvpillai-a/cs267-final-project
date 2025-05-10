#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Error checking macro
#define CUDA_CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

namespace parlayANN {

    

/**
 * CUDA kernel for computing Euclidean distances between a query point and multiple dataset points
 * 
 * @param query Pointer to query point coordinates
 * @param points Pointer to dataset points coordinates (all points in contiguous memory)
 * @param distances Output array for computed distances
 * @param candidates Array of indices of candidate points to compute distances for
 * @param numCandidates Number of candidates to process
 * @param dims Dimensionality of the points
 */
template<typename T>
__global__ void computeEuclideanDistancesKernel(
    const T* query,            // Query point
    const T* points,           // Dataset points 
    float* distances,          // Output distances
    const int* candidates,     // Candidate indices to check
    int numCandidates,         // Number of candidates
    int dims                   // Dimensionality
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numCandidates) {
        int pointIdx = candidates[tid];
        const T* point = points + pointIdx * dims;
        
        // Compute Euclidean distance
        float dist = 0.0f;
        for (int i = 0; i < dims; i++) {
            float diff = static_cast<float>(query[i]) - static_cast<float>(point[i]);
            dist += diff * diff;
        }
        
        distances[tid] = dist;
    }
}

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
) {
    // Allocate device memory
    T* d_query;
    T* d_points;
    int* d_candidates;
    float* d_distances;
    
    // Determine how many points we need to transfer
    int maxPointIdx = 0;
    for (int i = 0; i < numCandidates; i++) {
        if (candidates[i] > maxPointIdx) {
            maxPointIdx = candidates[i];
        }
    }
    
    // Calculate sizes
    size_t querySize = dims * sizeof(T);
    size_t pointsSize = (maxPointIdx + 1) * dims * sizeof(T);
    
    // Allocate and copy query
    CUDA_CHECK(cudaMalloc((void**)&d_query, querySize));
    CUDA_CHECK(cudaMemcpy(d_query, query, querySize, cudaMemcpyHostToDevice));
    
    // Allocate and copy points
    CUDA_CHECK(cudaMalloc((void**)&d_points, pointsSize));
    CUDA_CHECK(cudaMemcpy(d_points, points, pointsSize, cudaMemcpyHostToDevice));
    
    // Allocate and copy candidates
    CUDA_CHECK(cudaMalloc((void**)&d_candidates, numCandidates * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_candidates, candidates, numCandidates * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate distances
    CUDA_CHECK(cudaMalloc((void**)&d_distances, numCandidates * sizeof(float)));
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (numCandidates + blockSize - 1) / blockSize;
    computeEuclideanDistancesKernel<<<gridSize, blockSize>>>(
        d_query, d_points, d_distances, d_candidates, numCandidates, dims);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(distances, d_distances, numCandidates * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_distances));
}

// Helper function to check CUDA availability and print device info
bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// Print CUDA device info
void printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA devices:\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
    }
}

// Explicit template instantiations for common types
template void computeEuclideanDistances<float>(
    const float* query,
    const float* points,
    const int* candidates,
    int numCandidates,
    int dims,
    float* distances
);

template void computeEuclideanDistances<double>(
    const double* query,
    const double* points,
    const int* candidates,
    int numCandidates,
    int dims,
    float* distances
);

template void computeEuclideanDistances<unsigned char>(
    const unsigned char* query,
    const unsigned char* points,
    const int* candidates,
    int numCandidates,
    int dims,
    float* distances
);

template void computeEuclideanDistances<signed char>(
    const signed char* query,
    const signed char* points,
    const int* candidates,
    int numCandidates,
    int dims,
    float* distances
);

template void computeEuclideanDistances<unsigned short>(
    const unsigned short* query,
    const unsigned short* points,
    const int* candidates,
    int numCandidates,
    int dims,
    float* distances
);


} // namespace parlayANN