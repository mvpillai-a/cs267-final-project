#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include "cuda_euclidean_kernel.h"

namespace {
// Simple function to compute Euclidean distance on CPU for validation
template<typename T>
float computeEuclideanCPU(const T* a, const T* b, int dims) {
    float dist = 0.0f;
    for (int i = 0; i < dims; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}
}

// Test function to verify CUDA Euclidean distance calculation
void testEuclideanDistance() {
    const int numPoints = 1000;
    const int dims = 128;
    const int numCandidates = 500;
    
    // Allocate and initialize random data
    std::vector<float> query(dims);
    std::vector<float> points(numPoints * dims);
    std::vector<int> candidates(numCandidates);
    std::vector<float> distances_gpu(numCandidates);
    std::vector<float> distances_cpu(numCandidates);
    
    // Initialize random number generator
    srand(time(NULL));
    
    // Generate random query point
    for (int i = 0; i < dims; i++) {
        query[i] = (float)rand() / RAND_MAX;
    }
    
    // Generate random dataset points
    for (int i = 0; i < numPoints * dims; i++) {
        points[i] = (float)rand() / RAND_MAX;
    }
    
    // Generate random candidate indices
    for (int i = 0; i < numCandidates; i++) {
        candidates[i] = rand() % numPoints;
    }
    
    // Check if CUDA is available
    if (!parlayANN::isCudaAvailable()) {
        printf("CUDA is not available on this system\n");
        return;
    }
    
    // Print CUDA device info
    parlayANN::printCudaDeviceInfo();
    
    // Compute distances using CUDA
    printf("Computing distances using CUDA...\n");
    parlayANN::computeEuclideanDistances<float>(
        query.data(),
        points.data(),
        candidates.data(),
        numCandidates,
        dims,
        distances_gpu.data()
    );
    
    // Compute distances on CPU for validation
    printf("Computing distances on CPU for validation...\n");
    for (int i = 0; i < numCandidates; i++) {
        int idx = candidates[i];
        distances_cpu[i] = computeEuclideanCPU<float>(query.data(), 
                                                      &points[idx * dims], 
                                                      dims);
    }
    
    // Compare results
    printf("Comparing results...\n");
    float maxDiff = 0.0f;
    float sumDiff = 0.0f;
    for (int i = 0; i < numCandidates; i++) {
        float diff = fabs(distances_gpu[i] - distances_cpu[i]);
        maxDiff = fmax(maxDiff, diff);
        sumDiff += diff;
    }
    
    printf("Max difference: %f\n", maxDiff);
    printf("Average difference: %f\n", sumDiff / numCandidates);
    
    if (maxDiff < 1e-5) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED: Differences exceed threshold\n");
    }
}

// Make sure to explicitly define the main function
extern "C" int main() {
    testEuclideanDistance();
    return 0;
}