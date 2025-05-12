#include "nnd_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>

namespace nndescent_cuda {

// Constants
const int NONE = -1;
const char NEW = '1';
const char OLD = '0';

// Squared Euclidean distance kernel
__global__ void euclidean_distance_kernel(
    const float* data,
    const int* point_pairs,
    float* distances,
    int num_pairs,
    int dim
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx < num_pairs) {
        int idx0 = point_pairs[pair_idx * 2];
        int idx1 = point_pairs[pair_idx * 2 + 1];
        
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = data[idx0 * dim + i] - data[idx1 * dim + i];
            dist += diff * diff;
        }
        
        distances[pair_idx] = dist;
    }
}

// Function to calculate distances in batches on GPU
void calculate_distances_cuda(
    const float* h_data,
    const std::vector<std::pair<int, int>>& point_pairs,
    std::vector<float>& distances,
    int dim,
    int num_points
) {
    // Allocate device memory
    float* d_data;
    int* d_point_pairs;
    float* d_distances;
    
    int num_pairs = point_pairs.size();
    distances.resize(num_pairs);
    
    // Prepare point pairs in flat array format
    std::vector<int> flat_pairs(num_pairs * 2);
    for (int i = 0; i < num_pairs; ++i) {
        flat_pairs[i * 2] = point_pairs[i].first;
        flat_pairs[i * 2 + 1] = point_pairs[i].second;
    }
    
    // Allocate and copy data to device
    cudaMalloc(&d_data, num_points * dim * sizeof(float));
    cudaMalloc(&d_point_pairs, num_pairs * 2 * sizeof(int));
    cudaMalloc(&d_distances, num_pairs * sizeof(float));
    
    cudaMemcpy(d_data, h_data, num_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_point_pairs, flat_pairs.data(), num_pairs * 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_pairs + block_size - 1) / block_size;
    
    euclidean_distance_kernel<<<grid_size, block_size>>>(
        d_data, d_point_pairs, d_distances, num_pairs, dim
    );
    
    // Copy results back
    cudaMemcpy(distances.data(), d_distances, num_pairs * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_point_pairs);
    cudaFree(d_distances);
}

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
) {
    int num_points = data.nrows();
    int dim = data.ncols();
    
    if (max_candidates == NONE) {
        max_candidates = std::min(60, n_neighbors);
    }
    // Initialize with random neighbors
    std::vector<std::pair<int, int>> all_point_pairs;
    std::vector<int> pair_to_point_map;  // Maps pair index to source point
    std::vector<int> pair_to_neighbor_map;  // Maps pair index to neighbor index

    // First collect all random pairs
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < n_neighbors * 2; ++j) {
            int idx1 = rand() % num_points;
            if (idx1 != i) {
                all_point_pairs.push_back(std::make_pair(i, idx1));
                pair_to_point_map.push_back(i);
                pair_to_neighbor_map.push_back(j);
            }
        }
    }

    // Calculate all distances in one GPU call
    std::vector<float> all_distances;
    calculate_distances_cuda(data.m_ptr, all_point_pairs, all_distances, dim, num_points);

    // Update graph with calculated distances
    for (size_t i = 0; i < all_point_pairs.size(); ++i) {
        int source = pair_to_point_map[i];
        int target = all_point_pairs[i].second;
        float distance = all_distances[i];
        current_graph.checked_push(source, target, distance, NEW);
    }

    
    if (verbose) {
        std::cout << "CUDA NN descent for " << n_iters << " iterations" << std::endl;
    }
    
    // Main NN-Descent loop
    for (int iter = 0; iter < n_iters; ++iter) {
        if (verbose) {
            std::cout << "\t" << iter + 1 << "  /  " << n_iters << std::endl;
        }
        
        // Sample candidates (CPU side, same as in nnd.cpp)
        nndescent::HeapList<float> new_candidates(num_points, max_candidates, INT_MAX);
        nndescent::HeapList<float> old_candidates(num_points, max_candidates, INT_MAX);
        
        // Sample candidates
        for (int i = 0; i < num_points; ++i) {
            for (int j = 0; j < n_neighbors; ++j) {
                int idx1 = current_graph.indices(i, j);
                char flag = current_graph.flags(i, j);
                
                if (idx1 == NONE) continue;
                
                int priority = rand(); // Random priority for sampling
                
                if (flag == NEW) { // NEW
                    new_candidates.checked_push(i, idx1, priority);
                    new_candidates.checked_push(idx1, i, priority);
                } else { // OLD
                    if (rand() % 2 == 0) { // 50% sampling
                        old_candidates.checked_push(i, idx1, priority);
                        old_candidates.checked_push(idx1, i, priority);
                    }
                }
            }
        }
        
        // Mark sampled nodes as OLD
        for (int i = 0; i < num_points; ++i) {
            for (int j = 0; j < n_neighbors; ++j) {
                int idx1 = current_graph.indices(i, j);
                for (int k = 0; k < max_candidates; ++k) {
                    if (new_candidates.indices(i, k) == idx1) {
                        current_graph.flags(i, j) = OLD;
                        break;
                    }
                }
            }
        }
        
        // Generate point pairs for distance calculation
        std::vector<std::pair<int, int>> point_pairs;
        
        // New-new pairs
        for (int i = 0; i < num_points; ++i) {
            for (int j = 0; j < max_candidates; ++j) {
                int idx0 = new_candidates.indices(i, j);
                if (idx0 == NONE) continue;
                
                for (int k = j + 1; k < max_candidates; ++k) {
                    int idx1 = new_candidates.indices(i, k);
                    if (idx1 == NONE) continue;
                    
                    point_pairs.push_back(std::make_pair(idx0, idx1));
                }
            }
        }
        
        // New-old pairs
        for (int i = 0; i < num_points; ++i) {
            for (int j = 0; j < max_candidates; ++j) {
                int idx0 = new_candidates.indices(i, j);
                if (idx0 == NONE) continue;
                
                for (int k = 0; k < max_candidates; ++k) {
                    int idx1 = old_candidates.indices(i, k);
                    if (idx1 == NONE) continue;
                    
                    point_pairs.push_back(std::make_pair(idx0, idx1));
                }
            }
        }
        
        // Calculate distances using CUDA
        std::vector<float> distances;
        calculate_distances_cuda(data.m_ptr, point_pairs, distances, dim, num_points);
        
        // Apply updates (CPU side)
        int updates_applied = 0;
        
        for (size_t i = 0; i < point_pairs.size(); ++i) {
            int idx0 = point_pairs[i].first;
            int idx1 = point_pairs[i].second;
            float d = distances[i];
            
            // Only update if distance is better than current max
            if (d < current_graph.max(idx0) || d < current_graph.max(idx1)) {
                updates_applied += current_graph.checked_push(idx0, idx1, d, NEW);
                updates_applied += current_graph.checked_push(idx1, idx0, d, NEW);
            }
        }
        
        if (verbose) {
            std::cout << "\t\t" << updates_applied << " updates generated" << std::endl;
        }
        
        // Check for early termination
        if (updates_applied < delta * num_points * n_neighbors) {
            if (verbose) {
                std::cout << "Stopping threshold met -- exiting after " 
                         << iter + 1 << " iterations" << std::endl;
            }
            break;
        }
    }
    
    // Make sure every node has itself as a neighbor
    for (int i = 0; i < num_points; ++i) {
        current_graph.checked_push(i, i, 0.0f, NEW);
    }
    
    // Sort the results
    current_graph.heapsort();
    
    // Apply distance correction (sqrt for Euclidean)
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < n_neighbors; ++j) {
            float key = current_graph.keys(i, j);
            if (key < FLT_MAX) {
                current_graph.keys(i, j) = std::sqrt(key);
            }
        }
    }
}

} // namespace nndescent_cuda
