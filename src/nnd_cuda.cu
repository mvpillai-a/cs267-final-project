// nnd_cuda.cu - Fixed version with proper recall
#include "nnd_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace nndescent_cuda {

// Constants
const char OLD = '0';
const char NEW = '1';
const int NONE = -1;

// Squared Euclidean distance on GPU
__device__ float squared_euclidean_cuda(
    const float* data,
    int idx0,
    int idx1,
    int dim
) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = data[idx0 * dim + i] - data[idx1 * dim + i];
        result += diff * diff;
    }
    return result;
}

// Simple but correct heap push operation
__device__ int device_checked_push(
    int* indices,
    float* keys,
    char* flags,
    int heap_start,
    int heap_size,
    int idx,
    float key,
    char flag
) {
    if (key >= keys[heap_start]) {
        return 0;
    }
    
    // Check if already in heap
    for (int i = 0; i < heap_size; ++i) {
        if (indices[heap_start + i] == idx) {
            return 0;
        }
    }
    
    // Simple siftdown
    int current = 0;
    while (true) {
        int left_child = 2 * current + 1;
        int right_child = left_child + 1;
        int swap = -1;
        
        if (left_child >= heap_size) {
            break;
        } else if (right_child >= heap_size) {
            if (keys[heap_start + left_child] > key) {
                swap = left_child;
            }
        } else {
            if (keys[heap_start + left_child] >= keys[heap_start + right_child]) {
                if (keys[heap_start + left_child] > key) {
                    swap = left_child;
                }
            } else {
                if (keys[heap_start + right_child] > key) {
                    swap = right_child;
                }
            }
        }
        
        if (swap == -1) break;
        
        indices[heap_start + current] = indices[heap_start + swap];
        keys[heap_start + current] = keys[heap_start + swap];
        flags[heap_start + current] = flags[heap_start + swap];
        current = swap;
    }
    
    indices[heap_start + current] = idx;
    keys[heap_start + current] = key;
    flags[heap_start + current] = flag;
    return 1;
}

// Initialize with random neighbors
__global__ void init_random_kernel(
    int* indices,
    float* keys,
    char* flags,
    const float* data,
    int num_points,
    int n_neighbors,
    int dim,
    unsigned int seed
) {
    int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx0 < num_points) {
        curandState state;
        curand_init(seed, idx0, 0, &state);
        
        int heap_start = idx0 * n_neighbors;
        
        // Initialize heap
        for (int j = 0; j < n_neighbors; ++j) {
            indices[heap_start + j] = NONE;
            keys[heap_start + j] = FLT_MAX;
            flags[heap_start + j] = NEW;
        }
        
        // Add self
        device_checked_push(indices, keys, flags, heap_start, n_neighbors, idx0, 0.0f, NEW);
        
        // Add random neighbors
        for (int j = 0; j < n_neighbors * 2; ++j) {
            int idx1 = curand(&state) % num_points;
            if (idx1 != idx0) {
                float d = squared_euclidean_cuda(data, idx0, idx1, dim);
                device_checked_push(indices, keys, flags, heap_start, n_neighbors, idx1, d, NEW);
            }
        }
    }
}

// Sample candidates
__global__ void sample_candidates_kernel(
    const int* indices,
    const char* flags,
    int* new_candidates,
    int* old_candidates,
    int* new_counts,
    int* old_counts,
    int num_points,
    int n_neighbors,
    int max_candidates,
    unsigned int seed
) {
    int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx0 < num_points) {
        curandState state;
        curand_init(seed, idx0, 0, &state);
        
        int new_count = 0;
        int old_count = 0;
        
        for (int j = 0; j < n_neighbors; ++j) {
            int idx1 = indices[idx0 * n_neighbors + j];
            if (idx1 == NONE || idx1 < 0) continue;
            
            char flag = flags[idx0 * n_neighbors + j];
            
            if (flag == NEW && new_count < max_candidates) {
                new_candidates[idx0 * max_candidates + new_count] = idx1;
                new_count++;
            } else if (flag == OLD && old_count < max_candidates) {
                float rand_val = curand_uniform(&state);
                if (rand_val < 0.5f) {
                    old_candidates[idx0 * max_candidates + old_count] = idx1;
                    old_count++;
                }
            }
        }
        
        new_counts[idx0] = new_count;
        old_counts[idx0] = old_count;
    }
}

// Collect reverse neighbors
__global__ void collect_reverse_neighbors_kernel(
    const int* indices,
    const char* flags,
    int* reverse_new,
    int* reverse_old,
    int* reverse_new_counts,
    int* reverse_old_counts,
    int num_points,
    int n_neighbors,
    int max_candidates
) {
    int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx0 < num_points) {
        for (int j = 0; j < n_neighbors; ++j) {
            int idx1 = indices[idx0 * n_neighbors + j];
            if (idx1 == NONE || idx1 < 0 || idx1 >= num_points) continue;
            
            char flag = flags[idx0 * n_neighbors + j];
            
            if (flag == NEW) {
                int pos = atomicAdd(&reverse_new_counts[idx1], 1);
                if (pos < max_candidates) {
                    reverse_new[idx1 * max_candidates + pos] = idx0;
                } else {
                    atomicSub(&reverse_new_counts[idx1], 1);
                }
            }
        }
    }
}

// Generate updates - back to original logic
__global__ void generate_updates_kernel(
    const float* data,
    const int* indices,
    const float* keys,
    const int* new_candidates,
    const int* old_candidates,
    const int* reverse_new,
    const int* new_counts,
    const int* old_counts,
    const int* reverse_new_counts,
    int* updates,
    float* update_distances,
    int* update_count,
    int num_points,
    int n_neighbors,
    int max_candidates,
    int dim,
    int max_updates_per_point
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_points) {
        int count = 0;
        int update_offset = i * max_updates_per_point;
        
        int new_count = new_counts[i];
        int old_count = old_counts[i];
        
        // New-new pairs
        for (int j = 0; j < new_count; ++j) {
            int idx0 = new_candidates[i * max_candidates + j];
            if (idx0 == NONE) continue;
            
            for (int k = j + 1; k < new_count; ++k) {
                int idx1 = new_candidates[i * max_candidates + k];
                if (idx1 == NONE) continue;
                
                float d = squared_euclidean_cuda(data, idx0, idx1, dim);
                
                bool improve0 = (idx0 < num_points && d < keys[idx0 * n_neighbors]);
                bool improve1 = (idx1 < num_points && d < keys[idx1 * n_neighbors]);
                
                if ((improve0 || improve1) && count < max_updates_per_point) {
                    updates[update_offset + count * 2] = idx0;
                    updates[update_offset + count * 2 + 1] = idx1;
                    update_distances[update_offset + count] = d;
                    count++;
                }
            }
        }
        
        // New-old pairs
        for (int j = 0; j < new_count; ++j) {
            int idx0 = new_candidates[i * max_candidates + j];
            if (idx0 == NONE) continue;
            
            for (int k = 0; k < old_count; ++k) {
                int idx1 = old_candidates[i * max_candidates + k];
                if (idx1 == NONE) continue;
                
                float d = squared_euclidean_cuda(data, idx0, idx1, dim);
                
                bool improve0 = (idx0 < num_points && d < keys[idx0 * n_neighbors]);
                bool improve1 = (idx1 < num_points && d < keys[idx1 * n_neighbors]);
                
                if ((improve0 || improve1) && count < max_updates_per_point) {
                    updates[update_offset + count * 2] = idx0;
                    updates[update_offset + count * 2 + 1] = idx1;
                    update_distances[update_offset + count] = d;
                    count++;
                }
            }
        }
        
        // New reverse pairs
        int reverse_count = reverse_new_counts[i];
        for (int j = 0; j < reverse_count; ++j) {
            int idx0 = reverse_new[i * max_candidates + j];
            if (idx0 == NONE) continue;
            
            for (int k = j + 1; k < reverse_count; ++k) {
                int idx1 = reverse_new[i * max_candidates + k];
                if (idx1 == NONE) continue;
                
                float d = squared_euclidean_cuda(data, idx0, idx1, dim);
                
                bool improve0 = (idx0 < num_points && d < keys[idx0 * n_neighbors]);
                bool improve1 = (idx1 < num_points && d < keys[idx1 * n_neighbors]);
                
                if ((improve0 || improve1) && count < max_updates_per_point) {
                    updates[update_offset + count * 2] = idx0;
                    updates[update_offset + count * 2 + 1] = idx1;
                    update_distances[update_offset + count] = d;
                    count++;
                }
            }
        }
        
        update_count[i] = count;
    }
}

// Apply updates and mark as OLD
__global__ void apply_updates_kernel(
    int* indices,
    float* keys,
    char* flags,
    const int* updates,
    const float* update_distances,
    const int* update_counts,
    const int* new_candidates,
    const int* new_counts,
    int num_points,
    int n_neighbors,
    int max_candidates,
    int max_updates_per_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        // Apply updates
        for (int i = 0; i < num_points; ++i) {
            int update_offset = i * max_updates_per_point;
            int count = update_counts[i];
            
            for (int j = 0; j < count; ++j) {
                int idx0 = updates[update_offset + j * 2];
                int idx1 = updates[update_offset + j * 2 + 1];
                float d = update_distances[update_offset + j];
                
                if (idx0 == idx) {
                    int heap_start = idx0 * n_neighbors;
                    device_checked_push(indices, keys, flags, heap_start, n_neighbors, idx1, d, NEW);
                }
                if (idx1 == idx) {
                    int heap_start = idx1 * n_neighbors;
                    device_checked_push(indices, keys, flags, heap_start, n_neighbors, idx0, d, NEW);
                }
            }
        }
    }
}

// Mark sampled as OLD - separate kernel
__global__ void mark_old_kernel(
    char* flags,
    const int* indices,
    const int* new_candidates,
    const int* new_counts,
    int num_points,
    int n_neighbors,
    int max_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        int heap_start = idx * n_neighbors;
        int new_count = new_counts[idx];
        
        for (int j = 0; j < n_neighbors; ++j) {
            int idx1 = indices[heap_start + j];
            if (idx1 == NONE) continue;
            
            for (int k = 0; k < new_count; ++k) {
                if (new_candidates[idx * max_candidates + k] == idx1) {
                    flags[heap_start + j] = OLD;
                    break;
                }
            }
        }
    }
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
    
    // Allocate device memory
    float* d_data;
    int* d_indices;
    float* d_keys;
    char* d_flags;
    int* d_new_candidates;
    int* d_old_candidates;
    int* d_reverse_new;
    int* d_new_counts;
    int* d_old_counts;
    int* d_reverse_new_counts;
    int* d_updates;
    float* d_update_distances;
    int* d_update_counts;
    
    cudaMalloc(&d_data, num_points * dim * sizeof(float));
    cudaMalloc(&d_indices, num_points * n_neighbors * sizeof(int));
    cudaMalloc(&d_keys, num_points * n_neighbors * sizeof(float));
    cudaMalloc(&d_flags, num_points * n_neighbors * sizeof(char));
    cudaMalloc(&d_new_candidates, num_points * max_candidates * sizeof(int));
    cudaMalloc(&d_old_candidates, num_points * max_candidates * sizeof(int));
    cudaMalloc(&d_reverse_new, num_points * max_candidates * sizeof(int));
    cudaMalloc(&d_new_counts, num_points * sizeof(int));
    cudaMalloc(&d_old_counts, num_points * sizeof(int));
    cudaMalloc(&d_reverse_new_counts, num_points * sizeof(int));
    
    int max_updates_per_point = max_candidates * max_candidates * 2;
    cudaMalloc(&d_updates, num_points * max_updates_per_point * 2 * sizeof(int));
    cudaMalloc(&d_update_distances, num_points * max_updates_per_point * sizeof(float));
    cudaMalloc(&d_update_counts, num_points * sizeof(int));
    
    // Initialize arrays
    cudaMemcpy(d_data, data.m_ptr, num_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch configuration
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
    
    if (verbose) {
        std::cout << "CUDA NN descent for " << n_iters << " iterations" << std::endl;
    }
    
    // Initialize with random neighbors
    init_random_kernel<<<grid_size, block_size>>>(
        d_indices, d_keys, d_flags, d_data, num_points, n_neighbors, dim, rng_state[0]
    );
    cudaDeviceSynchronize();
    
    // Main NN-Descent loop
    for (int iter = 0; iter < n_iters; ++iter) {
        if (verbose) {
            std::cout << "\t" << iter + 1 << "  /  " << n_iters << std::endl;
        }
        
        // Clear counts
        cudaMemset(d_new_counts, 0, num_points * sizeof(int));
        cudaMemset(d_old_counts, 0, num_points * sizeof(int));
        cudaMemset(d_reverse_new_counts, 0, num_points * sizeof(int));
        cudaMemset(d_update_counts, 0, num_points * sizeof(int));
        
        // Sample candidates
        sample_candidates_kernel<<<grid_size, block_size>>>(
            d_indices, d_flags, d_new_candidates, d_old_candidates,
            d_new_counts, d_old_counts, num_points, n_neighbors, max_candidates,
            rng_state[0] + iter
        );
        cudaDeviceSynchronize();
        
        // Collect reverse neighbors
        collect_reverse_neighbors_kernel<<<grid_size, block_size>>>(
            d_indices, d_flags, d_reverse_new, nullptr,
            d_reverse_new_counts, nullptr,
            num_points, n_neighbors, max_candidates
        );
        cudaDeviceSynchronize();
        
        // Generate updates
        generate_updates_kernel<<<grid_size, block_size>>>(
            d_data, d_indices, d_keys, d_new_candidates, d_old_candidates,
            d_reverse_new, d_new_counts, d_old_counts, d_reverse_new_counts,
            d_updates, d_update_distances, d_update_counts,
            num_points, n_neighbors, max_candidates, dim, max_updates_per_point
        );
        cudaDeviceSynchronize();
        
        // Apply updates
        apply_updates_kernel<<<grid_size, block_size>>>(
            d_indices, d_keys, d_flags, d_updates, d_update_distances, d_update_counts,
            d_new_candidates, d_new_counts, num_points, n_neighbors, max_candidates, max_updates_per_point
        );
        cudaDeviceSynchronize();
        
        // Mark as OLD
        mark_old_kernel<<<grid_size, block_size>>>(
            d_flags, d_indices, d_new_candidates, d_new_counts,
            num_points, n_neighbors, max_candidates
        );
        cudaDeviceSynchronize();
        
        // Check convergence
        if (verbose) {
            int h_update_counts[num_points];
            cudaMemcpy(h_update_counts, d_update_counts, num_points * sizeof(int), cudaMemcpyDeviceToHost);
            
            int total_updates = 0;
            for (int i = 0; i < num_points; ++i) {
                total_updates += h_update_counts[i];
            }
            
            std::cout << "\t\t" << total_updates << " updates generated" << std::endl;
            
            if (total_updates < delta * num_points * n_neighbors) {
                std::cout << "Stopping threshold met -- exiting after " 
                         << iter + 1 << " iterations" << std::endl;
                break;
            }
        }
    }
    
    // Copy results back to host
    std::vector<int> h_indices(num_points * n_neighbors);
    std::vector<float> h_keys(num_points * n_neighbors);
    
    cudaMemcpy(h_indices.data(), d_indices, num_points * n_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys.data(), d_keys, num_points * n_neighbors * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Copy to HeapList with distance correction
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < n_neighbors; ++j) {
            current_graph.indices(i, j) = h_indices[i * n_neighbors + j];
            float key = h_keys[i * n_neighbors + j];
            if (key < FLT_MAX) {
                current_graph.keys(i, j) = std::sqrt(key);
            } else {
                current_graph.keys(i, j) = key;
            }
        }
    }
    
    // Sort the results
    current_graph.heapsort();
    
    // Clean up
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_keys);
    cudaFree(d_flags);
    cudaFree(d_new_candidates);
    cudaFree(d_old_candidates);
    cudaFree(d_reverse_new);
    cudaFree(d_new_counts);
    cudaFree(d_old_counts);
    cudaFree(d_reverse_new_counts);
    cudaFree(d_updates);
    cudaFree(d_update_distances);
    cudaFree(d_update_counts);
}

} // namespace nndescent_cuda