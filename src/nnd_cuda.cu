/**
 * @file nnd_cuda.cu
 * @brief CUDA-accelerated implementation of Nearest Neighbor Descent algorithm
 */

 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <thrust/device_vector.h>
 #include <thrust/sort.h>
 #include <vector>
 #include <iostream>
 #include <chrono>
 #include <cfloat>
 #include "nnd.h"
 
 namespace nndescent_cuda
 {
 
 // CUDA error checking macro
 #define CUDA_CHECK(call) \
     do { \
         cudaError_t error = call; \
         if (error != cudaSuccess) { \
             fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                     __FILE__, __LINE__, error, cudaGetErrorString(error)); \
             exit(EXIT_FAILURE); \
         } \
     } while (0)
 
 // Constants
 constexpr int BLOCK_SIZE = 128;
 constexpr int MAX_NEIGHBORS = 64;
 constexpr int MAX_CANDIDATES = 128;
 
 // CUDA kernels for different distance metrics
 __device__ float squared_euclidean_distance(
     const float* x, const float* y, int dim)
 {
     float result = 0.0f;
     for (int i = 0; i < dim; ++i) {
         float diff = x[i] - y[i];
         result += diff * diff;
     }
     return result;
 }
 
 __device__ float cosine_distance(
     const float* x, const float* y, int dim)
 {
     float dot = 0.0f, norm_x = 0.0f, norm_y = 0.0f;
     for (int i = 0; i < dim; ++i) {
         dot += x[i] * y[i];
         norm_x += x[i] * x[i];
         norm_y += y[i] * y[i];
     }
     if (norm_x == 0.0f && norm_y == 0.0f) return 0.0f;
     if (norm_x == 0.0f || norm_y == 0.0f) return 1.0f;
     return 1.0f - (dot / (sqrtf(norm_x) * sqrtf(norm_y)));
 }
 
 // Structure for device-side neighbor data
 struct DeviceNeighbor {
     int idx;
     float dist;
     char flag;
 };
 
 // CUDA kernel for distance computation
 __global__ void compute_distances_kernel(
     const float* data,
     const int* query_indices,
     const int* target_indices,
     float* distances,
     int n_queries,
     int n_targets,
     int dim,
     bool euclidean)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (idx < n_queries * n_targets) {
         int query_idx = idx / n_targets;
         int target_idx = idx % n_targets;
         
         const float* query_point = data + query_indices[query_idx] * dim;
         const float* target_point = data + target_indices[target_idx] * dim;
         
         if (euclidean) {
             distances[idx] = squared_euclidean_distance(query_point, target_point, dim);
         } else {
             distances[idx] = cosine_distance(query_point, target_point, dim);
         }
     }
 }
 
 // CUDA kernel for graph updates with max-heap
 __global__ void update_graph_kernel(
     DeviceNeighbor* graph,
     const float* distances,
     const int* source_indices,
     const int* target_indices,
     int n_updates,
     int n_neighbors,
     int n_nodes)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (idx < n_updates) {
         int source = source_indices[idx];
         int target = target_indices[idx];
         float dist = distances[idx];
         
         DeviceNeighbor* neighbors = graph + source * n_neighbors;
         
         float max_dist = 0.0f;
         int max_idx = -1;
         for (int i = 0; i < n_neighbors; ++i) {
             if (neighbors[i].idx == -1 || neighbors[i].dist > max_dist) {
                 max_dist = neighbors[i].dist;
                 max_idx = i;
             }
         }
         
         if (max_idx != -1 && dist < max_dist) {
             bool exists = false;
             for (int i = 0; i < n_neighbors; ++i) {
                 if (neighbors[i].idx == target) {
                     exists = true;
                     break;
                 }
             }
             
             if (!exists) {
                 neighbors[max_idx].idx = target;
                 neighbors[max_idx].dist = dist;
                 neighbors[max_idx].flag = 'N';
             }
         }
     }
 }
 
 // CUDA kernel for initializing random states
 __global__ void init_random_states_kernel(
     curandState* states, unsigned long long seed, int n)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n) {
         curand_init(seed, idx, 0, &states[idx]);
     }
 }
 
 // Host class for CUDA NN-Descent
 class NNDescentCUDA {
 private:
     float* d_data = nullptr;
     DeviceNeighbor* d_graph = nullptr;
     curandState* d_random_states = nullptr;
     
     int n_nodes;
     int n_features;
     int n_neighbors;
     bool use_euclidean;
     
 public:
     NNDescentCUDA(int nodes, int features, int neighbors, bool euclidean=true)
         : n_nodes(nodes), n_features(features), n_neighbors(neighbors),
           use_euclidean(euclidean) {}
     
     ~NNDescentCUDA() {
         if (d_data) cudaFree(d_data);
         if (d_graph) cudaFree(d_graph);
         if (d_random_states) cudaFree(d_random_states);
     }
     
     void allocate() {
         CUDA_CHECK(cudaMalloc(&d_data, n_nodes * n_features * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_graph, n_nodes * n_neighbors * sizeof(DeviceNeighbor)));
         CUDA_CHECK(cudaMalloc(&d_random_states, n_nodes * sizeof(curandState)));
         
         int blocks = (n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
         init_random_states_kernel<<<blocks, BLOCK_SIZE>>>(
             d_random_states, time(nullptr), n_nodes);
         CUDA_CHECK(cudaGetLastError());
     }
     
     void upload_data(const std::vector<float>& data) {
         CUDA_CHECK(cudaMemcpy(d_data, data.data(), 
                              data.size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
     }
     
     void initialize_graph(const std::vector<int>& initial_indices,
                          const std::vector<float>& initial_distances) {
         std::vector<DeviceNeighbor> host_graph(n_nodes * n_neighbors);
         
         for (int i = 0; i < n_nodes; ++i) {
             for (int j = 0; j < n_neighbors; ++j) {
                 int idx = i * n_neighbors + j;
                 if (j < initial_indices.size() / n_nodes) {
                     host_graph[idx].idx = initial_indices[i * (initial_indices.size() / n_nodes) + j];
                     host_graph[idx].dist = initial_distances[i * (initial_distances.size() / n_nodes) + j];
                     host_graph[idx].flag = 'N';
                 } else {
                     host_graph[idx].idx = -1;
                     host_graph[idx].dist = FLT_MAX;
                     host_graph[idx].flag = 'O';
                 }
             }
         }
         
         CUDA_CHECK(cudaMemcpy(d_graph, host_graph.data(),
                              host_graph.size() * sizeof(DeviceNeighbor),
                              cudaMemcpyHostToDevice));
     }
     
     void nn_descent_iteration(
         const std::vector<int>& candidates,
         std::vector<float>& distances)
     {
         int n_candidates = candidates.size() / 2;
         
         if (n_candidates <= 0) {
             std::cerr << "Warning: No candidates to process" << std::endl;
             distances.clear();
             return;
         }
         
         int* d_source_indices;
         int* d_target_indices;
         float* d_distances;
         
         CUDA_CHECK(cudaMalloc(&d_source_indices, n_candidates * sizeof(int)));
         CUDA_CHECK(cudaMalloc(&d_target_indices, n_candidates * sizeof(int)));
         CUDA_CHECK(cudaMalloc(&d_distances, n_candidates * sizeof(float)));
         
         std::vector<int> source_indices(n_candidates);
         std::vector<int> target_indices(n_candidates);
         
         for (int i = 0; i < n_candidates; ++i) {
             source_indices[i] = candidates[2 * i];
             target_indices[i] = candidates[2 * i + 1];
         }
         
         CUDA_CHECK(cudaMemcpy(d_source_indices, source_indices.data(),
                              n_candidates * sizeof(int), cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemcpy(d_target_indices, target_indices.data(),
                              n_candidates * sizeof(int), cudaMemcpyHostToDevice));
         
         int blocks = (n_candidates + BLOCK_SIZE - 1) / BLOCK_SIZE;
         compute_distances_kernel<<<blocks, BLOCK_SIZE>>>(
             d_data, d_source_indices, d_target_indices, d_distances,
             1, n_candidates, n_features, use_euclidean);
         CUDA_CHECK(cudaGetLastError());
         
         distances.resize(n_candidates);
         CUDA_CHECK(cudaMemcpy(distances.data(), d_distances,
                              n_candidates * sizeof(float), cudaMemcpyDeviceToHost));
         
         update_graph_kernel<<<blocks, BLOCK_SIZE>>>(
             d_graph, d_distances, d_source_indices, d_target_indices,
             n_candidates, n_neighbors, n_nodes);
         CUDA_CHECK(cudaGetLastError());
         
         cudaFree(d_source_indices);
         cudaFree(d_target_indices);
         cudaFree(d_distances);
     }
     
     void download_graph(std::vector<int>& indices, std::vector<float>& distances) {
         std::vector<DeviceNeighbor> host_graph(n_nodes * n_neighbors);
         CUDA_CHECK(cudaMemcpy(host_graph.data(), d_graph,
                              host_graph.size() * sizeof(DeviceNeighbor),
                              cudaMemcpyDeviceToHost));
         
         indices.resize(n_nodes * n_neighbors);
         distances.resize(n_nodes * n_neighbors);
         
         for (int i = 0; i < n_nodes * n_neighbors; ++i) {
             indices[i] = host_graph[i].idx;
             distances[i] = host_graph[i].dist;
         }
     }
 };
 
 // Host wrapper function for CUDA NN-Descent
 void nn_descent_cuda(
     nndescent::Matrix<float>& data,
     nndescent::HeapList<float>& current_graph,
     int n_neighbors,
     nndescent::RandomState& rng_state,
     int max_candidates,
     int n_iters,
     float delta,
     int n_threads,
     bool verbose,
     const std::string& metric)
 {
     int n_nodes = data.nrows();
     int n_features = data.ncols();
     bool use_euclidean = (metric == "euclidean" || metric == "sqeuclidean");
     
     NNDescentCUDA cuda_nnd(n_nodes, n_features, n_neighbors, use_euclidean);
     cuda_nnd.allocate();
     
     std::vector<float> flat_data(n_nodes * n_features);
     for (int i = 0; i < n_nodes; ++i) {
         for (int j = 0; j < n_features; ++j) {
             flat_data[i * n_features + j] = data(i, j);
         }
     }
     cuda_nnd.upload_data(flat_data);
     
     std::vector<int> initial_indices;
     std::vector<float> initial_distances;
     
     for (size_t i = 0; i < current_graph.nheaps(); ++i) {
         for (size_t j = 0; j < current_graph.nnodes(); ++j) {
             initial_indices.push_back(current_graph.indices(i, j));
             initial_distances.push_back(current_graph.keys(i, j));
         }
     }
     cuda_nnd.initialize_graph(initial_indices, initial_distances);
     
     for (int iter = 0; iter < n_iters; ++iter) {
         if (verbose) {
             std::cout << "CUDA Iteration " << iter + 1 << "/" << n_iters << std::endl;
         }
         
         std::vector<int> candidates;
         
         // Check if we need to generate random candidates (first iteration or empty graph)
         bool need_random_candidates = true;
         for (int i = 0; i < n_nodes && need_random_candidates; ++i) {
             for (int j = 0; j < n_neighbors; ++j) {
                 if (current_graph.indices(i, j) != nndescent::NONE) {
                     need_random_candidates = false;
                     break;
                 }
             }
         }
         
         if (need_random_candidates) {
             // Generate random candidates for bootstrap
             for (int i = 0; i < n_nodes; ++i) {
                 for (int j = 0; j < 3; ++j) {  // 3 random candidates per node
                     int neighbor = nndescent::rand_int(rng_state) % n_nodes;
                     if (neighbor != i) {
                         candidates.push_back(i);
                         candidates.push_back(neighbor);
                     }
                 }
             }
         } else {
             // Generate candidates from current graph
             for (int i = 0; i < n_nodes; ++i) {
                 for (int j = 0; j < std::min(max_candidates, n_neighbors); ++j) {
                     int neighbor = current_graph.indices(i, j);
                     if (neighbor != nndescent::NONE && neighbor >= 0 && neighbor < n_nodes) {
                         candidates.push_back(i);
                         candidates.push_back(neighbor);
                     }
                 }
             }
         }
         
         if (candidates.empty()) {
             if (verbose) {
                 std::cout << "No candidates generated, stopping" << std::endl;
             }
             break;
         }
         
         std::vector<float> distances;
         cuda_nnd.nn_descent_iteration(candidates, distances);
         
         int n_updates = 0;
         for (const auto& dist : distances) {
             if (dist < FLT_MAX) n_updates++;
         }
         
         if (n_updates < delta * n_nodes * n_neighbors) {
             if (verbose) {
                 std::cout << "Converged at iteration " << iter + 1 << std::endl;
             }
             break;
         }
     }
     
     std::vector<int> final_indices;
     std::vector<float> final_distances;
     cuda_nnd.download_graph(final_indices, final_distances);
     
     for (int i = 0; i < n_nodes; ++i) {
         for (int j = 0; j < n_neighbors; ++j) {
             int idx = i * n_neighbors + j;
             current_graph.indices(i, j) = final_indices[idx];
             current_graph.keys(i, j) = final_distances[idx];
         }
     }
 }
 
 } // namespace nndescent_cuda