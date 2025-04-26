// cuda_vamana.cu
#include "cuda_vamana.cuh"

namespace parlayANN {

// CUDA kernels
__global__ void compute_distances_kernel(const float* points, const int* candidates, 
                                       float* distances, int p, int num_candidates, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_candidates) {
        const float* p1 = &points[p * dim];
        const float* p2 = &points[candidates[idx] * dim];
        
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        distances[idx] = sqrtf(sum);
    }
}

__global__ void beam_search_kernel(const float* points, const int* graph, const int* offsets,
                                 const int* out_degrees, int* search_results, 
                                 int num_points, int beam_width, int dim, int max_degree) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    
    // Each thread handles one query point
    int* my_results = &search_results[tid * beam_width];
    
    // Initialize frontier with start point
    my_results[0] = 0; // start point
    int frontier_size = 1;
    
    // Beam search iterations
    for (int iter = 0; iter < beam_width; iter++) {
        // Expand frontier by exploring neighbors
        int frontier_idx = iter % beam_width;
        int current_node = my_results[frontier_idx];
        
        // Get neighbors of current node
        int offset = offsets[current_node];
        int degree = out_degrees[current_node];
        
        // Process each neighbor
        for (int i = 0; i < degree && frontier_size < beam_width; i++) {
            int neighbor = graph[offset + i];
            // Add to results if not already there
            bool found = false;
            for (int j = 0; j < frontier_size; j++) {
                if (my_results[j] == neighbor) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                my_results[frontier_size++] = neighbor;
            }
        }
    }
}

// Implementation of CudaVamana methods
template<typename indexType>
CudaVamana<indexType>::CudaVamana(const BuildParams& bp, int dimensions)
    : bp_(bp), dim_(dimensions) {
    // Initialize CUDA resources
    cudaStreamCreate(&stream_);
}

template<typename indexType>
CudaVamana<indexType>::~CudaVamana() {
    // Clean up CUDA resources
    if (d_points_) cudaFree(d_points_);
    if (d_graph_) cudaFree(d_graph_);
    if (d_offsets_) cudaFree(d_offsets_);
    if (d_out_degrees_) cudaFree(d_out_degrees_);
    cudaStreamDestroy(stream_);
}

template<typename indexType>
void CudaVamana<indexType>::transfer_to_device(const float* points, size_t num_points,
                                             const Graph<indexType>& graph) {
    num_points_ = num_points;
    
    // Allocate and copy points
    size_t points_size = num_points * dim_ * sizeof(float);
    cudaMalloc(&d_points_, points_size);
    cudaMemcpy(d_points_, points, points_size, cudaMemcpyHostToDevice);
    
    // Flatten graph structure
    std::vector<indexType> flat_graph;
    std::vector<int> offsets(num_points + 1, 0);
    std::vector<int> out_degrees(num_points);
    
    for (size_t i = 0; i < num_points; i++) {
        const auto& neighbors = graph[i];
        out_degrees[i] = neighbors.size();
        offsets[i + 1] = offsets[i] + neighbors.size();
        for (auto neighbor : neighbors) {
            flat_graph.push_back(neighbor);
        }
    }
    
    total_edges_ = flat_graph.size();
    
    // Allocate and copy graph data
    cudaMalloc(&d_graph_, total_edges_ * sizeof(indexType));
    cudaMalloc(&d_offsets_, (num_points + 1) * sizeof(int));
    cudaMalloc(&d_out_degrees_, num_points * sizeof(int));
    
    cudaMemcpy(d_graph_, flat_graph.data(), total_edges_ * sizeof(indexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets_, offsets.data(), (num_points + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_degrees_, out_degrees.data(), num_points * sizeof(int), cudaMemcpyHostToDevice);
}

template<typename indexType>
void CudaVamana<indexType>::build_index(Graph<indexType>& graph) {
    // Multiple passes
    for (int pass = 0; pass < bp_.num_passes; pass++) {
        double alpha = (pass == bp_.num_passes - 1) ? bp_.alpha : 1.0;
        batch_insert(alpha);
    }
    
    // Transfer results back to host
    transfer_to_host(graph);
}

template<typename indexType>
void CudaVamana<indexType>::batch_insert(double alpha) {
    int threads_per_block = 256;
    int num_blocks = (num_points_ + threads_per_block - 1) / threads_per_block;
    
    // Allocate search results
    indexType* d_search_results;
    cudaMalloc(&d_search_results, num_points_ * bp_.L * sizeof(indexType));
    
    // Launch beam search
    beam_search_kernel<<<num_blocks, threads_per_block, 0, stream_>>>(
        d_points_, d_graph_, d_offsets_, d_out_degrees_, d_search_results,
        num_points_, bp_.L, dim_, bp_.R
    );
    
    // TODO: Implement robust prune
    
    // TODO: Make edges bidirectional
    
    cudaFree(d_search_results);
}

template<typename indexType>
void CudaVamana<indexType>::transfer_to_host(Graph<indexType>& graph) {
    // Transfer graph back to host
    std::vector<indexType> flat_graph(total_edges_);
    std::vector<int> offsets(num_points_ + 1);
    std::vector<int> out_degrees(num_points_);
    
    cudaMemcpy(flat_graph.data(), d_graph_, total_edges_ * sizeof(indexType), cudaMemcpyDeviceToHost);
    cudaMemcpy(offsets.data(), d_offsets_, (num_points_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_degrees.data(), d_out_degrees_, num_points_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Reconstruct graph
    for (size_t i = 0; i < num_points_; i++) {
        std::vector<indexType> neighbors;
        for (int j = 0; j < out_degrees[i]; j++) {
            neighbors.push_back(flat_graph[offsets[i] + j]);
        }
        graph[i].update_neighbors(neighbors);
    }
}

// Explicit template instantiations
template class CudaVamana<int>;
template class CudaVamana<unsigned int>;
template class CudaVamana<long>;

} // namespace parlayANN
