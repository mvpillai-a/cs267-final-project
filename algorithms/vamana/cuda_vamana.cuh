// cuda_vamana.cuh
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "../utils/graph.h"
#include "../utils/types.h"

namespace parlayANN {

template<typename indexType>
class CudaVamana {
public:
    CudaVamana(const BuildParams& bp, int dimensions);
    ~CudaVamana();
    
    void build_index(Graph<indexType>& graph);
    void transfer_to_device(const float* points, size_t num_points, const Graph<indexType>& graph);
    void transfer_to_host(Graph<indexType>& graph);
    
private:
    void batch_insert(double alpha);
    void robust_prune(int point_id, const std::vector<std::pair<indexType, float>>& candidates);
    void make_bidirectional();
    
    // Device data
    float* d_points_ = nullptr;
    indexType* d_graph_ = nullptr;
    int* d_offsets_ = nullptr;
    int* d_out_degrees_ = nullptr;
    
    // Parameters
    BuildParams bp_;
    int dim_;
    size_t num_points_;
    size_t total_edges_;
    
    // CUDA stream
    cudaStream_t stream_;
};

} // namespace parlayANN
