#pragma once

#include "nnd.h"

namespace nndescent_cuda {

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