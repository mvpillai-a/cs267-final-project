#pragma once

#include "graph.h"
#include "types.h"
#include "beamSearch.h"

namespace parlayANN {

// Wrapper function for beam search that can use GPU acceleration
template<typename Point, typename PointRange, typename indexType>
std::pair<parlay::sequence<indexType>, long> beam_search_wrapper(
    const Point &query,
    const Graph<indexType> &G,
    const PointRange &Points,
    indexType starting_point,
    const QueryParams &QP);

} // namespace parlayANN