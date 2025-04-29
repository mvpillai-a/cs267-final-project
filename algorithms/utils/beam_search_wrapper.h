#pragma once

#include "cuda_beam_search.h"
#include "beamSearch.h"

namespace parlayANN {

template<typename Point, typename PointRange, typename indexType>
std::pair<parlay::sequence<indexType>, long> beam_search_wrapper(
    const Point &query,
    const Graph<indexType> &G,
    const PointRange &Points,
    indexType starting_point,
    const QueryParams &QP) {
    
    // Call the regular beam search - it will now use CUDA internally when the flag is set
    parlay::sequence<indexType> starting_points = {starting_point};
    auto [pairElts, dist_cmps] = beam_search(query, G, Points, starting_points, QP);
    
    // Extract just the point IDs from the beam search results for visited points
    parlay::sequence<indexType> visited_indices;
    visited_indices.reserve(pairElts.second.size());
    
    for (const auto &p : pairElts.second) {
        visited_indices.push_back(p.first);
    }
    
    return std::make_pair(visited_indices, dist_cmps);
}

} // namespace parlayANN