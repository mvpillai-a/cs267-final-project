#pragma once

#include "beamSearch.h"
#include "graph.h"
#include "types.h"

namespace parlayANN {

// Wrapper function for beam search that can decide whether to use CPU or GPU
template<typename Point, typename PointRange, typename indexType>
std::pair<parlay::sequence<indexType>, long> beam_search_wrapper(
    const Point &query,
    const Graph<indexType> &G,
    const PointRange &Points,
    indexType starting_point,
    const QueryParams &QP) {
    
    // Call the regular beam search - the distance function internally
    // will use CUDA if the flag is enabled in euclidian_point.h
    parlay::sequence<indexType> starting_points = {starting_point};
    auto [result, dist_cmps] = beam_search(query, G, Points, starting_points, QP);
    
    // Extract just the visited points for the robustPrune function
    parlay::sequence<indexType> visited_ids;
    visited_ids.reserve(result.second.size());
    
    for (const auto &p : result.second) {
        visited_ids.push_back(p.first);
    }
    
    return std::make_pair(visited_ids, dist_cmps);
}

} // namespace parlayANN