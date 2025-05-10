/**
 * @file nnd_cuda.h
 * @brief Header file for CUDA-accelerated NN-Descent implementation
 */

 #pragma once

 #include "nnd.h"
 #include <string>
 
 namespace nndescent_cuda
 {
 
 /**
  * @brief CUDA-accelerated implementation of NN-Descent algorithm
  * 
  * This function provides a GPU-accelerated version of the NN-Descent algorithm
  * for approximate nearest neighbor search. It is a drop-in replacement for the
  * CPU version with the same interface.
  * 
  * @param data The input data matrix
  * @param current_graph The current nearest neighbor graph (will be modified)
  * @param n_neighbors The number of neighbors to find
  * @param rng_state Random number generator state
  * @param max_candidates Maximum number of candidates to consider
  * @param n_iters Number of iterations
  * @param delta Convergence threshold
  * @param n_threads Number of threads (not used in CUDA version)
  * @param verbose Whether to print progress information
  * @param metric Distance metric ("euclidean" or "cosine")
  */
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
     const std::string& metric
 );
 
 } // namespace nndescent_cuda