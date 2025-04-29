add_executable(neighbors-vamana ../bench/neighborsTime.C)
  target_link_libraries(neighbors-vamana PRIVATE parlay)
  target_precompile_headers(neighbors-vamana PRIVATE neighbors.h)

# Add CUDA to your build system
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA compilation options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75 --expt-relaxed-constexpr)

# Add the CUDA files to your build
cuda_add_library(cuda_utils 
    cuda_euclidean_kernel.cu
)

# Link the CUDA library to your main target
target_link_libraries(neighbors-vamana PRIVATE cuda_utils ${CUDA_LIBRARIES})
