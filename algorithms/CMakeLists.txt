add_subdirectory(HCNNG)
add_subdirectory(HNSW)
add_subdirectory(pyNNDescent)
add_subdirectory(vamana)
add_subdirectory(vamanaRange)

# Add these lines to your CMakeLists.txt in the vamana directory

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  
  # CUDA compilation flags
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
  
  # Add CUDA implementation
  add_library(cuda_vamana 
    cuda_vamana.cu
  )
  
  target_include_directories(cuda_vamana PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  
  target_link_libraries(cuda_vamana PUBLIC
    CUDA::cudart
    CUDA::cublas
  )
  
  # Add CUDA support to main executable
  add_executable(neighbors-vamana-cuda ../bench/neighborsTime.C)
  target_link_libraries(neighbors-vamana-cuda PRIVATE parlay cuda_vamana)
  target_compile_definitions(neighbors-vamana-cuda PRIVATE USE_CUDA=1)
  target_precompile_headers(neighbors-vamana-cuda PRIVATE neighbors.h)
else()
  message(STATUS "CUDA not found, building CPU-only version")
endif()

# Original CPU-only executable
add_executable(neighbors-vamana ../bench/neighborsTime.C)
target_link_libraries(neighbors-vamana PRIVATE parlay)
target_precompile_headers(neighbors-vamana PRIVATE neighbors.h)