# cmake_minimum_required(VERSION 3.10)

# project(nndescent
#     VERSION 1.0.4
#     LANGUAGES CXX
#     DESCRIPTION "A C++ implementation of the nearest neighbour descent algorithm")

# set(CMAKE_CXX_STANDARD 11)


# # Enable the subset mode option for faster compile time
# option(SUBSET_MODE "Enable subset compile mode" OFF)

# # Standard mode includes all distance functions (slower)
# set(STANDARD_CXX_FLAGS
#     "-Wall -g -pg -Ofast -DALL_METRICS -march=native -flto -fno-math-errno -fopenmp -pg")

# # Subset mode (only subset of metrics)
# set(SUBSET_CXX_FLAGS
#     "-Wall -g -pg -Ofast -march=native -flto -fno-math-errno -fopenmp -pg")

# # Select the appropriate compiler flags
# if(SUBSET_MODE)
#     set(CMAKE_CXX_FLAGS "${SUBSET_CXX_FLAGS}")
# else()
#     set(CMAKE_CXX_FLAGS "${STANDARD_CXX_FLAGS}")
# endif()


# # Collect source files
# file(GLOB_RECURSE SRC_FILES src/*.cpp)

# # Library
# add_library(nndescent STATIC ${SRC_FILES})

# # Source code
# target_include_directories(nndescent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


# # Tests
# add_executable(test_distances tests/test_distances.cpp)
# target_link_libraries(test_distances PRIVATE nndescent)

# add_executable(simple tests/simple.cpp)
# target_link_libraries(simple PRIVATE nndescent)

# add_executable(faces tests/faces.cpp)
# target_link_libraries(faces PRIVATE nndescent)

# add_executable(sparse_faces tests/sparse_faces.cpp)
# target_link_libraries(sparse_faces PRIVATE nndescent)

# add_executable(fmnist tests/fmnist.cpp)
# target_link_libraries(fmnist PRIVATE nndescent)


cmake_minimum_required(VERSION 3.18)
project(nndescent
    VERSION 1.0.4
    LANGUAGES CXX CUDA
    DESCRIPTION "A C++ implementation of the nearest neighbour descent algorithm")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Enable the subset mode option for faster compile time
option(SUBSET_MODE "Enable subset compile mode" OFF)

# Standard mode includes all distance functions (slower)
set(STANDARD_CXX_FLAGS
    "-Wall -g -pg -Ofast -DALL_METRICS -march=native -flto -fno-math-errno -fopenmp -pg")

# Subset mode (only subset of metrics)
set(SUBSET_CXX_FLAGS
    "-Wall -g -pg -Ofast -march=native -flto -fno-math-errno -fopenmp -pg")

# Select the appropriate compiler flags
if(SUBSET_MODE)
    set(CMAKE_CXX_FLAGS "${SUBSET_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${STANDARD_CXX_FLAGS}")
endif()

# Collect source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Library
add_library(nndescent STATIC ${SRC_FILES})

# CUDA library
add_library(nndescent_cuda STATIC src/nnd_cuda.cu)
target_link_libraries(nndescent_cuda PUBLIC nndescent)
set_target_properties(nndescent_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

# Source code
target_include_directories(nndescent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(nndescent_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Tests
add_executable(test_distances tests/test_distances.cpp)
target_link_libraries(test_distances PRIVATE nndescent)

add_executable(simple tests/simple.cpp)
target_link_libraries(simple PRIVATE nndescent)

add_executable(faces tests/faces.cpp)
target_link_libraries(faces PRIVATE nndescent)

add_executable(sparse_faces tests/sparse_faces.cpp)
target_link_libraries(sparse_faces PRIVATE nndescent)

add_executable(fmnist tests/fmnist.cpp)
target_link_libraries(fmnist PRIVATE nndescent)

# CUDA benchmark test
add_executable(cuda_benchmark tests/cuda_benchmark.cpp)
target_link_libraries(cuda_benchmark PRIVATE nndescent nndescent_cuda)