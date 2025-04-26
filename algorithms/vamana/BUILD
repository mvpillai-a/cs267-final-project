# Vamana algorithm.

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        "@parlaylib//parlay:delayed",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/utils:graph",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:types",
        "//algorithms/utils:point_range",
    ],
)

cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        ":index",
        "//algorithms/utils:point_range",
        "//algorithms/utils:types",
        "//algorithms/utils:mmap",
        "//algorithms/utils:graph",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:stats",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "@parlaylib//parlay:delayed",
    ],
)

cc_library(
    name = "neighbors",
    hdrs = ["neighbors.h"],
    deps = [
        ":index",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:check_nn_recall",
        "//algorithms/utils:csvfile",
        "//algorithms/utils:parse_results",
        "//algorithms/utils:graph",
        "//algorithms/utils:jl_point",
        "//algorithms/utils:stats",
        "//algorithms/utils:types",
        "//algorithms/utils:point_range",
        "//algorithms/utils:euclidean_point",
        "//algorithms/utils:mips_point",
    ],
)

cc_test(
    name = "neighbors_test",
    size = "small",
    srcs = ["neighbors_test.cc"],
    deps = [
        "@googletest//:gtest_main",
        ":neighbors",
    ],
)

# Add to your existing BUILD file for Vamana algorithm

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# CUDA implementation library
cuda_library(
    name = "cuda_vamana",
    srcs = ["cuda_vamana.cu"],
    hdrs = ["cuda_vamana.cuh"],
    deps = [
        "//algorithms/utils:graph",
        "//algorithms/utils:types",
        "//algorithms/utils:point_range",
        "@cuda//:cuda_runtime",
        "@cuda//:thrust",
    ],
    copts = ["-O3"],
    cuda_deps = ["@cuda//:cublas"],
)

# Main Vamana index with optional CUDA support
cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        "@parlaylib//parlay:delayed",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/utils:graph",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:types",
        "//algorithms/utils:point_range",
    ] + select({
        "//conditions:default": [],
        "@platforms//gpu:gpu_available": [":cuda_vamana"],
    }),
    defines = select({
        "//conditions:default": [],
        "@platforms//gpu:gpu_available": ["USE_CUDA=1"],
    }),
)

# Same for neighbors library
cc_library(
    name = "neighbors",
    hdrs = ["neighbors.h"],
    deps = [
        ":index",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "//algorithms/utils:beamSearch",
        "//algorithms/utils:check_nn_recall",
        "//algorithms/utils:csvfile",
        "//algorithms/utils:parse_results",
        "//algorithms/utils:graph",
        "//algorithms/utils:jl_point",
        "//algorithms/utils:stats",
        "//algorithms/utils:types",
        "//algorithms/utils:point_range",
        "//algorithms/utils:euclidean_point",
        "//algorithms/utils:mips_point",
    ] + select({
        "//conditions:default": [],
        "@platforms//gpu:gpu_build": [":cuda_vamana"],
    }),
    defines = select({
        "//conditions:default": [],
        "@platforms//gpu:gpu_build": ["USE_CUDA=1"],
    }),
)

# CUDA-specific test
cc_test(
    name = "cuda_vamana_test",
    srcs = ["cuda_vamana_test.cc"],
    deps = [
        ":index",
        ":neighbors",
        "@googletest//:gtest_main",
    ],
    tags = ["gpu"],
)
