# Include the original parallel definitions
include ../bench/parallelDefsANN

# CUDA settings for Perlmutter at NERSC
NVCC = nvcc
CUDA_HOME ?= $(CUDA_ROOT)
CUDA_INCLUDE = -I$(CUDA_HOME)/include
CUDA_LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda
# A100 GPUs use Ampere architecture which corresponds to compute capability 8.0
CUDA_ARCH = -arch=sm_80
CUDA_FLAGS = $(CUDA_ARCH) --std=c++17 -O3

# Original requirements plus CUDA files
REQUIRE = ../utils/beamSearch.h index.h ../utils/check_nn_recall.h ../utils/NSGDist.h ../utils/parse_results.h ../utils/graph.h ../utils/point_range.h ../utils/euclidian_point.h ../utils/mips_point.h ../utils/jl_point.h ../utils/beam_search_wrapper.h

# Additional include directories
INCLUDE = -I ../../parlaylib/include/ $(CUDA_INCLUDE)

# Benchmark name
BENCH = neighbors

# CUDA object files
CUDA_OBJS = cuda_euclidean_kernel.o

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE) -c $< -o $@

# Main target
all: $(BENCH)-vamana

# Build the CUDA-enabled benchmark
$(BENCH)-vamana: $(CUDA_OBJS) ../bench/$(BENCH)Time.C $(BENCH).h $(REQUIRE)
	$(CC) -DSTATS $(CFLAGS) $(INCLUDE) -include $(BENCH).h -o $(BENCH)-vamana ../bench/$(BENCH)Time.C $(CUDA_OBJS) $(LFLAGS) $(CUDA_LIBS)

clean:
	rm -f $(BENCH)-vamana $(CUDA_OBJS)

cleanall: clean
	rm -f testInputs*; cd ../bench; make -s clean