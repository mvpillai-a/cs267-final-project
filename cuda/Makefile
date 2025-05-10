# CUDA settings for Perlmutter at NERSC
NVCC = nvcc
CUDA_HOME ?= $(CUDA_ROOT)
CUDA_INCLUDE = -I$(CUDA_HOME)/include
CUDA_LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda
CUDA_ARCH = -arch=sm_80
CUDA_FLAGS = $(CUDA_ARCH) --std=c++17 -O3

# C++ settings
CXX = g++
CXXFLAGS = -O3 -DUSE_CUDA -I. -I../hnswlib $(CUDA_INCLUDE) --std=c++17

# Source files
CUDA_SOURCES = cuda_distance.cu cuda_hnsw_kernels.cu
CPP_SOURCES = test_cuda.cpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Target executable
TARGET = test_hnsw_cuda

all: $(TARGET)

# Rule for CUDA compilation
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) -I. -c $< -o $@

# Rule for C++ compilation  
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link everything together
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(CXX) $(CUDA_OBJECTS) $(CPP_OBJECTS) -o $(TARGET) $(CUDA_LIBS)

clean:
	rm -f $(CUDA_OBJECTS) $(CPP_OBJECTS) *.o test_hnsw_cuda test_comparison test_performance

run: $(TARGET)
	./$(TARGET)

run-comparison: comparison
	./test_comparison

# Comparison test
comparison: $(CUDA_OBJECTS) test_comparison.cpp
	$(CXX) $(CXXFLAGS) test_comparison.cpp $(CUDA_OBJECTS) -o test_comparison $(CUDA_LIBS)

batch-comparison: $(CUDA_OBJECTS) test_batch_comparison.cpp
	$(CXX) $(CXXFLAGS) test_batch_comparison.cpp $(CUDA_OBJECTS) -o test_batch_comparison $(CUDA_LIBS)

run-batch-comparison: batch-comparison
	./test_batch_comparison

.PHONY: all clean run comparison performance run-comparison run-performance