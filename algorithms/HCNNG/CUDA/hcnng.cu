// Optimized HCNNG with CUDA implementation
// Compile with: nvcc hcnng.cu -o hcnng -std=c++17 -O3


#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <ctime>
#include <cmath>
#include <sys/time.h>
#include "common.h"
#include <mutex>
#include <random>
#include <time.h>
#include <thread>
#include <chrono>
#include <atomic>
// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using namespace std;
using namespace std::chrono;


// Structure to hold timing information
struct TimingInfo {
   high_resolution_clock::time_point start_time;
   double total_time = 0.0;
   double gpu_time = 0.0;
   double cpu_time = 0.0;
   double clustering_time = 0.0;
   double mst_time = 0.0;
   double sort_time = 0.0;
   int gpu_calls = 0;
   int cpu_calls = 0;
};


// Global timing structure
TimingInfo timing;


// Start timing
void start_timer() {
   timing.start_time = high_resolution_clock::now();
}


// Stop timing and return elapsed time in seconds
double stop_timer() {
   auto end_time = high_resolution_clock::now();
   duration<double> elapsed = end_time - timing.start_time;
   return elapsed.count();
}


// Global flag to disable CUDA if errors occur
bool g_cuda_enabled = true;


// CUDA error checking macro
#define CUDA_CHECK(call) { \
   if (!g_cuda_enabled) return; \
   cudaError_t err = call; \
   if (err != cudaSuccess) { \
       fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
       g_cuda_enabled = false; \
       fprintf(stderr, "Disabling CUDA and switching to CPU-only mode\n"); \
   } \
}


// Version that doesn't return
#define CUDA_CHECK_CONTINUE(call) { \
   if (g_cuda_enabled) { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
           fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
           g_cuda_enabled = false; \
           fprintf(stderr, "Disabling CUDA and switching to CPU-only mode\n"); \
       } \
   } \
}


// Utility function to get available GPU memory
void printGPUMemoryInfo() {
   if (!g_cuda_enabled) {
       printf("GPU disabled, using CPU-only mode\n");
       return;
   }
  
   size_t free_byte, total_byte;
   cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
   if (err != cudaSuccess) {
       fprintf(stderr, "Error getting GPU memory info: %s\n", cudaGetErrorString(err));
       return;
   }
  
   double free_db = (double)free_byte;
   double total_db = (double)total_byte;
   double used_db = total_db - free_db;
   printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
       used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}


int xxx = 0;


// Simple CPU-based L2 distance function as failsafe
float cpu_dist_L2(float* a, float* b, int dim) {
   float sum = 0.0f;
   for (int i = 0; i < dim; i++) {
       float diff = a[i] - b[i];
       sum += diff * diff;
   }
   return sqrt(sum);
}


// Efficient CUDA kernel for distance calculation
__global__ void simple_dist_kernel(float* points, int dim, int num_points, int* idx_left, int* idx_right, float* distances) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < num_points) {
       int left_idx = idx_left[idx];
       int right_idx = idx_right[idx];
      
       float sum = 0.0f;
       for (int k = 0; k < dim; k++) {
           float diff = points[left_idx * dim + k] - points[right_idx * dim + k];
           sum += diff * diff;
       }
      
       distances[idx] = sqrt(sum);
   }
}


// Compute pairwise distances using GPU
bool gpu_compute_distances(float* d_points, int dim, int* h_idx1, int* h_idx2, int count, float* h_distances) {
   if (!g_cuda_enabled || count <= 0) {
       return false;
   }
  
   start_timer(); // Start GPU timing
  
   try {
       int* d_idx1 = nullptr;
       int* d_idx2 = nullptr;
       float* d_distances = nullptr;
      
       // Allocate device memory
       CUDA_CHECK_CONTINUE(cudaMalloc(&d_idx1, count * sizeof(int)));
       CUDA_CHECK_CONTINUE(cudaMalloc(&d_idx2, count * sizeof(int)));
       CUDA_CHECK_CONTINUE(cudaMalloc(&d_distances, count * sizeof(float)));
      
       if (!g_cuda_enabled) {
           // If CUDA was disabled during allocation, clean up and return
           if (d_idx1) cudaFree(d_idx1);
           if (d_idx2) cudaFree(d_idx2);
           if (d_distances) cudaFree(d_distances);
           return false;
       }
      
       // Copy indices to device
       CUDA_CHECK_CONTINUE(cudaMemcpy(d_idx1, h_idx1, count * sizeof(int), cudaMemcpyHostToDevice));
       CUDA_CHECK_CONTINUE(cudaMemcpy(d_idx2, h_idx2, count * sizeof(int), cudaMemcpyHostToDevice));
      
       if (!g_cuda_enabled) {
           if (d_idx1) cudaFree(d_idx1);
           if (d_idx2) cudaFree(d_idx2);
           if (d_distances) cudaFree(d_distances);
           return false;
       }
      
       // Initialize results to 0
       CUDA_CHECK_CONTINUE(cudaMemset(d_distances, 0, count * sizeof(float)));
      
       // Calculate grid and block dimensions
       int threadsPerBlock = 256;
       int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
      
       // Launch kernel
       simple_dist_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, dim, count, d_idx1, d_idx2, d_distances);
      
       // Check for kernel errors
       CUDA_CHECK_CONTINUE(cudaGetLastError());
       CUDA_CHECK_CONTINUE(cudaDeviceSynchronize());
      
       if (!g_cuda_enabled) {
           if (d_idx1) cudaFree(d_idx1);
           if (d_idx2) cudaFree(d_idx2);
           if (d_distances) cudaFree(d_distances);
           return false;
       }
      
       // Copy results back to host
       CUDA_CHECK_CONTINUE(cudaMemcpy(h_distances, d_distances, count * sizeof(float), cudaMemcpyDeviceToHost));
      
       // Free device memory
       CUDA_CHECK_CONTINUE(cudaFree(d_idx1));
       CUDA_CHECK_CONTINUE(cudaFree(d_idx2));
       CUDA_CHECK_CONTINUE(cudaFree(d_distances));
      
       timing.gpu_time += stop_timer(); // Add GPU execution time
       timing.gpu_calls++;
      
       return g_cuda_enabled;
   }
   catch (...) {
       g_cuda_enabled = false;
       fprintf(stderr, "Exception in GPU distance calculation, disabling CUDA\n");
       return false;
   }
}


// Optimized version of MST creation
tuple<Graph, float> kruskal_mst(vector<Edge> &edges, int N, Matrix<float> &points, int max_mst_degree) {
   sort(edges.begin(), edges.end());
   Graph MST(N);
   DisjointSet *disjset = new DisjointSet(N);
   float cost = 0;
  
   // Pre-allocate memory for MST edges
   for(int i = 0; i < N; i++) {
       MST[i].reserve(max_mst_degree);
   }
  
   for(Edge &e : edges) {
       if(disjset->find(e.v1) != disjset->find(e.v2) &&
          MST[e.v1].size() < max_mst_degree &&
          MST[e.v2].size() < max_mst_degree) {
           MST[e.v1].push_back(e);
           MST[e.v2].push_back(Edge(e.v2, e.v1, e.weight));
           disjset->_union(e.v1, e.v2);
           cost += e.weight;
       }
   }
   delete disjset;
   return make_tuple(MST, cost);
}


// CPU implementation for MST
Graph create_exact_mst_cpu(Matrix<float> &points, int *idx_points, int left, int right, int max_mst_degree) {
   start_timer(); // Start CPU MST timing
  
   int N = right-left+1;
   if(N==1) {
       xxx++;
       if (xxx % 1000 == 0) {
           printf("Progress: %d single-point MSTs\n", xxx);
       }
       Graph empty_mst(N);
       return empty_mst;
   }
  
   float cost;
   vector<Edge> full;
   Graph mst;
   full.reserve(N*(N-1));
  
   // CPU-based distance calculation
   start_timer(); // Start CPU distance timing
   for(int i=0; i<N; i++) {
       for(int j=0; j<N; j++) {
           if(i!=j) {
               float distance = dist_L2(points[idx_points[left+i]], points[idx_points[left+j]], points.cols);
               full.push_back(Edge(i, j, distance));
           }
       }
   }
   timing.cpu_time += stop_timer(); // Add CPU distance calculation time
   timing.cpu_calls++;
  
   tie(mst, cost) = kruskal_mst(full, N, points, max_mst_degree);
  
   timing.mst_time += stop_timer(); // Add MST creation time
  
   return mst;
}


// GPU-enhanced MST creation
Graph create_exact_mst(Matrix<float> &points, int *idx_points, int left, int right,
                      int max_mst_degree, float* d_points) {
   start_timer(); // Start MST timing
  
   // If CUDA is disabled, use CPU implementation
   if (!g_cuda_enabled) {
       return create_exact_mst_cpu(points, idx_points, left, right, max_mst_degree);
   }
  
   int N = right-left+1;
  
   // Use CPU implementation for very small or very large clusters
   if(N==1 || N > 2000) {
       return create_exact_mst_cpu(points, idx_points, left, right, max_mst_degree);
   }
  
   float cost;
   vector<Edge> full;
   Graph mst;
  
   // More careful memory usage for large datasets
   try {
       full.reserve(N*(N-1));
      
       // Prepare arrays for the pairwise distance calculation
       int* idx1 = new int[N*(N-1)];
       int* idx2 = new int[N*(N-1)];
       float* distances = new float[N*(N-1)];
      
       // Fill index arrays for all pairs
       int count = 0;
       for(int i=0; i<N; i++) {
           for(int j=0; j<N; j++) {
               if(i!=j) {
                   idx1[count] = idx_points[left+i];
                   idx2[count] = idx_points[left+j];
                   count++;
               }
           }
       }
      
       // Calculate distances using GPU
       bool gpu_success = gpu_compute_distances(d_points, points.cols, idx1, idx2, count, distances);
      
       if (!gpu_success) {
           // If GPU computation failed, fall back to CPU
           delete[] idx1;
           delete[] idx2;
           delete[] distances;
           return create_exact_mst_cpu(points, idx_points, left, right, max_mst_degree);
       }
      
       // Create edges from distances
       count = 0;
       for(int i=0; i<N; i++) {
           for(int j=0; j<N; j++) {
               if(i!=j) {
                   full.push_back(Edge(i, j, distances[count]));
                   count++;
               }
           }
       }
      
       // Clean up
       delete[] idx1;
       delete[] idx2;
       delete[] distances;
      
   } catch (...) {
       // If anything goes wrong, fall back to CPU
       printf("Exception in MST creation, falling back to CPU\n");
       return create_exact_mst_cpu(points, idx_points, left, right, max_mst_degree);
   }
  
   tie(mst, cost) = kruskal_mst(full, N, points, max_mst_degree);
  
   timing.mst_time += stop_timer(); // Add MST creation time
  
   return mst;
}


bool check_in_neighbors(int u, vector<Edge> &neigh) {
   for(int i=0; i<neigh.size(); i++)
       if(neigh[i].v2 == u)
           return true;
   return false;
}


// Calculate distances for split operation using CPU
void calculate_distances_cpu(Matrix<float> &points, int *idx_points, int left, int num_points,
                           vector<pair<float,int>> &dx, vector<pair<float,int>> &dy,
                           int x, int y) {
   start_timer(); // Start CPU distance timing
  
   for(int i=0; i<num_points; i++) {
       dx[i] = make_pair(dist_L2(points[idx_points[x]], points[idx_points[left+i]], points.cols),
                         idx_points[left+i]);
       dy[i] = make_pair(dist_L2(points[idx_points[y]], points[idx_points[left+i]], points.cols),
                         idx_points[left+i]);
   }
  
   timing.cpu_time += stop_timer(); // Add CPU distance calculation time
   timing.cpu_calls++;
}


// Calculate distances using GPU
void calculate_distances(Matrix<float> &points, int *idx_points, int left, int num_points,
                       vector<pair<float,int>> &dx, vector<pair<float,int>> &dy,
                       int x, int y, float* d_points) {
  
   // If CUDA is disabled or for small clusters, use CPU
   if (!g_cuda_enabled || num_points < 1000) {
       calculate_distances_cpu(points, idx_points, left, num_points, dx, dy, x, y);
       return;
   }
  
   try {
       // Prepare arrays for GPU computation
       int* idx1 = new int[num_points * 2];
       int* idx2 = new int[num_points * 2];
       float* distances = new float[num_points * 2];
      
       // Fill index arrays for both pivot distances
       for(int i=0; i<num_points; i++) {
           // Point i to pivot x
           idx1[i] = idx_points[left+i];
           idx2[i] = idx_points[x];
          
           // Point i to pivot y
           idx1[i + num_points] = idx_points[left+i];
           idx2[i + num_points] = idx_points[y];
       }
      
       // Calculate distances using GPU
       bool gpu_success = gpu_compute_distances(d_points, points.cols, idx1, idx2, num_points * 2, distances);
      
       if (!gpu_success) {
           // If GPU computation failed, use CPU
           delete[] idx1;
           delete[] idx2;
           delete[] distances;
           calculate_distances_cpu(points, idx_points, left, num_points, dx, dy, x, y);
           return;
       }
      
       // Create distance pairs
       for(int i=0; i<num_points; i++) {
           dx[i] = make_pair(distances[i], idx_points[left+i]);
           dy[i] = make_pair(distances[i + num_points], idx_points[left+i]);
       }
      
       // Clean up
       delete[] idx1;
       delete[] idx2;
       delete[] distances;
      
   } catch (...) {
       // If anything goes wrong, use CPU
       printf("Exception in GPU distance calculation, using CPU\n");
       calculate_distances_cpu(points, idx_points, left, num_points, dx, dy, x, y);
   }
}


// The main cluster creation function
void create_clusters(Matrix<float> &points, int *idx_points, int left, int right, Graph &graph,
                   int minsize_cl, vector<mutex>& locks, int max_mst_degree, float* d_points) {
   int num_points = right - left + 1;


   if(num_points < minsize_cl) {
       Graph mst = create_exact_mst(points, idx_points, left, right, max_mst_degree, d_points);
      
       for(int i=0; i<num_points; i++) {
           for(int j=0; j<mst[i].size(); j++) {
               locks[idx_points[left+i]].lock();
               if(!check_in_neighbors(idx_points[left+mst[i][j].v2], graph[idx_points[left+i]]))
                   graph[idx_points[left+i]].push_back(Edge(idx_points[left+i],
                                                        idx_points[left+mst[i][j].v2],
                                                        mst[i][j].weight));
               locks[idx_points[left+i]].unlock();
           }
       }
   } else {
       int x = rand_int(left, right);
       int y = rand_int(left, right);
       while(y==x) y = rand_int(left, right);


       vector<pair<float,int>> dx(num_points);
       vector<pair<float,int>> dy(num_points);
       unordered_set<int> taken;
      
       // Calculate distances
       calculate_distances(points, idx_points, left, num_points, dx, dy, x, y, d_points);
      
       sort(dx.begin(), dx.end());
       sort(dy.begin(), dy.end());
      
       int i = 0, j = 0, turn = rand_int(0, 1), p = left, q = right;
       while(i<num_points || j<num_points) {
           if(turn == 0) {
               if(i<num_points) {
                   if(taken.find(dx[i].second) == taken.end()) {
                       idx_points[p] = dx[i].second;
                       taken.insert(dx[i].second);
                       p++;
                       turn = (turn+1)%2;
                   }
                   i++;
               } else {
                   turn = (turn+1)%2;
               }
           } else {
               if(j<num_points) {
                   if(taken.find(dy[j].second) == taken.end()) {
                       idx_points[q] = dy[j].second;
                       taken.insert(dy[j].second);
                       q--;
                       turn = (turn+1)%2;
                   }
                   j++;
               } else {
                   turn = (turn+1)%2;
               }
           }
       }
      
       dx.clear();
       dy.clear();
       taken.clear();
       vector<pair<float,int>>().swap(dx);
       vector<pair<float,int>>().swap(dy);


       create_clusters(points, idx_points, left, p-1, graph, minsize_cl, locks, max_mst_degree, d_points);
       create_clusters(points, idx_points, p, right, graph, minsize_cl, locks, max_mst_degree, d_points);
   }
}


// Function to determine optimal thread count
int getOptimalThreadCount() {
   int cpu_threads = thread::hardware_concurrency();
   if (cpu_threads <= 0) cpu_threads = 4; // Default if hardware_concurrency fails
  
   return max(1, min(cpu_threads - 1, 32)); // Cap at 8 threads to avoid system overload
}


// Main function to create HCNNG graph
Graph HCNNG_create_graph(Matrix<float> &points, int Dim, int num_cl, int minsize_cl, int max_mst_degree) {
   auto overall_start = high_resolution_clock::now(); // Overall timing start
  
   int N = points.rows;
   Graph G(N);
  
   // Replace OpenMP locks with std::mutex
   vector<mutex> locks(N);
  
   for(int i=0; i<N; i++) {
       G[i].reserve(max_mst_degree*num_cl);
   }
  
   // Print GPU memory info if CUDA is enabled
   if (g_cuda_enabled) {
       printGPUMemoryInfo();
   } else {
       printf("CUDA disabled, using CPU-only implementation\n");
   }
  
   // Only allocate GPU memory if CUDA is enabled
   float* d_points = nullptr;
   if (g_cuda_enabled) {
       try {
           size_t points_size = (size_t)N * Dim * sizeof(float);
          
           // Check if we have enough GPU memory
           size_t free_byte, total_byte;
           cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
           if (err != cudaSuccess) {
               fprintf(stderr, "Error checking GPU memory: %s\n", cudaGetErrorString(err));
               g_cuda_enabled = false;
           } else {
               if (free_byte < points_size * 1.2) { // Add 20% margin
                   fprintf(stderr, "Warning: Not enough GPU memory for full dataset.\n");
                   size_t max_points = free_byte * 0.7 / sizeof(float) / Dim;
                   fprintf(stderr, "Using first %zu points for GPU computation.\n", max_points);
                   points_size = max_points * Dim * sizeof(float);
               }
              
               err = cudaMalloc(&d_points, points_size);
               if (err != cudaSuccess) {
                   fprintf(stderr, "Error allocating GPU memory: %s\n", cudaGetErrorString(err));
                   g_cuda_enabled = false;
               } else {
                   // Create a temporary buffer to copy data
                   float* temp_buffer = new float[N * Dim];
                  
                   // Copy data from Matrix to continuous buffer
                   for(int i = 0; i < N; i++) {
                       for(int j = 0; j < Dim; j++) {
                           temp_buffer[i * Dim + j] = points[i][j];
                       }
                   }
                  
                   // Copy as much as we can to device
                   size_t points_to_copy = min(points_size, (size_t)N * Dim * sizeof(float));
                   err = cudaMemcpy(d_points, temp_buffer, points_to_copy, cudaMemcpyHostToDevice);
                   delete[] temp_buffer;
                  
                   if (err != cudaSuccess) {
                       fprintf(stderr, "Error copying data to GPU: %s\n", cudaGetErrorString(err));
                       g_cuda_enabled = false;
                       cudaFree(d_points);
                       d_points = nullptr;
                   }
               }
           }
       } catch (...) {
           fprintf(stderr, "Exception during GPU memory allocation\n");
           g_cuda_enabled = false;
           if (d_points) {
               cudaFree(d_points);
               d_points = nullptr;
           }
       }
   }
  
   printf("creating clusters...\n");
  
   // Start clustering timing
   auto clustering_start = high_resolution_clock::now();
  
   // Optimize thread count for system
   int thread_count = min(getOptimalThreadCount(), num_cl);
   printf("Using %d threads for processing\n", thread_count);
  
   // Use a counter with mutex for thread work distribution
   atomic<int> cluster_counter(0);
  
   vector<thread> threads;
   for(int t = 0; t < thread_count; t++) {
       threads.push_back(thread([&]() {
           while(true) {
               int i = cluster_counter.fetch_add(1);
              
               if (i >= num_cl) break;
              
               int* idx_points = new int[N];
               for(int j=0; j<N; j++)
                   idx_points[j] = j;
              
               try {
                   create_clusters(points, idx_points, 0, N-1, G, minsize_cl, locks, max_mst_degree, d_points);
                   printf("end cluster %d\n", i);
               } catch (const exception& e) {
                   fprintf(stderr, "Exception in cluster %d: %s\n", i, e.what());
               } catch (...) {
                   fprintf(stderr, "Unknown exception in cluster %d\n", i);
               }
              
               delete[] idx_points;
           }
       }));
   }
  
   // Wait for all threads to complete
   for(auto& t : threads) {
       t.join();
   }
  
   // Calculate clustering time
   auto clustering_end = high_resolution_clock::now();
   timing.clustering_time = duration_cast<duration<double>>(clustering_end - clustering_start).count();
  
   // Free GPU memory if allocated
   if (d_points != nullptr) {
       cudaFree(d_points);
   }
  
   printf("sorting...\n");
  
   // Start sort timing
   auto sort_start = high_resolution_clock::now();
  
   sort_edges(G);
  
   // Calculate sort time
   auto sort_end = high_resolution_clock::now();
   timing.sort_time = duration_cast<duration<double>>(sort_end - sort_start).count();
  
   print_stats_graph(G);
  
   // Calculate overall time
   auto overall_end = high_resolution_clock::now();
   timing.total_time = duration_cast<duration<double>>(overall_end - overall_start).count();
  
   // Print timing information
   printf("\n--- Timing Information ---\n");
   printf("Total execution time: %.2f seconds\n", timing.total_time);
   printf("Clustering time: %.2f seconds (%.1f%%)\n",
          timing.clustering_time,
          (timing.clustering_time / timing.total_time) * 100);
   printf("Edge sorting time: %.2f seconds (%.1f%%)\n",
          timing.sort_time,
          (timing.sort_time / timing.total_time) * 100);
   printf("MST creation time: %.2f seconds\n", timing.mst_time);
  
   if (g_cuda_enabled) {
       printf("GPU computation time: %.2f seconds (%.1f%%) in %d calls\n",
              timing.gpu_time,
              (timing.gpu_time / timing.total_time) * 100,
              timing.gpu_calls);
   }
  
   printf("CPU computation time: %.2f seconds (%.1f%%) in %d calls\n",
          timing.cpu_time,
          (timing.cpu_time / timing.total_time) * 100,
          timing.cpu_calls);
  
   return G;
}


int main(int argc, char** argv) {
   auto program_start = high_resolution_clock::now(); // Start overall program timing
  
   // For reproducibility
   srand(42);
  
   // Initialize CUDA with error handling
   int deviceCount;
   cudaError_t err = cudaGetDeviceCount(&deviceCount);
   if (err != cudaSuccess) {
       fprintf(stderr, "CUDA initialization error: %s\n", cudaGetErrorString(err));
       fprintf(stderr, "Continuing with CPU-only implementation\n");
       g_cuda_enabled = false;
   }
  
   if (g_cuda_enabled && deviceCount > 0) {
       // Select the first CUDA device
       err = cudaSetDevice(0);
       if (err != cudaSuccess) {
           fprintf(stderr, "Error selecting CUDA device: %s\n", cudaGetErrorString(err));
           g_cuda_enabled = false;
       } else {
           // Print device properties
           cudaDeviceProp prop;
           err = cudaGetDeviceProperties(&prop, 0);
           if (err != cudaSuccess) {
               fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
           } else {
               printf("Using GPU: %s\n", prop.name);
               printf("Total GPU memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
               printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
               printf("Maximum grid dimensions: %d x %d x %d\n",
                     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
           }
       }
   } else {
       printf("No CUDA-capable devices found. Using CPU implementation.\n");
       g_cuda_enabled = false;
   }
  
   // Original argument handling
   int N, Dim;
   string file_dataset(argv[1]);
   int minsize_cl = atoi(argv[2]);
   int num_cl = atoi(argv[3]);
   int max_mst_degree = 3;
   string file_graph(argv[4]);
  
   printf("\n***************************\n");
   printf("MIN SZ CL:\t%d\n", minsize_cl);
   printf("NUMBER CL:\t%d\n", num_cl);
   printf("***************************\n\n");


   auto data_load_start = high_resolution_clock::now(); // Start data loading timing
   Matrix<float> points = read_fvecs(file_dataset, N, Dim);
   auto data_load_end = high_resolution_clock::now(); // End data loading timing
  
   double data_load_time = duration_cast<duration<double>>(data_load_end - data_load_start).count();
   printf("base read (%d,%d) ... took %.2f seconds\n", N, Dim, data_load_time);
  
   Graph nngraph = HCNNG_create_graph(points, Dim, num_cl, minsize_cl, max_mst_degree);
  
   auto write_start = high_resolution_clock::now(); // Start graph writing timing
   write_graph(file_graph, nngraph);
   auto write_end = high_resolution_clock::now(); // End graph writing timing
  
   double write_time = duration_cast<duration<double>>(write_end - write_start).count();
   printf("Writing graph took %.2f seconds\n", write_time);
  
   // Cleanup CUDA if enabled
   if (g_cuda_enabled) {
       cudaDeviceReset();
   }
  
   // Calculate and print overall program execution time
   auto program_end = high_resolution_clock::now();
   double program_time = duration_cast<duration<double>>(program_end - program_start).count();
  
   printf("\n--- Complete Program Timing ---\n");
   printf("Total program execution time: %.2f seconds\n", program_time);
   printf("Data loading time: %.2f seconds (%.1f%%)\n",
          data_load_time, (data_load_time / program_time) * 100);
   printf("Graph creation time: %.2f seconds (%.1f%%)\n",
          timing.total_time, (timing.total_time / program_time) * 100);
   printf("Graph writing time: %.2f seconds (%.1f%%)\n",
          write_time, (write_time / program_time) * 100);
  
   return 0;
}

