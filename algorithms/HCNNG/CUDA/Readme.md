build - 
    nvcc hcnng.cu -o hcnng -std=c++17 -O3

run -
    ./hcnng ../sift_base.fvecs 1000 20 hcnng_sift.ivecs
