#include <stdlib.h>
#include <stdio.h>
#include "histogram.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
// #include <PAPIProf.h>
#include <omp.h>
#include <string>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;

// Global memory atomics histogram
__global__ void histogram(const double * input,
                          int * output,
                          const double cut_left,
                          const double cut_right,
                          const int bins,
                          const int n)
{
    const double inv_bin_width = bins / (cut_right - cut_left);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < bins;
            i += blockDim.x * gridDim.x) {
        output[i] = 0;
    }

    __syncthreads();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n;
            i += blockDim.x * gridDim.x)
    {
        if ((input[i] < cut_left) || (input[i] > cut_right))
            continue;
        int bin = (int) ((input[i] - cut_left) * inv_bin_width);
        atomicAdd(&output[bin], 1);
    }

}


int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_particles = 1000000;
    int n_slices = 1000;
    int blocks = 512;
    int threads = 1024;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) n_slices = atoi(argv[3]);
    if (argc > 4) blocks = atoi(argv[4]);
    if (argc > 5) threads = atoi(argv[5]);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dt, dE;
    vector<int> profile;
    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);
    double cut_left, cut_right;
    profile.resize(n_slices);

    cut_left = 1.05 * (*min_element(dt.begin(), dt.end()));
    cut_right = 0.95 * (*max_element(dt.begin(), dt.end()));
    if (cut_left > cut_right) swap(cut_left, cut_right);

    thrust::device_vector<double> dev_dt = dt;
    thrust::device_vector<int> dev_profile = profile;

    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        // thrust::fill(dev_profile.begin(), dev_profile.end(), 0);
        histogram <<< blocks, threads>>>(
            thrust::raw_pointer_cast(dev_dt.data()),
            thrust::raw_pointer_cast(dev_profile.data()),
            cut_left, cut_right,
            n_slices, n_particles);
        cudaThreadSynchronize();
        
    }

    auto end = chrono::high_resolution_clock::now();

    thrust::copy(dev_profile.begin(), dev_profile.end(), profile.begin());

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("histogram_gpu_v0\ttime(ms)\t%d\t0\t1\n", duration);
    printf("profile: %d\n", accumulate(profile.begin(), profile.end(), 0) / n_slices);

    return 0;
}