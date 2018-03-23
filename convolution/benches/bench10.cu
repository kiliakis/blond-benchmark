#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "cuda_utils.h"

#include <algorithm>

using namespace std;

typedef float TYPE;

__global__ void convolution(const TYPE *signal, const int signalLen,
                            const TYPE *kernel, const int kernelLen,
                            TYPE *result)
{

    for (int t =  blockIdx.x * blockDim.x + threadIdx.x;
            t < signalLen * kernelLen;
            t += blockDim.x * gridDim.x) {
        int i = t / kernelLen;
        int j = t % kernelLen;
        atomicAdd(&result[i + j], signal[i] * kernel[j]);
    }

}

template <typename T>
void print_vector(vector<T> &v) {
    for (auto &i : v) cout << i << "\n";
}

int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_signal = 1000;
    int n_kernel = 1000;
    int blocks = 512;
    int threads = 64;
    // int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_signal = atoi(argv[2]);
    if (argc > 3) n_kernel = atoi(argv[3]);
    if (argc > 4) blocks = atoi(argv[4]);
    if (argc > 5) threads = atoi(argv[5]);

    // setup random engine
    default_random_engine gen;
    // uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<int> signal, kernel;
    vector<int> result;
    signal.resize(n_signal);
    kernel.resize(n_kernel);
    result.resize(n_signal + n_kernel - 1);

    for (int i = 0; i < n_signal; ++i) {
        // signal[i] = d(gen);
        signal[i] = i + 1;
    }

    for (int i = 0; i < n_kernel; ++i) {
        // kernel[i] = d(gen);
        kernel[i] = n_signal + i + 1;
    }

    thrust::device_vector<TYPE> d_signal(result.size(), 0.);
    thrust::device_vector<TYPE> d_kernel(result.size(), 0.);
    thrust::copy(signal.begin(), signal.end(), d_signal.begin());
    thrust::copy(kernel.begin(), kernel.end(), d_kernel.begin());
    thrust::device_vector<TYPE> d_result(result.size());


    TYPE *d_signal_ptr = thrust::raw_pointer_cast(d_signal.data());
    TYPE *d_kernel_ptr = thrust::raw_pointer_cast(d_kernel.data());
    TYPE *d_result_ptr = thrust::raw_pointer_cast(d_result.data());


    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("convolution");
    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        convolution <<< blocks, threads>>>(d_signal_ptr, n_signal,
                                           d_kernel_ptr, n_kernel,
                                           d_result_ptr);
        cudaThreadSynchronize();
    }
    auto end = chrono::high_resolution_clock::now();
    thrust::copy(d_result.begin(), d_result.end(), result.begin());

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("convolution_gpu_v10\ttime(ms)\t%d\t0\t1\n", duration);
    // print_vector(result);
    printf("result: %lf\n", accumulate(result.begin(), result.end(), 0.0) / (n_signal + n_kernel - 1));
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}