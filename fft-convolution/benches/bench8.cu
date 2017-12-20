#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <cufft.h>
#include "cuda_utils.h"
#include "fftconvolve.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;

template <typename T>
void print_vector(vector<T> &v) {
    for (auto &i : v) cout << i << "\n";
}

template <typename T>
void print_vector(T *v, int n) {
    for (int i = 0; i < n; i++) cout << v[i] << "\n";
}


int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_signal = 1000;
    int n_kernel = 1000;
    // int blocks = 512;
    // int threads = 1024;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_signal = atoi(argv[2]);
    if (argc > 3) n_kernel = atoi(argv[3]);
    // if (argc > 4) blocks = atoi(argv[4]);
    // if (argc > 5) threads = atoi(argv[5]);
    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> signal, kernel;
    vector<double> result;
    signal.resize(n_signal);
    kernel.resize(n_kernel);
    result.resize(n_signal + n_kernel - 1);

    for (int i = 0; i < n_signal; ++i) {
        signal[i] = d(gen);
    }

    for (int i = 0; i < n_kernel; ++i) {
        kernel[i] = d(gen);
    }

    thrust::device_vector<double> d_signal(result.size(), 0.);
    thrust::device_vector<double> d_kernel(result.size(), 0.);
    thrust::copy(signal.begin(), signal.end(), d_signal.begin());
    thrust::copy(kernel.begin(), kernel.end(), d_kernel.begin());
    thrust::device_vector<double> d_result(result.size());


    double *d_signal_ptr = thrust::raw_pointer_cast(d_signal.data());
    double *d_kernel_ptr = thrust::raw_pointer_cast(d_kernel.data());
    double *d_result_ptr = thrust::raw_pointer_cast(d_result.data());

    // thrust::device_vector<cufftDoubleComplex> d_out(result.size() / 2 + 1);
    // cufftDoubleComplex *d_out_ptr = thrust::raw_pointer_cast(d_out.data());

    cufftHandle fwplan, bwplan;

    if (cufftPlan1d(&fwplan, result.size(), CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    if (cufftPlan1d(&bwplan, result.size(), CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }


    convolve_real_no_memcpy_v1(d_signal_ptr, n_signal,
                               d_kernel_ptr, n_kernel,
                               d_result_ptr,
                               fwplan, bwplan);
    cudaThreadSynchronize();

    // // main loop
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n_turns; ++i) {
        convolve_real_no_memcpy_v1(d_signal_ptr, n_signal,
                                   d_kernel_ptr, n_kernel,
                                   d_result_ptr,
                                   fwplan, bwplan);
        cudaThreadSynchronize();
    }
    cufftDestroy(fwplan); cufftDestroy(bwplan);

    auto end = chrono::high_resolution_clock::now();
    thrust::copy(d_result.begin(), d_result.end(), result.begin());

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("fft_convolution_gpu_v8\ttime(ms)\t%d\t0\t1\n", duration);
    printf("result: %lf\n", accumulate(result.begin(), result.end(), 0.0) / (n_signal + n_kernel - 1));
    return 0;
}