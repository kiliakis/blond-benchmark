#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "cuda_utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;


__global__ void precalc_interp_kick(const float * voltage_array,
                                    const float * bin_centers,
                                    float * volt_kick,
                                    float * factor,
                                    const int bins,
                                    const float acc_kick)
{
    const float inv_bin_width = (bins - 1) / (bin_centers[bins - 1] - bin_centers[0]);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < bins - 1;
            i += blockDim.x * gridDim.x) {
        volt_kick[i] = (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        factor[i] = voltage_array[i] - bin_centers[i] * volt_kick[i] + acc_kick;
    }
}


__global__ void linear_interp_kick(const float * input,
                                   float * output,
                                   const float * volt_kick,
                                   const float * factor,
                                   const float * bin_centers,
                                   const int bins,
                                   const int n,
                                   const float acc_kick)
{
    const float center0 = bin_centers[0];
    const float inv_bin_width = (bins - 1) / (bin_centers[bins - 1] - center0);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n;
            i += blockDim.x * gridDim.x) {
        unsigned bin = (unsigned) floor((input[i] - center0) * inv_bin_width);
        if (bin < bins - 1)
            output[i] += input[i] * volt_kick[bin] + factor[bin];
        else
            output[i] += acc_kick;

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
    uniform_real_distribution<float> d(0.0, 1.0);

    // initialize variables
    vector<float> dE, dt;
    vector<float> voltage, edges, bin_centers;
    float cut_left, cut_right, acc_kick;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);

    voltage.resize(n_slices);
    for (int i = 0; i < n_slices; ++i) {
        voltage[i] = d(gen);
    }
    cut_left = 1.05 * (*min_element(dt.begin(), dt.end()));
    cut_right = 0.95 * (*max_element(dt.begin(), dt.end()));

    // cut_left = dt[rand() % n_slices];
    // cut_right = dt[rand() % n_slices];
    acc_kick = 10e6 * d(gen);
    if (cut_left > cut_right) swap(cut_left, cut_right);

    edges.resize(n_slices);
    linspace(cut_left, cut_right, n_slices + 1, edges.data());

    bin_centers.resize(n_slices);
    for (int i = 0; i < n_slices; ++i) {
        bin_centers[i] = (edges[i] + edges[i + 1]) / 2.;
    }

    thrust::device_vector<float> d_dE = dE;
    thrust::device_vector<float> d_dt = dt;
    thrust::device_vector<float> d_voltage = voltage;
    thrust::device_vector<float> d_bin_centers = bin_centers;
    thrust::device_vector<float> d_volt_kick(n_slices - 1);
    thrust::device_vector<float> d_factor(n_slices - 1);

    float *d_dE_ptr = thrust::raw_pointer_cast(d_dE.data());
    float *d_dt_ptr = thrust::raw_pointer_cast(d_dt.data());
    float *d_bin_centers_ptr = thrust::raw_pointer_cast(d_bin_centers.data());
    float *d_voltage_ptr = thrust::raw_pointer_cast(d_voltage.data());
    float *d_volt_kick_ptr = thrust::raw_pointer_cast(d_volt_kick.data());
    float *d_factor_ptr = thrust::raw_pointer_cast(d_factor.data());

    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        precalc_interp_kick <<< (n_slices + 63) / 64, 64 >>> (
            d_voltage_ptr,
            d_bin_centers_ptr,
            d_volt_kick_ptr,
            d_factor_ptr,
            n_slices, acc_kick);
        linear_interp_kick <<< blocks, threads>>> (d_dt_ptr,
                d_dE_ptr,
                d_volt_kick_ptr,
                d_factor_ptr,
                d_bin_centers_ptr,
                n_slices, n_particles, acc_kick);
        cudaThreadSynchronize();
    }

    auto end = chrono::high_resolution_clock::now();
    thrust::copy(d_dE.begin(), d_dE.end(), dE.begin());
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("interp_kick_gpu_v10\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0) / n_particles);
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}