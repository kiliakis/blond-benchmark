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

__global__ void linear_interp_kick(const double * input,
                                   double * output,
                                   const double * voltage_array,
                                   const double * bin_centers,
                                   const int bins,
                                   const int n,
                                   const double acc_kick)
{

    const double center0 = bin_centers[0];
    const double inv_bin_width = (bins - 1) /
                                 (bin_centers[bins - 1] - center0);

    extern __shared__ double sh_mem[];
    double *sh_volt_kick = sh_mem;
    double *sh_factor = &sh_mem[bins - 1];

    for (size_t i = threadIdx.x;
            i < bins - 1;
            i += blockDim.x)
    {
        sh_volt_kick[i] = (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        sh_factor[i] = voltage_array[i] - bin_centers[i] * sh_volt_kick[i] + acc_kick;
    }
    __syncthreads();


    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n;
            i += blockDim.x * gridDim.x)
    {
        unsigned bin = (unsigned) floor((input[i] - center0) * inv_bin_width);
        if (bin < bins - 1)
            output[i] += input[i] * sh_volt_kick[bin] + sh_factor[bin];
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
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dE, dt;
    vector<double> voltage, edges, bin_centers;
    double cut_left, cut_right, acc_kick;

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

    thrust::device_vector<double> dev_dE = dE;
    thrust::device_vector<double> dev_dt = dt;
    thrust::device_vector<double> dev_voltage = voltage;
    thrust::device_vector<double> dev_bin_centers = bin_centers;

    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        linear_interp_kick <<< blocks, threads, 2 * (n_slices - 1)*sizeof(double) >>> (
            thrust::raw_pointer_cast(dev_dt.data()),
            thrust::raw_pointer_cast(dev_dE.data()),
            thrust::raw_pointer_cast(dev_voltage.data()),
            thrust::raw_pointer_cast(dev_bin_centers.data()),
            n_slices, n_particles, acc_kick);
        cudaThreadSynchronize();
    }

    auto end = chrono::high_resolution_clock::now();
    thrust::copy(dev_dE.begin(), dev_dE.end(), dE.begin());
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("interp_kick_gpu_v8\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0) / n_particles);
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}