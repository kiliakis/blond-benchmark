#include <stdlib.h>
#include <stdio.h>
// #include "synchrotron_radiation.h"
#include "utils.h"
#include "cuda_utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <curand.h>
#include <cuda.h>

using namespace std;


int random_generator(double *rand_array, const int n, const double mean,
                     const double std)
{

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormalDouble(gen, rand_array, n, mean, std);
    curandDestroyGenerator(gen);
    return 0;
}

__global__ void synchrotron_radiation_full(double * beam_dE,
        const double *rand_array,
        const double const_synch_rad,
        const int n_particles)
{

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n_particles;
            i += blockDim.x * gridDim.x) {
        beam_dE[i] = rand_array[i] + const_synch_rad * beam_dE[i];
    }
}

int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_particles = 1000000;
    const int n_kicks = 1;
    int blocks = 512;
    int threads = 512;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) blocks = atoi(argv[3]);
    if (argc > 4) threads = atoi(argv[4]);

    // initialize variables
    vector<double> dE, dt;
    double U0, sigma_dE, tau_z, energy;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);
    U0 = 754257950.345;
    sigma_dE = 0.00142927197106;
    tau_z = 232.014940939;
    energy = 175000000000.0;

    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 1.0 - 2.0 / tau_z;

    thrust::device_vector<double> d_dE = dE;
    double *d_dE_ptr = thrust::raw_pointer_cast(d_dE.data());
    double *d_rand_array;
    cudaMalloc((void **)&d_rand_array, n_particles * sizeof(double));
    // main loop
    auto start = chrono::high_resolution_clock::now();

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    for (int i = 0; i < n_turns; ++i) {
        // random_generator(d_rand_array, n_particles, 0.0, 1.0);
        curandGenerateNormalDouble(gen, d_rand_array, n_particles,
                                   -U0, const_quantum_exc);
        cudaThreadSynchronize();
        synchrotron_radiation_full <<< blocks, threads>>>(
            d_dE_ptr, d_rand_array, const_synch_rad, n_particles);
        cudaThreadSynchronize();

    }
    curandDestroyGenerator(gen);

    auto end = chrono::high_resolution_clock::now();
    thrust::copy(d_dE.begin(), d_dE.end(), dE.begin());

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("sync_rad_gpu_v9\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0) / n_particles);

    // papiprof->stop_counters();
    // papiprof->report_timing();

    return 0;
}