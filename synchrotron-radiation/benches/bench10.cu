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
#include <curand_kernel.h>

using namespace std;


__global__ void setup_kernel(curandStateMRG32k3a *state, unsigned long long seed)
{
    ssize_t id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}


// __device__ double generate(curandStateMRG32k3a *globalState, ssize_t ind)
// {
//     //copy state to local mem
//     curandState localState = globalState[ind];
//     //apply uniform distribution with calculated random
//     double rndval = curand_normal_double(&localState);
//     //update state
//     globalState[ind] = localState;
//     //return value
//     return rndval;
// }

__global__ void synchrotron_radiation_full(double * beam_dE,
        const double U0,
        const int n_particles,
        const double sigma_dE,
        const double tau_z,
        const double energy,
        const int n_kicks,
        curandStateMRG32k3a *states)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;
    const ssize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = states[idx];

    for (size_t i = idx; i < n_particles; i += blockDim.x * gridDim.x) {
        beam_dE[i] += const_quantum_exc * curand_normal_double(&localState)
                      - const_synch_rad * beam_dE[i] - U0;
    }
    states[idx] = localState;
}

int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_particles = 1000000;
    const int n_kicks = 1;
    int blocks = 256;
    int threads = 256;

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

    thrust::device_vector<double> d_dE = dE;
    double *d_dE_ptr = thrust::raw_pointer_cast(d_dE.data());
    // double *d_rand_array;
    // cudaMalloc((void **)&d_rand_array, n_particles * sizeof(double));
    // main loop
    // curandState_t *d_states;
    curandStateMRG32k3a *d_states;
    cudaMalloc((void **)&d_states, blocks * threads * sizeof(*d_states));
    setup_kernel <<< blocks, threads>>> (d_states, 1234ULL);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n_turns; ++i) {
        // random_generator(d_rand_array, n_particles, 0.0, 1.0);
        // curandGenerateNormalDouble(gen, d_rand_array, n_particles, 0.0, 1.0);
        // cudaThreadSynchronize();
        synchrotron_radiation_full <<< blocks, threads>>>(
            d_dE_ptr, U0, n_particles,
            sigma_dE, tau_z, energy, n_kicks, d_states);
        cudaThreadSynchronize();

    }
    // curandDestroyGenerator(gen);

    auto end = chrono::high_resolution_clock::now();
    thrust::copy(d_dE.begin(), d_dE.end(), dE.begin());

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("sync_rad_gpu_v10\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0) / n_particles);

    // papiprof->stop_counters();
    // papiprof->report_timing();

    return 0;
}