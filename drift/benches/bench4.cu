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

__global__ void drift(double * beam_dt,
                      const double * beam_dE,
                      // const char * solver,
                      const double T0,
                      const double length_ratio,
                      const int alpha_order,
                      const double eta_zero,
                      const double eta_one,
                      const double eta_two,
                      const double beta,
                      const double energy,
                      const int n_macroparticles)
{

    if ( alpha_order == 0 ) {

        const double coeff = T0 * length_ratio * eta_zero / (beta * beta * energy);
        for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
                i < n_macroparticles;
                i += blockDim.x * gridDim.x) {
            beam_dt[i] += coeff * beam_dE[i];
        }
    } else if ( alpha_order == 1 ) {
        const double T = T0 * length_ratio;
        const double eta0 = eta_zero / (beta * beta * energy);
        for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
                i < n_macroparticles;
                i += blockDim.x * gridDim.x) {
            beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
        }
    } else if (alpha_order == 2) {
        const double T = T0 * length_ratio;
        const double coeff = 1. / (beta * beta * energy);
        const double eta0 = eta_zero * coeff;
        const double eta1 = eta_one * coeff * coeff;

        for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
                i < n_macroparticles;
                i += blockDim.x * gridDim.x) {
            beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                     - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
        }
    } else {
        const double T = T0 * length_ratio;
        const double coeff = 1. / (beta * beta * energy);
        const double eta0 = eta_zero * coeff;
        const double eta1 = eta_one * coeff * coeff;
        const double eta2 = eta_two * coeff * coeff * coeff;

        for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
                i < n_macroparticles;
                i += blockDim.x * gridDim.x) {
            beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                     - eta1 * beam_dE[i] * beam_dE[i]
                                     - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
        }
    }
}



int main(int argc, char const * argv[])
{
    int n_turns = 5000;
    int n_particles = 1000000;
    int alpha_order = 0;
    int blocks = 512;
    int threads = 64;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) alpha_order = atoi(argv[3]);
    if (argc > 4) blocks = atoi(argv[4]);
    if (argc > 5) threads = atoi(argv[5]);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dE, dt;
    double T0, length_ratio, eta0, eta1, eta2;
    double beta, energy;

    dE.resize(n_particles); dt.resize(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        dE[i] = 10e6 * d(gen);
        dt[i] = 10e-6 * d(gen);
    }
    T0 = d(gen);
    length_ratio = d(gen);
    eta0 = d(gen); eta1 = d(gen); eta2 = d(gen);
    beta = d(gen); energy = d(gen);
    const char *solver = alpha_order > 0 ? "f" : "s";
    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("drift");

    thrust::device_vector<double> dev_dE = dE;
    thrust::device_vector<double> dev_dt = dt;
    char *dev_solver;
    cudaMalloc(&dev_solver, sizeof(char));
    cudaMemcpy(dev_solver, solver, sizeof(char), cudaMemcpyHostToDevice);


    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        drift <<< blocks, threads>>>(thrust::raw_pointer_cast(dev_dt.data()),
                                     thrust::raw_pointer_cast(dev_dE.data()),
                                     T0, length_ratio, alpha_order, eta0,
                                     eta1, eta2, beta, energy,
                                     n_particles);
        cudaThreadSynchronize();
    }
    auto end = chrono::high_resolution_clock::now();

    thrust::copy(dev_dt.begin(), dev_dt.end(), dt.begin());

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("drift_gpu_v0\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dt: %lf\n", accumulate(dt.begin(), dt.end(), 0.0) / (n_particles));
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}