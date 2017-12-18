#include <stdlib.h>
#include <stdio.h>
#include "cuda_utils.h"
#include <vector>
#include <random>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;

__global__ void kick(const double * beam_dt,
                     double * beam_dE,
                     const int n_rf,
                     const double * voltage,
                     const double * omega_rf,
                     const double * phi_rf,
                     const int n_macroparticles,
                     const double acc_kick)
{
    extern __shared__ double sh[];
    double *sh_voltage = sh;
    double *sh_omega_rf = &sh_voltage[n_rf];
    double *sh_phi_rf = &sh_voltage[n_rf];

    for (int i = threadIdx.x; i < n_rf; i += blockDim.x) {
        sh_voltage[i] = voltage[i];
        sh_omega_rf[i] = omega_rf[i];
        sh_phi_rf[i] = phi_rf[i];
    }

    for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
            i < n_macroparticles;
            i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_rf; j++) {
            beam_dE[i] = beam_dE[i] + sh_voltage[j] *
                         sin(beam_dt[i] * sh_omega_rf[j] + sh_phi_rf[j]);
        }
    }

    for (int i =  blockIdx.x * blockDim.x + threadIdx.x;
            i < n_macroparticles;
            i += blockDim.x * gridDim.x) {
        beam_dE[i] = beam_dE[i] + acc_kick;
    }
}


int main(int argc, char const *argv[])
{
    int n_turns = 5000;
    int n_particles = 1000000;
    int n_rf = 4;
    int blocks = 512;
    int threads = 64;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) n_rf = atoi(argv[3]);
    if (argc > 4) blocks = atoi(argv[4]);
    if (argc > 5) threads = atoi(argv[5]);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dE, dt;
    vector<double> voltage, omega_rf, phi_rf;
    double acc_kick;

    dE.resize(n_particles); dt.resize(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        dE[i] = 10e6 * d(gen);
        dt[i] = 10e-6 * d(gen);
    }

    voltage.resize(n_rf);
    omega_rf.resize(n_rf);
    phi_rf.resize(n_rf);
    for (int i = 0; i < n_rf; ++i) {
        voltage[i] = d(gen);
        omega_rf[i] = d(gen);
        phi_rf[i] = d(gen);
    }
    acc_kick = 10e6 * d(gen);

    thrust::device_vector<double> dev_dE = dE;
    thrust::device_vector<double> dev_dt = dt;
    thrust::device_vector<double> dev_voltage = voltage;
    thrust::device_vector<double> dev_omega_rf = omega_rf;
    thrust::device_vector<double> dev_phi_rf = phi_rf;
    // main loop
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n_turns; ++i) {
        kick <<< blocks, threads, 3 * n_rf * sizeof(double) >>> (
            thrust::raw_pointer_cast(dev_dt.data()),
            thrust::raw_pointer_cast(dev_dE.data()),
            n_rf,
            thrust::raw_pointer_cast(dev_voltage.data()),
            thrust::raw_pointer_cast(dev_omega_rf.data()),
            thrust::raw_pointer_cast(dev_phi_rf.data()),
            n_particles, acc_kick);
        cudaThreadSynchronize();

    }
    auto end = chrono::high_resolution_clock::now();

    thrust::copy(dev_dE.begin(), dev_dE.end(), dE.begin());
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("kick_gpu_v0\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0) / n_particles);

    return 0;
}