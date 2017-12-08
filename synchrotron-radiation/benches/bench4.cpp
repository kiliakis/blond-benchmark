#include <stdlib.h>
#include <stdio.h>
// #include "synchrotron_radiation.h"
#include "utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <string>
// #include <PAPIProf.h>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mkl.h>

using namespace std;


void synchrotron_radiation_full_mkl(float * __restrict__ beam_dE,
        const float U0,
        const int n_macroparticles, const float sigma_dE,
        const float tau_z, const float energy,
        const int n_kicks)
{
    const float const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const float const_synch_rad = 2.0 / tau_z;

    float *rand_array = (float *) malloc (sizeof(float) * n_macroparticles);

    #pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int count = (n_macroparticles + threads - 1) / threads;
        const int computeCount = std::min(count, n_macroparticles - tid * count);
        // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = 0;

        int status;
        VSLStreamStatePtr stream;
        status = vslNewStream(&stream, VSL_BRNG_MT19937, seed);
        if (status != VSL_STATUS_OK){
            printf("[%d] Error in %s\n", tid, "vslNewStream");
        }
        // Methods
        // VSL_RNG_METHOD_GAUSSIAN_ICDF
        // VSL_RNG_METHOD_GAUSSIAN_BOXMULLER
        // VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
        status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                               computeCount, &rand_array[tid * count],
                               0.0, 1.0);
        if (status != VSL_STATUS_OK){
            printf("[%d] Error in %s\n", tid, "vdRngGaussian");
        }

        #pragma omp for
        for (int i = tid * count; i < tid * count + computeCount; i++) {
            beam_dE[i] = beam_dE[i] + const_quantum_exc * rand_array[i]
                         - const_synch_rad * beam_dE[i] - U0;
        }
    }
    free(rand_array);
}

int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_particles = 1000000;
    const int n_kicks = 1;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    // if (argc > 3) n_kicks = 1;
    if (argc > 3) n_threads = atoi(argv[3]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<float> d(0.0, 1.0);

    // initialize variables
    vector<float> dE, dt;
    float U0, sigma_dE, tau_z, energy;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);

    U0 = 754257950.345;
    sigma_dE = 0.00142927197106;
    tau_z = 232.014940939;
    energy = 175000000000.0;

    // auto papiprof = new PAPIProf();
    // main loop
    // papiprof->start_counters("synchrotron_radiation");
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < n_turns; ++i) {
        synchrotron_radiation_full_mkl(dE.data(), U0, n_particles,
                                      sigma_dE, tau_z, energy, n_kicks);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("sync_rad_v4\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0)/n_particles);

    // papiprof->stop_counters();
    // papiprof->report_timing();

    return 0;
}