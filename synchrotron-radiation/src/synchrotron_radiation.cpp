#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "synchrotron_radiation.h"
#include <math.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <mkl.h>

#ifdef USE_BOOST

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
using namespace boost;

#else

#include <random>
using namespace std;

#endif

// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
extern "C" void synchrotron_radiation_full_v0(double * __restrict__ beam_dE,
        const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;


    normal_distribution<> dist(0.0, 1.0);
    #pragma omp parallel
    {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng;
        rng.seed(seed + omp_get_thread_num());
        // printf("[%d] %lf\n",omp_get_thread_num(), dist(rng));
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            // Applying quantum excitation term, SR damping term due to energy spread, and average energy change due to SR
            beam_dE[i] += const_quantum_exc * dist(rng)
                          - const_synch_rad * beam_dE[i] - U0;
        }
    }
}

extern "C" void synchrotron_radiation_full_v1(double * __restrict__ beam_dE,
        const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;
    // double *array = (double *) malloc(n_macroparticles * sizeof(double));
    const int STEP = 64;
    normal_distribution<> dist(0.0, 1.0);
    #pragma omp parallel
    {
        double temp[STEP];
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        mt19937 rng;
        rng.seed(seed + omp_get_thread_num());
        // printf("[%d] %lf\n",omp_get_thread_num(), dist(rng));
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; ++j) temp[j] = dist(rng);

            for (int j = 0; j < loop_count; ++j)
                beam_dE[i + j] = beam_dE[i + j] + const_quantum_exc * temp[j]
                                 - const_synch_rad * beam_dE[i + j] - U0;

        }

    }
}



extern "C" void synchrotron_radiation_full_mkl(double * __restrict__ beam_dE,
        const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;

    double *rand_array = (double *) malloc (sizeof(double) * n_macroparticles);

    #pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int count = (n_macroparticles + threads - 1) / threads;
        const int computeCount = std::min(count, n_macroparticles - tid * count);
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();

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
        status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
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