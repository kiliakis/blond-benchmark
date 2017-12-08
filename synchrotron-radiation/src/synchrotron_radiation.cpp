#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "synchrotron_radiation.h"
#include <math.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
// #include <mkl.h>

// #ifdef USE_BOOST

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
// using namespace boost;

// #else

#include <random>
// using namespace std;

// #endif

// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
void synchrotron_radiation_full_v0(double * __restrict__ beam_dE,
        const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;


    std::normal_distribution<> dist(0.0, 1.0);
    #pragma omp parallel
    {
        // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = 0;
        std::mt19937 rng;
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

void synchrotron_radiation_full_v4(float * __restrict__ beam_dE,
        const float U0,
        const int n_macroparticles, const float sigma_dE,
        const float tau_z, const float energy,
        const int n_kicks)
{
    const float const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const float const_synch_rad = 2.0 / tau_z;


    std::normal_distribution<float> dist(0.0, 1.0);
    #pragma omp parallel
    {
        // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = 0;
        std::mt19937 rng;
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

void synchrotron_radiation_full_v1(double * __restrict__ beam_dE,
        const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;


    boost::normal_distribution<> dist(0.0, 1.0);
    #pragma omp parallel
    {
        // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = 0;
        boost::mt19937 rng;
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

void synchrotron_radiation_full_v3(float * __restrict__ beam_dE,
        const float U0,
        const int n_macroparticles, const float sigma_dE,
        const float tau_z, const float energy,
        const int n_kicks)
{
    const float const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const float const_synch_rad = 2.0 / tau_z;


    boost::normal_distribution<float> dist(0.0, 1.0);
    #pragma omp parallel
    {
        // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = 0;
        boost::mt19937 rng;
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


void synchrotron_radiation_full_v2(double * __restrict__ beam_dE,
        const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        const int n_kicks)
{
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;
    // double *array = (double *) malloc(n_macroparticles * sizeof(double));
    const int STEP = 64;
    boost::normal_distribution<> dist(0.0, 1.0);
    #pragma omp parallel
    {
        double temp[STEP];
        // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = 0;
        boost::mt19937 rng;
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

