#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "synchrotron_radiation.h"
#include <math.h>
#include <omp.h>
#include <chrono>
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
extern "C" void synchrotron_radiation_full(double * __restrict__ beam_dE, const double U0,
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
        mt19937_64 rng;
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