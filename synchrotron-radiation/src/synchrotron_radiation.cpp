#include <string.h>     // memset()
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "synchrotron_radiation.h"
#include <math.h>
#include <random>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


// This function calculates and applies only the synchrotron radiation damping term
/*
extern "C" void synchrotron_radiation(double * __restrict__ beam_dE,
                                      const double U0,
                                      const int n_macroparticles,
                                      const double tau_z,
                                      const int n_kicks) {

    // SR damping constant
    const double const_synch_rad = 2.0 / tau_z;

    for (int j = 0; j < n_kicks; j++) {
        // SR damping term due to energy spread
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] -= const_synch_rad * beam_dE[i];

        // Average energy change due to SR
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] -= U0;
    }
}
*/

// This function calculates and applies synchrotron radiation damping
// and quantum excitation terms
// Random number generator for the quantum excitation term
boost::mt19937_64 *rng = new boost::mt19937_64();
boost::normal_distribution<> distribution(0.0, 1.0);
boost::variate_generator< boost::mt19937_64, boost::normal_distribution<> > dist(*rng, distribution);

extern "C" void synchrotron_radiation_full(double * __restrict__ beam_dE, const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        double * __restrict__ random_array,
        const int n_kicks) {

    // Quantum excitation  and synchrotron radiation constants
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;
    // Setting a seed for the random generator
    rng->seed(std::random_device{}());

    // for (int j = 0; j < n_kicks; j++) {

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        // Applying quantum excitation term, SR damping term due to energy spread, and average energy change due to SR
        beam_dE[i] += (const_quantum_exc * dist() - const_synch_rad * beam_dE[i] - U0);
    }
    // }
}


// Random number generator for the quantum excitation term
std::random_device rd;
std::mt19937_64 gen(rd());
std::normal_distribution<> d(0.0, 1.0);

// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
extern "C" void synchrotron_radiation_full(double * __restrict__ beam_dE, const double U0,
        const int n_macroparticles, const double sigma_dE,
        const double tau_z, const double energy,
        double * __restrict__ random_array,
        const int n_kicks) {

    // Quantum excitation constant
    const double const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    const double const_synch_rad = 2.0 / tau_z;

    // for (int j=0; j<n_kicks; j++){

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        // Applying quantum excitation term, SR damping term due to energy spread, and average energy change due to SR
        beam_dE[i] += (const_quantum_exc * d(gen) - const_synch_rad * beam_dE[i] - U0);
    }
    // }
}