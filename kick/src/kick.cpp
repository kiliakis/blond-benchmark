
#include "sin.h"
#include "kick.h"
#include <cmath>


// std::sin
void kick_v0(const double * __restrict__ beam_dt,
             double * __restrict__ beam_dE, const int n_rf,
             const double * __restrict__ voltage,
             const double * __restrict__ omega_RF,
             const double * __restrict__ phi_RF,
             const int n_macroparticles,
             const double acc_kick)
{
    // KICK
    for (int j = 0; j < n_rf; j++)
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] + voltage[j]
                         * std::sin(omega_RF[j] * beam_dt[i] + phi_RF[j]);

    // SYNCHRONOUS ENERGY CHANGE
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] = beam_dE[i] + acc_kick;

}

// std::sin
void kick_v1(const float * __restrict__ beam_dt,
             float * __restrict__ beam_dE, const int n_rf,
             const float * __restrict__ voltage,
             const float * __restrict__ omega_RF,
             const float * __restrict__ phi_RF,
             const int n_macroparticles,
             const float acc_kick)
{
    // KICK
    for (int j = 0; j < n_rf; j++)
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] + voltage[j]
                         * std::sin(omega_RF[j] * beam_dt[i] + phi_RF[j]);

    // SYNCHRONOUS ENERGY CHANGE
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] = beam_dE[i] + acc_kick;

}


// fast_sin, check with or without vectorization
void kick_v2(const double * __restrict__ beam_dt,
             double * __restrict__ beam_dE, const int n_rf,
             const double * __restrict__ voltage,
             const double * __restrict__ omega_RF,
             const double * __restrict__ phi_RF,
             const int n_macroparticles,
             const double acc_kick)
{
    // KICK
    for (int j = 0; j < n_rf; j++)
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] + voltage[j]
                         * vdt::fast_sin(omega_RF[j] * beam_dt[i] + phi_RF[j]);

    // SYNCHRONOUS ENERGY CHANGE
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] = beam_dE[i] + acc_kick;

}


// fast_sin, with floats, check with or without vectorization
void kick_v3(const float * __restrict__ beam_dt,
             float * __restrict__ beam_dE, const int n_rf,
             const float * __restrict__ voltage,
             const float * __restrict__ omega_RF,
             const float * __restrict__ phi_RF,
             const int n_macroparticles,
             const float acc_kick)
{
    // KICK
    for (int j = 0; j < n_rf; j++)
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] = beam_dE[i] + voltage[j]
                         * vdt::fast_sinf(omega_RF[j] * beam_dt[i] + phi_RF[j]);

    // SYNCHRONOUS ENERGY CHANGE
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] = beam_dE[i] + acc_kick;

}

