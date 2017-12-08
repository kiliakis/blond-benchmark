#pragma once

void synchrotron_radiation_full_v0(double * __restrict__ beam_dE,
    const double U0,
    const int n_macroparticles, const double sigma_dE,
    const double tau_z, const double energy,
    const int n_kicks);

void synchrotron_radiation_full_v1(double * __restrict__ beam_dE,
    const double U0,
    const int n_macroparticles, const double sigma_dE,
    const double tau_z, const double energy,
    const int n_kicks);


void synchrotron_radiation_full_v2(double * __restrict__ beam_dE,
    const double U0,
    const int n_macroparticles, const double sigma_dE,
    const double tau_z, const double energy,
    const int n_kicks);

void synchrotron_radiation_full_v3(float * __restrict__ beam_dE,
    const float U0,
    const int n_macroparticles, const float sigma_dE,
    const float tau_z, const float energy,
    const int n_kicks);

void synchrotron_radiation_full_v4(float * __restrict__ beam_dE,
    const float U0,
    const int n_macroparticles, const float sigma_dE,
    const float tau_z, const float energy,
    const int n_kicks);

// void synchrotron_radiation_full_mkl(double * __restrict__ beam_dE,
//         const double U0,
//         const int n_macroparticles, const double sigma_dE,
//         const double tau_z, const double energy,
//         const int n_kicks);