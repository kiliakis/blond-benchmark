#pragma once

void fast_resonator_v0(
    double *__restrict__ impedanceReal,
    double *__restrict__ impedanceImag,
    const double *__restrict__ frequencies,
    const double *__restrict__ shunt_impedances,
    const double *__restrict__ Q_values,
    const double *__restrict__ resonant_frequencies,
    const int n_resonators,
    const int n_frequencies);

void fast_resonator_v1(
    double *__restrict__ impedanceReal,
    double *__restrict__ impedanceImag,
    const double *__restrict__ frequencies,
    const double *__restrict__ shunt_impedances,
    const double *__restrict__ Q_values,
    const double *__restrict__ resonant_frequencies,
    const int n_resonators,
    const int n_frequencies);

void fast_resonator_v2(
    double *__restrict__ impedanceReal,
    double *__restrict__ impedanceImag,
    const double *__restrict__ frequencies,
    const double *__restrict__ shunt_impedances,
    const double *__restrict__ Q_values,
    const double *__restrict__ resonant_frequencies,
    const int n_resonators,
    const int n_frequencies);
