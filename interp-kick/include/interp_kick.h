#pragma once


void linear_interp_kick_v0(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);

void linear_interp_kick_v1(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);

void linear_interp_kick_v2(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);


void linear_interp_kick_v3(
    const float * __restrict__ beam_dt,
    float * __restrict__ beam_dE,
    const float * __restrict__ voltage_array,
    const float * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick);

void linear_interp_kick_v4(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);

void linear_interp_kick_v5(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick);

void linear_interp_kick_v6(
    const float * __restrict__ beam_dt,
    float * __restrict__ beam_dE,
    const float * __restrict__ voltage_array,
    const float * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick);