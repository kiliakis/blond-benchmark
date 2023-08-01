#pragma once


void overlap_track_v0(double * __restrict__ dt,
                      double * __restrict__ dE,
                      double * __restrict__ profile,
                      const double * __restrict__ voltage_array,
                      const double * __restrict__ bin_centers,
                      const int n_slices,
                      const int n_particles,
                      const double acc_kick,
                      const char * __restrict__ solver,
                      const double T0,
                      const double length_ratio,
                      const double alpha_order,
                      const double eta_zero,
                      const double eta_one,
                      const double eta_two,
                      const double beta,
                      const double energy,
                      const double charge,
                      const double cut_left,
                      const double cut_right);



void overlap_track_v1(double * __restrict__ dt,
                      double * __restrict__ dE,
                      double * __restrict__ profile,
                      const double * __restrict__ voltage_array,
                      const double * __restrict__ bin_centers,
                      const int n_slices,
                      const int n_particles,
                      const double acc_kick,
                      const char * __restrict__ solver,
                      const double T0,
                      const double length_ratio,
                      const double alpha_order,
                      const double eta_zero,
                      const double eta_one,
                      const double eta_two,
                      const double beta,
                      const double energy,
                      const double charge,
                      const double cut_left,
                      const double cut_right);


void overlap_track_v2(double * __restrict__ dt,
                      double * __restrict__ dE,
                      double * __restrict__ profile,
                      const double * __restrict__ voltage_array,
                      const double * __restrict__ bin_centers,
                      const int n_slices,
                      const int n_particles,
                      const double acc_kick,
                      const char * __restrict__ solver,
                      const double T0,
                      const double length_ratio,
                      const double alpha_order,
                      const double eta_zero,
                      const double eta_one,
                      const double eta_two,
                      const double beta,
                      const double energy,
                      const double charge,
                      const double cut_left,
                      const double cut_right);


void overlap_track_v3(double * __restrict__ dt,
                      double * __restrict__ dE,
                      double * __restrict__ profile,
                      const double * __restrict__ voltage_array,
                      const double * __restrict__ bin_centers,
                      const int n_slices,
                      const int n_particles,
                      const double acc_kick,
                      const char * __restrict__ solver,
                      const double T0,
                      const double length_ratio,
                      const double alpha_order,
                      const double eta_zero,
                      const double eta_one,
                      const double eta_two,
                      const double beta,
                      const double energy,
                      const double charge,
                      const double cut_left,
                      const double cut_right);