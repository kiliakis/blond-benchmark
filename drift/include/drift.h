#pragma once

void drift_v0(double * __restrict__ beam_dt,
              const double * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const double T0, const double length_ratio,
              const double alpha_order, const double eta_zero,
              const double eta_one, const double eta_two,
              const double beta, const double energy,
              const int n_macroparticles);

void drift_v1(double * __restrict__ beam_dt,
              const double * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const double T0, const double length_ratio,
              const double alpha_order, const double eta_zero,
              const double eta_one, const double eta_two,
              const double beta, const double energy,
              const int n_macroparticles);


void drift_v2(float * __restrict__ beam_dt,
              const float * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const float T0, const float length_ratio,
              const float alpha_order, const float eta_zero,
              const float eta_one, const float eta_two,
              const float beta, const float energy,
              const int n_macroparticles);