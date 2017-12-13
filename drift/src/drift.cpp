#include <string.h>     // memset()
#include <stdlib.h>
#include <cmath>
#include "drift.h"


void drift_v0(double * __restrict__ beam_dt,
              const double * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const double T0, const double length_ratio,
              const double alpha_order, const double eta_zero,
              const double eta_one, const double eta_two,
              const double beta, const double energy,
              const int n_macroparticles)
{

    const double T = T0 * length_ratio;

    if ( strcmp (solver, "simple") == 0 ) {
        double coeff = T * eta_zero / (beta * beta * energy);
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dt[i] += coeff * beam_dE[i];
    } else {
        const double coeff = 1. / (beta * beta * energy);
        const double eta0 = eta_zero * coeff;
        const double eta1 = eta_one * coeff * coeff;
        const double eta2 = eta_two * coeff * coeff * coeff;

        if ( alpha_order == 1 )
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
        else if (alpha_order == 2)
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
        else
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]
                                         - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
    }

}


void drift_v1(double * __restrict__ beam_dt,
              const double * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const double T0, const double length_ratio,
              const double alpha_order, const double eta_zero,
              const double eta_one, const double eta_two,
              const double beta, const double energy,
              const int n_macroparticles)
{

    const double T = T0 * length_ratio;

    if ( strcmp (solver, "simple") == 0 ) {
        double coeff = T * eta_zero / (beta * beta * energy);
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dt[i] += coeff * beam_dE[i];
    } else {
        const double coeff = 1. / (beta * beta * energy);
        const double eta0 = eta_zero * coeff;
        const double eta1 = eta_one * coeff * coeff;
        const double eta2 = eta_two * coeff * coeff * coeff;
        const double coeff2 = T * eta0;

        if ( alpha_order == 1 )
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += coeff2 * beam_dE[i] / (1.0 - eta0 * beam_dE[i]);
        else if (alpha_order == 2)
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
        else
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]
                                         - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
    }

}


void drift_v2(float * __restrict__ beam_dt,
              const float * __restrict__ beam_dE,
              const char * __restrict__ solver,
              const float T0, const float length_ratio,
              const float alpha_order, const float eta_zero,
              const float eta_one, const float eta_two,
              const float beta, const float energy,
              const int n_macroparticles)
{

    const float T = T0 * length_ratio;

    if ( strcmp (solver, "simple") == 0 ) {
        float coeff = T * eta_zero / (beta * beta * energy);
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dt[i] += coeff * beam_dE[i];
    } else {
        const float coeff = 1.0f / (beta * beta * energy);
        const float eta0 = eta_zero * coeff;
        const float eta1 = eta_one * coeff * coeff;
        const float eta2 = eta_two * coeff * coeff * coeff;

        if ( alpha_order == 1 )
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1.0f / (1.0f - eta0 * beam_dE[i]) - 1.0f);
        else if (alpha_order == 2)
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1.0f / (1.0f - eta0 * beam_dE[i]
                                           - eta1 * beam_dE[i] * beam_dE[i]) - 1.0f);
        else
            #pragma omp parallel for
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1.0f / (1.0f - eta0 * beam_dE[i]
                                           - eta1 * beam_dE[i] * beam_dE[i]
                                           - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.0f);
    }

}