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
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
        else if (alpha_order == 2)
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
        else
            for (int i = 0; i < n_macroparticles; i++ )
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]
                                         - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
    }

}