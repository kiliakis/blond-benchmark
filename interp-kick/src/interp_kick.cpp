#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#include <string.h>     // memset()
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include "interp_kick.h"


void linear_interp_kick_v0(
    double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    double * __restrict__ voltage_array,
    double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const double inv_bin_width = (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        double voltageKick;
        const double a = beam_dt[i];
        const int ffbin = (int) floor((a - bin_centers[0]) * inv_bin_width);
        if ((a < bin_centers[0]) || (a > bin_centers[n_slices - 1]))
            voltageKick = 0.;
        else
            voltageKick = voltage_array[ffbin]
                          + (a - bin_centers[ffbin])
                          * (voltage_array[ffbin + 1] - voltage_array[ffbin])
                          * inv_bin_width;
        beam_dE[i] = beam_dE[i] + voltageKick + acc_kick;
    }

}

void linear_interp_kick_v4(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 32;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        int fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // directive recognized only by icc
#pragma simd
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (int) floor((beam_dt[i + j] - bin_centers[0])
                                      * inv_bin_width);
                beam_dE[i + j] += acc_kick;
            }

            for (int j = 0; j < loop_count; j++) {
                int bin = fbin[j];
                if (bin >= 0 && bin < n_slices - 1) {
                    beam_dE[i + j] += voltage_array[bin]
                                      + (beam_dt[i + j] - bin_centers[bin])
                                      * voltageKick[bin];
                }
            }
        }
    }
    free(voltageKick);

}

void linear_interp_kick_v5(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 32;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        unsigned int fbin[STEP];
        int flag[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // directive recognized only by icc
#pragma simd
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned int) floor((beam_dt[i + j] - bin_centers[0])
                                               * inv_bin_width);
                beam_dE[i + j] += acc_kick;
                flag[j] = fbin[j] < (n_slices - 1);
                fbin[j] = flag[j] ? fbin[j] : 0;
            }

            for (int j = 0; j < loop_count; j++) {
                beam_dE[i + j] += flag[j] * (voltage_array[fbin[j]]
                                             + (beam_dt[i + j] - bin_centers[fbin[j]])
                                             * voltageKick[fbin[j]]);
            }
        }
    }
    free(voltageKick);

}



void linear_interp_kick_v6(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 32;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        float fbin[STEP];
        unsigned int bin[STEP];
        unsigned int flag[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // directive recognized only by icc
#pragma simd
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (beam_dt[i + j] - bin_centers[0]) * inv_bin_width;
                beam_dE[i + j] += acc_kick;
            }

#pragma simd
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = floor(fbin[j]);
                bin[j] = (unsigned int) fbin[j];
                flag[j] = bin[j] < (n_slices - 1);
                bin[j] = flag[j] ? bin[j] : 0;
            }


            for (int j = 0; j < loop_count; j++) {
                beam_dE[i + j] += flag[j] * (voltage_array[bin[j]]
                                             + (beam_dt[i + j] - bin_centers[bin[j]])
                                             * voltageKick[bin[j]]);
            }
        }
    }
    free(voltageKick);

}