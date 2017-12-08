#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#include <string.h>     // memset()
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "interp_kick.h"
#include <utility>

using namespace std;

// original basic implementation
void linear_interp_kick_v0(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const double inv_bin_width = (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        double voltageKick;
        const double a = beam_dt[i];
        const int ffbin = (int) std::floor((a - bin_centers[0]) * inv_bin_width);
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

// precalculate the voltages
void linear_interp_kick_v1(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < n_slices - 1; i++) {
        voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
    }

    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {

        unsigned fbin = (unsigned) std::floor((beam_dt[i] - bin_centers[0]) * inv_bin_width);
        beam_dE[i] += acc_kick;
        if (fbin < n_slices - 1) {
            beam_dE[i] += voltage_array[fbin] + (beam_dt[i] - bin_centers[fbin])
                          * voltageKick[fbin];
        }
    }

    free(voltageKick);

}


// precalculate voltages, loop-tiling
void linear_interp_kick_v2(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 64;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                           * inv_bin_width);
                beam_dE[i + j] += acc_kick;
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    beam_dE[i + j] += voltage_array[fbin[j]]
                                      + (beam_dt[i + j] - bin_centers[fbin[j]])
                                      * voltageKick[fbin[j]];
                }
            }
        }
    }
    free(voltageKick);

}


// precalculate voltages, loop-tiling, floats
void linear_interp_kick_v3(
    const float * __restrict__ beam_dt,
    float * __restrict__ beam_dE,
    const float * __restrict__ voltage_array,
    const float * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick)
{


    const int STEP = 64;
    const float inv_bin_width = (n_slices - 1)
                                / (bin_centers[n_slices - 1]
                                   - bin_centers[0]);

    float *voltageKick = (float *) malloc ((n_slices - 1) * sizeof(float));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                           * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    beam_dE[i + j] += voltage_array[fbin[j]]
                                      + (beam_dt[i + j] - bin_centers[fbin[j]])
                                      * voltageKick[fbin[j]];
                }
                beam_dE[i + j] += acc_kick;
            }
        }
    }
    free(voltageKick);

}


// precalculate voltages, loop-tiling, precalculate functor
void linear_interp_kick_v4(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 64;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));
    double *factor = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = voltage_array[i] - bin_centers[i] * voltageKick[i];
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                           * inv_bin_width);
                beam_dE[i + j] += acc_kick;
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    beam_dE[i + j] += beam_dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }
            }
        }
    }
    free(voltageKick);
    free(factor);

}


// precalculate voltages, loop-tiling, precalculate functor, optimized AoS
void linear_interp_kick_v5(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 64;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    auto factors = (pair<double, double> *) malloc ((n_slices - 1) * sizeof(pair<double, double>));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            factors[i].first =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factors[i].second = voltage_array[i] - bin_centers[i] * factors[i].first;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                           * inv_bin_width);
                beam_dE[i + j] += acc_kick;
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    beam_dE[i + j] += beam_dt[i + j] * factors[fbin[j]].first
                                      + factors[fbin[j]].second;
                }
            }
        }
    }
    free(factors);

}


// precalculate voltages, loop-tiling, precalculate functor, optimized AoS
void linear_interp_kick_v6(
    const float * __restrict__ beam_dt,
    float * __restrict__ beam_dE,
    const float * __restrict__ voltage_array,
    const float * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick)
{


    const int STEP = 64;
    const float inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    auto factors = (pair<float, float> *) malloc ((n_slices - 1) * sizeof(pair<float, float>));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            factors[i].first =  (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factors[i].second = voltage_array[i] - bin_centers[i] * factors[i].first;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                           * inv_bin_width);
                beam_dE[i + j] += acc_kick;
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    beam_dE[i + j] += beam_dt[i + j] * factors[fbin[j]].first
                                      + factors[fbin[j]].second;
                }
            }
        }
    }
    free(factors);

}


/*
void linear_interp_kick_v5(
    const double * __restrict__ beam_dt,
    double * __restrict__ beam_dE,
    const double * __restrict__ voltage_array,
    const double * __restrict__ bin_centers,
    const int n_slices,
    const int n_macroparticles,
    const double acc_kick)
{


    const int STEP = 128;
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
*/