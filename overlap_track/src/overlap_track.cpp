#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#include <string.h>     // memset()
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "overlap_track.h"
#include <utility>

using namespace std;

inline unsigned myfloor(const double &x) {
    if (x < 0.0) return static_cast<unsigned>(-1);
    else return static_cast<unsigned>(x);
}


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
                      const double cut_right)
{


    // interp kick
    double inv_bin_width = (n_slices - 1) / (bin_centers[n_slices - 1] - bin_centers[0]);

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        double voltageKick;
        const double a = dt[i];
        const int ffbin = (int) std::floor((a - bin_centers[0]) * inv_bin_width);
        if ((a < bin_centers[0]) || (a > bin_centers[n_slices - 1]))
            voltageKick = 0.;
        else
            voltageKick = voltage_array[ffbin]
                          + (a - bin_centers[ffbin])
                          * (voltage_array[ffbin + 1] - voltage_array[ffbin])
                          * inv_bin_width;
        dE[i] = dE[i] + voltageKick + acc_kick;
    }



    // drift
    const double T = T0 * length_ratio;

    if ( strcmp (solver, "simple") == 0 ) {
        double coeff = T * eta_zero / (beta * beta * energy);
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++)
            dt[i] += coeff * dE[i];
    } else {
        const double coeff = 1. / (beta * beta * energy);
        const double eta0 = eta_zero * coeff;
        const double eta1 = eta_one * coeff * coeff;
        const double eta2 = eta_two * coeff * coeff * coeff;

        if ( alpha_order == 1 )
            #pragma omp parallel for
            for (int i = 0; i < n_particles; i++ )
                dt[i] += T * (1. / (1. - eta0 * dE[i]) - 1.);
        else if (alpha_order == 2)
            #pragma omp parallel for
            for (int i = 0; i < n_particles; i++ )
                dt[i] += T * (1. / (1. - eta0 * dE[i]
                                    - eta1 * dE[i] * dE[i]) - 1.);
        else
            #pragma omp parallel for
            for (int i = 0; i < n_particles; i++ )
                dt[i] += T * (1. / (1. - eta0 * dE[i]
                                    - eta1 * dE[i] * dE[i]
                                    - eta2 * dE[i] * dE[i] * dE[i]) - 1.);
    }


    // histogram
    inv_bin_width = n_slices / (cut_right - cut_left);

    for (int i = 0; i < n_slices; i++) {
        profile[i] = 0.0;
    }

    for (int i = 0; i < n_particles; i++) {
        double a = dt[i];
        if ((a < cut_left) || (a > cut_right))
            continue;
        double fbin = (a - cut_left) * inv_bin_width;
        int ffbin = (int)(fbin);
        profile[ffbin] = profile[ffbin] + 1.0;
    }

}





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
                      const double cut_right)
{


    // interp kick and drift
    int STEP = 64;
    double inv_bin_width = (n_slices - 1)
                           / (bin_centers[n_slices - 1]
                              - bin_centers[0]);
    const double coeff = T0 * length_ratio * eta_zero / (beta * beta * energy);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));
    double *factor = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = charge * (voltage_array[i] - bin_centers[i] * voltageKick[i]) + acc_kick;
        }

        #pragma omp for
        for (int i = 0; i < n_particles; i += STEP) {

            const int loop_count = n_particles - i > STEP ?
                                   STEP : n_particles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((dt[i + j] - bin_centers[0])
                                                * inv_bin_width);
                // dE[i + j] += acc_kick;
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    dE[i + j] += dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }
            }

            for (int j = 0; j < loop_count; j++) {
                dt[i + j] += coeff * dE[i + j];
            }
        }
    }
    free(voltageKick);
    free(factor);


    // histogram
    STEP = 32;
    inv_bin_width = n_slices / (cut_right - cut_left);

    double **histo = (double **) malloc (omp_get_max_threads() * sizeof(double *));

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        histo[id] = (double *) malloc (n_slices * sizeof(double));

        memset(histo[id], 0, n_slices * sizeof(double));

        unsigned int fbin[STEP];
        #pragma omp for
        for (int i = 0; i < n_particles; i += STEP) {

            int loop_count = STEP;
            if (loop_count > n_particles - i)
                loop_count = n_particles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned int) floor((dt[i + j] - cut_left) * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices)
                    histo[id][fbin[j]] += 1.0;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            profile[i] = 0.;
            for (int t = 0; t < threads; t++)
                profile[i] += histo[t][i];
        }
        free(histo[id]);
    }

    free(histo);

}



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
                      const double cut_right)
{


    // interp kick and drift
    int STEP = 512;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    const double inv_bin_width2 = n_slices / (cut_right - cut_left);

    const double coeff = T0 * length_ratio * eta_zero / (beta * beta * energy);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));
    double *factor = (double *) malloc ((n_slices - 1) * sizeof(double));
    double **histo = (double **) malloc (omp_get_max_threads() * sizeof(double *));



    #pragma omp parallel
    {
        unsigned int fbin[STEP];

        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        histo[id] = (double *) malloc (n_slices * sizeof(double));
        memset(histo[id], 0, n_slices * sizeof(double));

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = charge * (voltage_array[i] - bin_centers[i] * voltageKick[i]) + acc_kick;
        }

        #pragma omp for
        for (int i = 0; i < n_particles; i += STEP) {

            const int loop_count = n_particles - i > STEP ?
                                   STEP : n_particles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((dt[i + j] - bin_centers[0])
                                                * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    dE[i + j] += dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }
            }

            for (int j = 0; j < loop_count; j++) {
                dt[i + j] += coeff * dE[i + j];
            }

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((dt[i + j] - cut_left)
                                                * inv_bin_width2);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices)
                    histo[id][fbin[j]] += 1.0;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            profile[i] = 0.;
            for (int t = 0; t < threads; t++)
                profile[i] += histo[t][i];
        }
        free(histo[id]);
    }
    free(voltageKick);
    free(factor);
    free(histo);

}



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
                      const double cut_right)
{


    // interp kick and drift
    int STEP = 512;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    const double inv_bin_width2 = n_slices / (cut_right - cut_left);

    const double coeff = T0 * length_ratio * eta_zero / (beta * beta * energy);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));
    double *factor = (double *) malloc ((n_slices - 1) * sizeof(double));
    double **histo = (double **) malloc (omp_get_max_threads() * sizeof(double *));



    #pragma omp parallel
    {
        unsigned int fbin[STEP];

        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        histo[id] = (double *) malloc (n_slices * sizeof(double));
        memset(histo[id], 0, n_slices * sizeof(double));

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = charge * (voltage_array[i] - bin_centers[i] * voltageKick[i]) + acc_kick;
        }

        #pragma omp for
        for (int i = 0; i < n_particles; i += STEP) {

            const int loop_count = n_particles - i > STEP ?
                                   STEP : n_particles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((dt[i + j] - bin_centers[0])
                                                * inv_bin_width);


                if (fbin[j] < n_slices - 1) {
                    dE[i + j] += dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }

                dt[i + j] += coeff * dE[i + j];

                fbin[j] = (unsigned) std::floor((dt[i + j] - cut_left)
                                                * inv_bin_width2);

                if (fbin[j] < n_slices)
                    histo[id][fbin[j]] += 1.0;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            profile[i] = 0.;
            for (int t = 0; t < threads; t++)
                profile[i] += histo[t][i];
        }
        free(histo[id]);
    }
    free(voltageKick);
    free(factor);
    free(histo);

}