#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#include <string.h>     // memset()
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "histogram.h"


void histogram_v0(const double *__restrict__ input,
                  double *__restrict__ output, const double cut_left,
                  const double cut_right, const int n_slices,
                  const int n_macroparticles)
{

    const double inv_bin_width = n_slices / (cut_right - cut_left);

    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_slices * sizeof(double));

    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        // memset(histo[id], 0., n_slices * sizeof(double));
        for (int i = 0; i < n_slices; ++i) histo[id][i] = 0.0;

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i ++) {
            if ((input[i] < cut_left) || input[i] > cut_right)
                continue;
            int bin = (int) (input[i] - cut_left) * inv_bin_width;
            histo[id][bin] += 1.;
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
    }

    free(histo[0]);
    free(histo);
}

void histogram_v6(const double *__restrict__ input,
                  double *__restrict__ output, const double cut_left,
                  const double cut_right, const int n_slices,
                  const int n_macroparticles)
{

    const int STEP = 8;
    const double inv_bin_width = n_slices / (cut_right - cut_left);

    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_slices * sizeof(double));

    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        // memset(histo[id], 0., n_slices * sizeof(double));
        for (int i = 0; i < n_slices; ++i) histo[id][i] = 0.0;

        unsigned int fbin[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned int) floor((input[i + j] - cut_left) * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices)
                    histo[id][fbin[j]] += 1.;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
    }

    free(histo[0]);
    free(histo);
}

void histogram_v9(const double *__restrict__ input,
                  double *__restrict__ output, const double cut_left,
                  const double cut_right, const int n_slices,
                  const int n_macroparticles)
{

    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);

    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_slices * sizeof(double));

    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_slices * sizeof(double));
        unsigned fbin[STEP];
        // unsigned int bin[STEP];
        float flag[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;
#pragma simd
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) floor((input[i + j] - cut_left)
                                           * inv_bin_width);
            }
// #pragma simd
            for (int j = 0; j < loop_count; j++) {
                flag[j] = fbin[j] < n_slices ? 1.0 : 0.0;
                fbin[j] = flag[j] > 0.0 ? fbin[j] : 0;
            }

// #pragma simd
            for (int j = 0; j < loop_count; j++) {
                // unsigned bin  = (unsigned) floor(fbin[j]);
                // if (fbin[j] < n_slices)
                histo[id][fbin[j]] += flag[j];
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
    }

    free(histo[0]);
    free(histo);
}
