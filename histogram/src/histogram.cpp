#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#include <string.h>     // memset()
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "histogram.h"


// naive implementation
void histogram_v0(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles) {

    int i;
    FTYPE a;
    FTYPE fbin;
    int ffbin;
    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    for (i = 0; i < n_slices; i++) {
        output[i] = 0.0;
    }

    for (i = 0; i < n_macroparticles; i++) {
        a = input[i];
        if ((a < cut_left) || (a > cut_right))
            continue;
        fbin = (a - cut_left) * inv_bin_width;
        ffbin = (int)(fbin);
        output[ffbin] = output[ffbin] + 1;
    }
}


// parallel, global histogram
void histogram_v1(const FTYPE * __restrict__ input,
                  HIST_T * __restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles) {

    int i;
    FTYPE a;
    FTYPE fbin;
    int ffbin;
    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    for (i = 0; i < n_slices; i++) {
        output[i] = 0.0;
    }
    #pragma omp parallel for
    for (i = 0; i < n_macroparticles; i++) {
        a = input[i];
        if ((a < cut_left) || (a > cut_right))
            continue;
        fbin = (a - cut_left) * inv_bin_width;
        ffbin = (int)(fbin);

        #pragma omp atomic
        output[ffbin] = output[ffbin] + 1;
    }
}


// parallel, local histo, serial reduction, different mem allocation
void histogram_v2(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output, const FTYPE cut_left,
                  const FTYPE cut_right, const int n_slices,
                  const int n_macroparticles)
{

    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    HIST_T *histo = NULL;

    #pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int id = omp_get_thread_num();
        const int offset = id * n_slices;
        #pragma omp single
        {
            histo = (HIST_T *) malloc (n_slices * threads * sizeof(HIST_T));
        }
        memset(&histo[offset], 0, n_slices * sizeof(HIST_T));

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i ++) {
            if ((input[i] < cut_left) || input[i] > cut_right)
                continue;
            int bin = (int) (input[i] - cut_left) * inv_bin_width;
            histo[offset + bin] += 1;
        }

        #pragma omp single
        {
            memset(output, 0, n_slices * sizeof(HIST_T));
            for (int t = 0; t < n_slices * threads; t += n_slices)
                for (int i = 0; i < n_slices; i++)
                    output[i] += histo[t + i];
        }
    }
    free(histo);
}



// parallel, local histo, parallel reduction method 1
void histogram_v3(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output, const FTYPE cut_left,
                  const FTYPE cut_right, const int n_slices,
                  const int n_macroparticles)
{

    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    HIST_T *histo = NULL;

    #pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int id = omp_get_thread_num();
        const int offset = id * n_slices;
        #pragma omp single
        {
            histo = (HIST_T *) malloc (n_slices * threads * sizeof(HIST_T));
        }
        memset(&histo[offset], 0, n_slices * sizeof(HIST_T));

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i ++) {
            if ((input[i] < cut_left) || input[i] > cut_right)
                continue;
            int bin = (int) (input[i] - cut_left) * inv_bin_width;
            histo[offset + bin] += 1.;
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t * n_slices + i];
        }
    }
    free(histo);
}

/*
// parallel, local histo, parallel reduction method 2
void histogram_v4(const FTYPE *__restrict__ input,
                      HIST_T *__restrict__ output, const FTYPE cut_left,
                      const FTYPE cut_right, const int n_slices,
                      const int n_macroparticles)
{

    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    // memset(output, 0, n_slices * sizeof(FTYPE));

    FTYPE *histo = NULL;

    #pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int id = omp_get_thread_num();
        const int offset = id * n_slices;
        #pragma omp single
        {
            histo = (FTYPE *) malloc (n_slices * threads * sizeof(FTYPE));
        }
        memset(&histo[offset], 0, n_slices * sizeof(FTYPE));

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i ++) {
            if ((input[i] < cut_left) || input[i] > cut_right)
                continue;
            int bin = (int) (input[i] - cut_left) * inv_bin_width;
            histo[offset + bin] += 1.;
        }

        for (int s = 1; s < threads; s *= 2) {
            if (id % (2 * s) == 0) {
                for (int i = 0; i < n_slices; i++)
                    histo[offset + i] += histo[offset + s * n_slices + i];
            }
        }
        if (id == 0) {
            for (int i = 0; i < n_slices; i++)
                output[i] = histo[offset + i];
        }
    }

    free(histo);
}
*/


// parallel, local histo, parallel reduce, loop tiling
void histogram_v4(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles)
{

    const int STEP = 8;
    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    HIST_T *histo = NULL;

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        #pragma omp single
        {
            histo = (HIST_T *) malloc (n_slices * threads * sizeof(HIST_T));
        }
        HIST_T *histo_loc = &histo[id * n_slices];
        memset(histo_loc, 0, n_slices * sizeof(HIST_T));

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
                    histo_loc[fbin[j]] += 1.;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t * n_slices + i];
        }
    }
    free(histo);
}


// parallel, local histo, parallel reduce, loop tiling, parallel allocation
void histogram_v5(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles)
{

    const int STEP = 32;
    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    HIST_T **histo = (HIST_T **) malloc (omp_get_max_threads() * sizeof(HIST_T *));

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        histo[id] = (HIST_T *) malloc (n_slices * sizeof(HIST_T));

        memset(histo[id], 0, n_slices * sizeof(HIST_T));

        unsigned int fbin[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            int loop_count = STEP;
            if (loop_count > n_macroparticles - i)
                loop_count = n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned int) floor((input[i + j] - cut_left) * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices)
                    histo[id][fbin[j]] += 1;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
        free(histo[id]);
    }

    free(histo);
}


// parallel, local histo, parallel reduce, loop tiling, parallel allocation
// heavier optimizations
void histogram_v6(const FTYPE *__restrict__ input,
                  HIST_T *__restrict__ output,
                  const FTYPE cut_left,
                  const FTYPE cut_right,
                  const int n_slices,
                  const int n_macroparticles)
{

    const int STEP = 32;
    const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

    HIST_T **histo = (HIST_T **) malloc (omp_get_max_threads() * sizeof(HIST_T *));

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        histo[id] = (HIST_T *) malloc (n_slices * sizeof(HIST_T));

        memset(histo[id], 0, n_slices * sizeof(HIST_T));

        FTYPE fbin[STEP];
        unsigned int ubin[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            int loop_count = STEP;
            if (loop_count > n_macroparticles - i)
                loop_count = n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = ((input[i + j] - cut_left) * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                ubin[j] = floor(fbin[j]);
            }

            for (int j = 0; j < loop_count; j++) {
                if (ubin[j] < n_slices)
                    histo[id][ubin[j]] += 1;
            }
        }

        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
        free(histo[id]);
    }

    free(histo);
}

// void histogram_v9(const FTYPE *__restrict__ input,
//                   HIST_T *__restrict__ output, const FTYPE cut_left,
//                   const FTYPE cut_right, const int n_slices,
//                   const int n_macroparticles)
// {

//     const int STEP = 16;
//     const FTYPE inv_bin_width = n_slices / (cut_right - cut_left);

//     HIST_T **histo = (HIST_T **) malloc(omp_get_max_threads() * sizeof(HIST_T *));
//     histo[0] = (HIST_T *) malloc (omp_get_max_threads() * n_slices * sizeof(HIST_T));

//     for (int i = 0; i < omp_get_max_threads(); i++)
//         histo[i] = (*histo + n_slices * i);

//     #pragma omp parallel
//     {
//         const int id = omp_get_thread_num();
//         const int threads = omp_get_num_threads();
//         memset(histo[id], 0., n_slices * sizeof(HIST_T));
//         unsigned fbin[STEP];
//         // unsigned int bin[STEP];
//         float flag[STEP];
//         #pragma omp for
//         for (int i = 0; i < n_macroparticles; i += STEP) {

//             const int loop_count = n_macroparticles - i > STEP ?
//                                    STEP : n_macroparticles - i;
// #pragma simd
//             for (int j = 0; j < loop_count; j++) {
//                 fbin[j] = (unsigned) floor((input[i + j] - cut_left)
//                                            * inv_bin_width);
//             }
// // #pragma simd
//             for (int j = 0; j < loop_count; j++) {
//                 flag[j] = fbin[j] < n_slices ? 1.0 : 0.0;
//                 fbin[j] = flag[j] > 0.0 ? fbin[j] : 0;
//             }

// // #pragma simd
//             for (int j = 0; j < loop_count; j++) {
//                 // unsigned bin  = (unsigned) floor(fbin[j]);
//                 // if (fbin[j] < n_slices)
//                 histo[id][fbin[j]] += flag[j];
//             }
//         }

//         #pragma omp for
//         for (int i = 0; i < n_slices; i++) {
//             output[i] = 0.;
//             for (int t = 0; t < threads; t++)
//                 output[i] += histo[t][i];
//         }
//     }

//     free(histo[0]);
//     free(histo);
// }
