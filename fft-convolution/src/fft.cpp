/*
 * fft.cpp
 *
 *  Created on: Mar 21, 2016
 *      Author: kiliakis
 */

#include "fft.h"
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <functional>
#include <iostream>
#include <omp.h>

// FFTW_PATIENT: run a lot of ffts to discover the best plan.
// Will not use the multithreaded version unless the fft size
// is reasonable enough

// FFTW_MEASURE : run some ffts to find the best plan.
// (first run should take some more seconds)

// FFTW_ESTIMATE : dont run any ffts, just make an estimation.
// Usually leads to suboptimal solutions

// FFTW_DESTROY_INPUT : use the original input to store arbitaty data.
// May yield better performance but the input is not usable any more.
// Can be combined with all the above

static std::vector<fft_plan_t> planV;
static bool hasBeenInit = false;
// const uint FFTW_FLAGS = FFTW_MEASURE | FFTW_DESTROY_INPUT;
// const uint FFTW_FLAGS = FFTW_PATIENT;
const uint FFTW_FLAGS = FFTW_ESTIMATE;
// const uint FFTW_FLAGS = FFTW_MEASURE;

using namespace std;
// Parameters are like python's numpy.fft.rfft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the transformed array
extern "C" {

    fftw_plan init_fft(const int n,  complex_t *in, complex_t *out,
                       const int sign = FFTW_FORWARD,
                       const unsigned flag = FFTW_ESTIMATE,
                       const int threads = 1)
    {
        if (threads > 1) {
            if (!hasBeenInit){
                if(fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
        // cout << "Threads: " << threads << "\n";

        fftw_complex *a = reinterpret_cast<fftw_complex *>(in);
        fftw_complex *b = reinterpret_cast<fftw_complex *>(out);
        return fftw_plan_dft_1d(n, a, b, sign, flag);
    }

    fftw_plan init_rfft(const int n, double *in, complex_t *out,
                        const unsigned flag = FFTW_ESTIMATE,
                        const int threads = 1)

    {
        if (threads > 1) {
            if (!hasBeenInit){
                if(fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
        // cout << "Threads: " << threads << "\n";
        fftw_complex *b = reinterpret_cast<fftw_complex *>(out);
        return fftw_plan_dft_r2c_1d(n, in, b, flag);
    }

    fftw_plan init_irfft(const int n, complex_t *in, double *out,
                         const unsigned flag = FFTW_ESTIMATE,
                         const int threads = 1)
    {
        if (threads > 1) {
            if (!hasBeenInit){
                if(fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
        // cout << "Threads: " << threads << "\n";
        
        fftw_complex *b = reinterpret_cast<fftw_complex *>(in);
        return fftw_plan_dft_c2r_1d(n, b, out, flag);
    }

    void run_fft(const fftw_plan &p) { fftw_execute(p);}

    void destroy_fft(fftw_plan &p) { fftw_destroy_plan(p); }




    fft_plan_t find_plan(int fftSize, int inSize, fft_type_t type, int threads,
                         vector<fft_plan_t> &v)
    {
        // const uint flag = FFTW_FLAGS;
        auto it =
        find_if(v.begin(), v.end(), [inSize, fftSize, type](const fft_plan_t &s) {
            return ((s.inSize == inSize) && (s.fftSize == fftSize) && (s.type == type));
        });

        if (it == v.end()) {
            fft_plan_t plan;
            plan.inSize = inSize;
            plan.fftSize = fftSize;
            plan.type = type;

            if (type == FFT) {
                fftw_complex *in =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);
                fftw_complex *out =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);

                auto p = init_fft(fftSize, reinterpret_cast<complex_t *>(in),
                                  reinterpret_cast<complex_t *>(out),
                                  FFTW_FORWARD, FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;
            } else if (type == IFFT) {

                fftw_complex *in =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);
                fftw_complex *out =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);
                auto p = init_fft(fftSize, reinterpret_cast<complex_t *>(in),
                                  reinterpret_cast<complex_t *>(out),
                                  FFTW_BACKWARD, FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else if (type == RFFT) {
                double *in = (double *)fftw_malloc(sizeof(double) * fftSize);
                fftw_complex *out = (fftw_complex *)fftw_malloc(
                                        sizeof(fftw_complex) * (fftSize / 2 + 1));
                auto p = init_rfft(fftSize, in, reinterpret_cast<complex_t *>(out),
                                   FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else if (type == IRFFT) {
                fftw_complex *in = (fftw_complex *)fftw_malloc(
                                       sizeof(fftw_complex) * (fftSize / 2 + 1));
                double *out = (double *)fftw_malloc(sizeof(double) * fftSize);
                auto p = init_irfft(fftSize, reinterpret_cast<complex_t *>(in), out,
                                    FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else {
                printf("[fft::find_plan]: ERROR Wrong fft type!\n");
                exit(-1);
            }

            v.push_back(plan);
            return plan;
        } else {
            return *it;
        }
    }



    void destroy_plans()
    {
        for (auto &i : planV) {
            fftw_destroy_plan(i.p);
            fftw_free(i.in);
            fftw_free(i.out);
        }
        planV.clear();
    }

    void rfft(double *in, const int inSize,
              complex_t *out, int fftSize,
              const int threads)
    {
        if (fftSize == 0) fftSize = inSize / 2 + 1;
        // cout << "Num of threads: " << omp_get_max_threads() << "\n";
        auto plan = find_plan(fftSize, inSize, RFFT, omp_get_max_threads(), planV);
        auto from = (double *)plan.in;
        auto to = (complex_t *)plan.out;

        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }
        run_fft(plan.p);

        copy(to, to + (fftSize / 2 + 1), out);
    }

    // Inverse of rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // Missing n: size of output
    void irfft(complex_t *in, const int inSize,
               double *out, int fftSize,
               const int threads)
    {
        if (fftSize == 0) fftSize = 2 * (inSize - 1);

        auto plan = find_plan(fftSize, inSize, IRFFT, omp_get_max_threads(), planV);
        auto from = (complex_t *)plan.in;
        auto to = (double *)plan.out;

        copy(in, in + inSize, from);

        run_fft(plan.p);

        transform(&to[0], &to[fftSize], out,
                  bind2nd(divides<double>(), fftSize));
    }


    // Parameters are like python's numpy.fft.ifft
    // @in:  input data
    // @n:   number of points to use. If n < in.size() then the input is cropped
    //       if n > in.size() then input is padded with zeros
    // @out: the inverse Fourier transform of input data
    void ifft(complex_t *in, const int inSize,
              complex_t *out, int fftSize,
              const int threads)
    {
        if (fftSize == 0) fftSize = inSize;

        auto plan = find_plan(fftSize, inSize, IFFT, omp_get_max_threads(), planV);
        auto from = (complex_t *)plan.in;
        auto to = (complex_t *)plan.out;
        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }

        run_fft(plan.p);

        transform(&to[0], &to[fftSize], out,
                  bind2nd(divides<complex_t>(), fftSize));
    }


// Parameters are like python's numpy.fft.fft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the transformed array
    void fft(complex_t *in, const int inSize,
             complex_t *out, int fftSize,
             const int threads)
    {
        if (fftSize == 0) fftSize = inSize;

        auto plan = find_plan(fftSize, inSize, FFT, omp_get_max_threads(), planV);
        auto from = (complex_t *)plan.in;
        auto to = (complex_t *)plan.out;

        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }
        run_fft(plan.p);

        copy(&to[0], &to[fftSize], out);
    }



// Same as python's numpy.fft.rfftfreq
// @ n: window length
// @ d (optional) : Sample spacing
// @return: A vector of length (n div 2) + 1 of the sample frequencies
    void rfftfreq(const int n, double *out, const double d)
    {
        const double factor = 1.0 / (d * n);
        #pragma omp parallel for
        for (int i = 0; i < n / 2 + 1; ++i) {
            out[i] = i * factor;
        }
    }


    void fft_convolution(double * signal, const int signalLen,
                         double * kernel, const int kernelLen,
                         double * res, const int threads)
    {
        const size_t realSize = signalLen + kernelLen - 1;
        const size_t complexSize = realSize / 2 + 1;
        complex_t *z1 = (complex_t *) fftw_alloc_complex (2 * complexSize);
        complex_t *z2 = z1 + complexSize;

        rfft(signal, signalLen, z1, realSize, threads);
        rfft(kernel, kernelLen, z2, realSize, threads);

        // transform(z1, z1 + complexSize, z2, z1,
        //           multiplies<complex_t>());

        // irfft(z1, complexSize, res, realSize, threads);

        fftw_free(z1);
    }
}