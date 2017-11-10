#include "convolution.h"
#include <algorithm>    // std::min

void convolution_v0(const double *__restrict__ signal,
                    const int SignalLen,
                    const double *__restrict__ kernel,
                    const int KernelLen,
                    double *__restrict__ res)
{
    const int size = KernelLen + SignalLen - 1;

    #pragma omp parallel for
    for (int n = 0; n < size; ++n) {
        res[n] = 0;
        const int kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
        const int kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
        for (int k = kmin; k <= kmax; k++) {
            res[n] += signal[k] * kernel[n - k];
        }
    }
}

void convolution_v1(const double *__restrict__ signal,
                    const int SignalLen,
                    const double *__restrict__ kernel,
                    const int KernelLen,
                    double *__restrict__ res,
                    const int resLen)
{
    const int size = std::min(resLen, KernelLen + SignalLen - 1);

    #pragma omp parallel for
    for (int n = 0; n < size; ++n) {
        res[n] = 0;
        const int kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
        const int kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
        for (int k = kmin; k <= kmax; k++) {
            res[n] += signal[k] * kernel[n - k];
        }
    }
}


void convolution_v2(const double *__restrict__ signal,
                    const int SignalLen,
                    double *__restrict__ kernel,
                    const int KernelLen,
                    double *__restrict__ res)
{
    const int size = KernelLen + SignalLen - 1;
    #pragma omp parallel for
    for (int i = 0; i < KernelLen / 2; ++i) {
        std::swap(kernel[i], kernel[KernelLen - i]);
    }
    const int STEP = 16;
    #pragma omp parallel for
    for (int n = 0; n < size; ++n) {
        res[n] = 0;
        const int kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
        const int kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
        double products[STEP];
        for (int k = kmin; k <= kmax; k += STEP) {
            const int loop_count = kmax - k > STEP ? STEP : kmax - k;
            for (int j = 0; j < loop_count; j++) {
                products[j] += signal[k + j] * kernel[k + j];
            }
            for (int j = 0; j < loop_count; j++)
                res[n] += products[j];
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < KernelLen / 2; ++i) {
        std::swap(kernel[i], kernel[KernelLen - i]);
    }

}
