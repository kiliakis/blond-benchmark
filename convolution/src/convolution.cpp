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