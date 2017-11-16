#include "convolution.h"
#include <string.h>
#include <omp.h>
#include <mkl_vsl.h>
#include <algorithm>    // std::min

using namespace std;

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


void convolution_mkl_v0(const double * __restrict__ signal,
                        const int signalLen,
                        const double * __restrict__ kernel,
                        const int kernelLen,
                        double * __restrict__ result)
{
    static VSLConvTaskPtr task = nullptr;
    int status;
    // VSL_CONV_MODE_DIRECT compute convolution directly
    // VSL_CONV_MODE_FFT use FFT
    // VSL_CONV_MODE_AUTO FFT or directly
    if (!task) {
        vsldConvNewTask1D(&task, VSL_CONV_MODE_DIRECT,
                          signalLen, kernelLen,
                          signalLen + kernelLen - 1);
    }
    status = vsldConvExec1D(task, signal, 1, kernel, 1, result, 1);
    // vslConvDeleteTask(&task);
}

void convolution_mkl_v1(const double * __restrict__ signal,
                        const int signalLen,
                        const double * __restrict__ kernel,
                        const int kernelLen,
                        double * __restrict__ result,
                        const int threads)
{

    memset(result, 0.0, (signalLen + kernelLen - 1) * sizeof(double));
    double **resultPT = (double **) malloc(threads * sizeof(double *));

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        static thread_local VSLConvTaskPtr task = nullptr;
        const int kernelLenPT = (kernelLen + threads - 1) / threads;
        const int resultLen = signalLen + kernelLenPT - 1;
        const int computeLen = signalLen + min(kernelLenPT, kernelLen - tid * kernelLenPT) - 1;
        int status;
        #pragma omp single
        {
            resultPT[0] = (double *) malloc (threads * resultLen * sizeof(double));
        }

        resultPT[tid] = (*resultPT + resultLen * tid);
        memset(resultPT[tid], 0.0, computeLen * sizeof(double));

        if (!task) {
            status = vsldConvNewTask1D(&task, VSL_CONV_MODE_DIRECT, signalLen,
                                       kernelLenPT, computeLen);
            if (status != VSL_STATUS_OK) {
                printf("[%d] Error in %s\n", tid, "vsldConvNewTask1D");
                exit(-1);
            }
        }

        status = vsldConvExec1D(task, signal, 1, &kernel[tid * kernelLenPT], 1,
                                resultPT[tid], 1);
        if (status != VSL_STATUS_OK) {
            printf("[%d] Error in %s\n", tid, "vsldConvExec1D");
            exit(-1);
        }

        for (int i = 0; i < computeLen; ++i) {
            #pragma omp atomic
            result[i + tid * kernelLenPT] += resultPT[tid][i];
        }
    }
    free(resultPT[0]);
    free(resultPT);
}
