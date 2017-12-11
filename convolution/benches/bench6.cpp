#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
// #include <PAPIProf.h>
#include <omp.h>
#include <algorithm>
#include <string.h>
#include <mkl_vsl.h>

using namespace std;


void convolution_mkl(const double * __restrict__ signal,
                     const int signalLen,
                     const double * __restrict__ kernel,
                     const int kernelLen,
                     double * __restrict__ result,
                     const int threads)
{

    const int STEP = 64;
    memset(result, 0.0, (signalLen + kernelLen - 1) * sizeof(double));

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        // static thread_local VSLConvTaskPtr task = nullptr;
        VSLConvTaskPtr task = nullptr;
        const int kernelLenPT = (kernelLen + threads - 1) / threads;
        // const int computeLen = signalLen + min(kernelLenPT, kernelLen - tid * kernelLenPT) - 1;
        double resultPT[STEP] = {0.0};

        for (int i = 0; i < kernelLenPT; i += STEP) {
            int loop_count = STEP;
            if (kernelLenPT - i < STEP) loop_count = kernelLenPT - i;

            vsldConvNewTask1D(&task, VSL_CONV_MODE_DIRECT, signalLen,
                              loop_count, singalLen + loop_count - 1);
            vsldConvExec1D(task, signal, 1, &kernel[tid * kernelLenPT + i],
                           1, resultPT, 1);
            for (int j = 0; j < loop_count; ++j) {
                #pragma omp atomic
                result[i + j + tid * kernelLenPT] += resultPT[j];
            }

        }
    }
}
// WARNING: This implementatio is not correct, the result has to be at least 
//  signalLen size so tiling to save memory is not worth it


int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_signal = 1000;
    int n_kernel = 1000;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_signal = atoi(argv[2]);
    if (argc > 3) n_kernel = atoi(argv[3]);
    if (argc > 4) n_threads = atoi(argv[4]);
    omp_set_num_threads(n_threads);
    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> signal, kernel;
    vector<double> result;
    signal.resize(n_signal);
    kernel.resize(n_kernel);
    result.resize(n_signal + n_kernel - 1);

    for (int i = 0; i < n_signal; ++i) {
        signal[i] = d(gen);
    }

    for (int i = 0; i < n_kernel; ++i) {
        kernel[i] = d(gen);
    }


    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("convolution");
    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        convolution_mkl(signal.data(), n_signal,
                        kernel.data(), n_kernel,
                        result.data(), n_threads);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("convolution_v6\ttime(ms)\t%d\t0\t1\n", duration);
    printf("result: %lf\n", accumulate(result.begin(), result.end(), 0.0) / (n_signal + n_kernel - 1));
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}