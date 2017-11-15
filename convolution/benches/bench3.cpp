#include <stdlib.h>
#include <stdio.h>
#include "convolution.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
#include <PAPIProf.h>
#include <omp.h>
#include <mkl_vsl.h>
using namespace std;

void convolution_mkl(const double * __restrict__ signal,
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

    auto papiprof = new PAPIProf();
    papiprof->start_counters("convolution_mkl");
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        convolution_mkl(signal.data(), n_signal,
                        kernel.data(), n_kernel,
                        result.data());
    }
    papiprof->stop_counters();
    papiprof->report_timing();
    // report results

    return 0;
}