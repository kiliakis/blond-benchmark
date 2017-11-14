#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
#include <PAPIProf.h>
#include <omp.h>
#include "fft.h"
#include <iostream>

using namespace std;

template <typename T>
void print_vector(vector<T> &v) {
    for (auto &i : v) cout << i << "\n";
}

template <typename T>
void print_vector(T *v, int n) {
    for (int i = 0; i < n; i++) cout << v[i] << "\n";
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

    fft_convolution(signal.data(), n_signal,
                    kernel.data(), n_kernel,
                    result.data(), n_threads);

    auto papiprof = new PAPIProf();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        papiprof->start_counters("fft_convolution");
        fft_convolution(signal.data(), n_signal,
                        kernel.data(), n_kernel,
                        result.data(), n_threads);
        papiprof->stop_counters();
    }
    papiprof->report_timing();
    destroy_plans();
    // report results
    return 0;
}