#include <stdlib.h>
#include <stdio.h>
#include "fast_resonator.h"
#include "utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <string>
// #include <PAPIProf.h>
#include <omp.h>
#include <algorithm>

// #include <ittnotify.h>

using namespace std;

int main(int argc, char const *argv[])
{

    int n_turns = 50000;
    int n_points = 1000000;
    int n_res = 1000;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_points = atoi(argv[2]);
    if (argc > 3) n_res = atoi(argv[3]);
    if (argc > 4) n_threads = atoi(argv[4]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> impReal, impImag;
    vector<double> freq, R_S, Q, freq_R;

    impReal.resize(n_points, 0);
    impImag.resize(n_points, 0);
    freq.resize(n_points);
    R_S.resize(n_res);
    Q.resize(n_res);
    freq_R.resize(n_res);

    for (int i = 0; i < n_points; ++i) {
        freq[i] = d(gen); 
    }

    for (int i = 0; i < n_res; ++i) {
        R_S[i] = d(gen); 
        Q[i] = d(gen); 
        freq_R[i] = d(gen); 
    }
    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("interp_kick");
    auto start = chrono::high_resolution_clock::now();
    // main loop
    // __itt_resume();
    for (int i = 0; i < n_turns; ++i) {
        fast_resonator_v0(impReal.data(), impImag.data(), freq.data(),
                              R_S.data(), Q.data(), freq_R.data(), n_res, 
                              n_points);
    }
    // __itt_detach();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("fast_resonator_v0\ttime(ms)\t%d\t0\t1\n", duration);
    printf("impReal: %lf\n", accumulate(impReal.begin(), impReal.end(), 0.0)/n_points);
    printf("impImag: %lf\n", accumulate(impImag.begin(), impImag.end(), 0.0)/n_points);
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}