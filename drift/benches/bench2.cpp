#include <stdlib.h>
#include <stdio.h>
#include "drift.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
// #include <PAPIProf.h>
#include <omp.h>
#include <algorithm>

using namespace std;

int main(int argc, char const *argv[])
{
    int n_turns = 5000;
    int n_particles = 1000000;
    int alpha_order = 0;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) alpha_order = atoi(argv[3]);
    if (argc > 4) n_threads = atoi(argv[4]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<float> d(0.0, 1.0);

    // initialize variables
    vector<float> dE, dt;
    float T0, length_ratio, eta0, eta1, eta2;
    float beta, energy;

    dE.resize(n_particles); dt.resize(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        dE[i] = 10e6 * d(gen);
        dt[i] = 10e-6 * d(gen);
    }
    T0 = d(gen);
    length_ratio = d(gen);
    eta0 = d(gen); eta1 = d(gen); eta2 = d(gen);
    beta = d(gen); energy = d(gen);
    const char *solver = alpha_order > 0 ? "full" : "simple";
    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("drift");
    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        drift_v2(dt.data(), dE.data(), solver,
                 T0, length_ratio, alpha_order, eta0,
                 eta1, eta2, beta, energy,
                 n_particles);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("drift_v2\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dt: %lf\n", accumulate(dt.begin(), dt.end(), 0.0) / (n_particles));
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}