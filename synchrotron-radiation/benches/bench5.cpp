#include <stdlib.h>
#include <stdio.h>
#include "synchrotron_radiation.h"
#include "utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <string>
// #include <PAPIProf.h>
#include <omp.h>
#include <algorithm>
#include <chrono>

using namespace std;

#define USE_BOOST

int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_particles = 1000000;
    const int n_kicks = 1;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    // if (argc > 3) n_kicks = 1;
    if (argc > 3) n_threads = atoi(argv[3]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<float> d(0.0, 1.0);

    // initialize variables
    vector<float> dE, dt;
    float U0, sigma_dE, tau_z, energy;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);

    U0 = 754257950.345;
    sigma_dE = 0.00142927197106;
    tau_z = 232.014940939;
    energy = 175000000000.0;
    // auto papiprof = new PAPIProf();
    // main loop
    // papiprof->start_counters("synchrotron_radiation");
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < n_turns; ++i) {
        synchrotron_radiation_full_v3(dE.data(), U0, n_particles,
                                      sigma_dE, tau_z, energy, n_kicks);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("sync_rad_v6\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0)/n_particles);

    // papiprof->stop_counters();
    // papiprof->report_timing();

    return 0;
}