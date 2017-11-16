#include <stdlib.h>
#include <stdio.h>
#include "synchrotron_radiation.h"
#include "utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <PAPIProf.h>
#include <omp.h>
#include <algorithm>

using namespace std;

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
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dE, dt;
    double U0, sigma_dE, tau_z, energy;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);

    U0 = d(gen);
    sigma_dE = d(gen);
    tau_z = d(gen);
    energy = d(gen);

    auto papiprof = new PAPIProf();
    // main loop
    papiprof->start_counters("synchrotron_radiation_mkl");
    // printf("I am about to start\n");
    for (int i = 0; i < n_turns; ++i) {
        synchrotron_radiation_full_v2(dE.data(), U0, n_particles,
                                      sigma_dE, tau_z, energy, n_kicks);
    }
    papiprof->stop_counters();
    papiprof->report_timing();

    return 0;
}