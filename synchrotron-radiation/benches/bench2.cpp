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
    int n_kicks = 1000;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) n_kicks = atoi(argv[3]);
    if (argc > 4) n_threads = atoi(argv[4]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dE, random_array;
    vector<double> random_array;
    double U0, sigma_dE, tau_z, energy;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, random_array, dE);

    U0 = d(gen);
    sigma_dE = d(gen);
    tau_z = d(gen);
    energy = d(gen);

    auto papiprof = new PAPIProf();
    papiprof->start_counters("synchrotron_radiation");
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        synchrotron_radiation_full_v1(dE.data(), U0, n_particles,
                                      sigma_dE, tau_z, energy,
                                      random_array.data(), n_kicks);
    }
    papiprof->stop_counters();
    papiprof->report_timing();

    return 0;
}