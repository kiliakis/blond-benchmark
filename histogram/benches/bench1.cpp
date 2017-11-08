#include <stdlib.h>
#include <stdio.h>
#include "histogram.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
#include <PAPIProf.h>
#include <omp.h>

using namespace std;

int main(int argc, char const *argv[])
{
    int n_turns = 50000;
    int n_particles = 1000000;
    int n_slices = 1000;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) n_slices = atoi(argv[3]);
    if (argc > 4) n_threads = atoi(argv[4]);
    omp_set_num_threads(n_threads);
    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dt;
    vector<double> profile;
    double cut_left, cut_right;
    profile.resize(n_slices);
    dt.resize(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        dt[i] = 10e-6 * d(gen);
    }

    cut_left = dt[rand() % n_slices];
    cut_right = dt[rand() % n_slices];
    if (cut_left > cut_right) swap(cut_left, cut_right);

    auto papiprof = new PAPIProf();
    papiprof->start_counters("histogram");
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        histogram_v0(dt.data(), profile.data(),
                     cut_left, cut_right,
                     n_slices, n_particles);
    }
    papiprof->stop_counters();
    papiprof->report_timing();
    // report results

    return 0;
}