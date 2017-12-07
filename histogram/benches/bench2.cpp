#include <stdlib.h>
#include <stdio.h>
#include "histogram.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
#include <PAPIProf.h>
#include <omp.h>
#include <string>
#include <algorithm>

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
    uniform_real_distribution<FTYPE> d(0.0, 1.0);

    // initialize variables
    vector<FTYPE> dt, dE;
    vector<HIST_T> profile;
    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);
    FTYPE cut_left, cut_right;
    profile.resize(n_slices);
    cut_left = 1.05 * (*min_element(dt.begin(), dt.end()));
    cut_right = 0.95 * (*max_element(dt.begin(), dt.end()));
    // cut_left = dt[rand() % n_slices];
    // cut_right = dt[rand() % n_slices];
    if (cut_left > cut_right) swap(cut_left, cut_right);

    auto papiprof = new PAPIProf();
    papiprof->start_counters("histogram_v2");
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        histogram_v2(dt.data(), profile.data(),
                     cut_left, cut_right,
                     n_slices, n_particles);
    }
    papiprof->stop_counters();
    papiprof->report_timing();
    // report results

    return 0;
}