#include <stdlib.h>
#include <stdio.h>
#include "interp_kick.h"
#include "utils.h"
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <string>
// #include <PAPIProf.h>
#include <omp.h>
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
    uniform_real_distribution<float> d(0.0, 1.0);

    // initialize variables
    vector<float> dE, dt;
    vector<float> voltage, edges, bin_centers;
    float cut_left, cut_right, acc_kick;

    string input = HOME "/input_files/distribution_10M_particles.txt";
    read_distribution(input, n_particles, dt, dE);

    voltage.resize(n_slices);
    for (int i = 0; i < n_slices; ++i) {
        voltage[i] = d(gen);
    }
    cut_left = 1.05 * (*min_element(dt.begin(), dt.end()));
    cut_right = 0.95 * (*max_element(dt.begin(), dt.end()));

    // cut_left = dt[rand() % n_slices];
    // cut_right = dt[rand() % n_slices];
    acc_kick = 10e6 * d(gen);
    if (cut_left > cut_right) swap(cut_left, cut_right);

    edges.resize(n_slices);
    linspace(cut_left, cut_right, n_slices + 1, edges.data());

    bin_centers.resize(n_slices);
    for (int i = 0; i < n_slices; ++i) {
        bin_centers[i] = (edges[i] + edges[i + 1]) / 2.;
    }

    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("interp_kick");
    auto start = chrono::high_resolution_clock::now();
    // main loop
    for (int i = 0; i < n_turns; ++i) {
        linear_interp_kick_v6(dt.data(), dE.data(), voltage.data(),
                              bin_centers.data(), n_slices, n_particles,
                              acc_kick);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("interp_kick_v6\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0)/n_particles);
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}