#include <stdlib.h>
#include <stdio.h>
#include "overlap_track.h"
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

    int n_turns = 5000;
    int n_particles = 1000000;
    int n_slices = 1000;
    int alpha_order = 0;

    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) n_slices = atoi(argv[3]);
    if (argc > 4) alpha_order = atoi(argv[4]);
    if (argc > 5) n_threads = atoi(argv[5]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<double> d(0.0, 1.0);

    // initialize variables
    vector<double> dE, dt;
    vector<double> voltage, edges, bin_centers, profile;
    double cut_left, cut_right, acc_kick;
    double T0, length_ratio, eta0, eta1, eta2;
    double beta, energy, charge;

    T0 = 8.89246551651e-05;
    length_ratio = 1.0;
    eta0 = 0.000317286758678;
    eta1 = 0.000317286758678;
    eta2 = 0.000317286758678;
    beta = 0.999997826292;
    energy = 450000978171.0;
    charge = 1.0;
    acc_kick = 0;

    const char *solver = alpha_order > 0 ? "full" : "simple";

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

    if (cut_left > cut_right) swap(cut_left, cut_right);

    profile.resize(n_slices, 0.0);

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
    // __itt_resume();
    for (int i = 0; i < n_turns; ++i) {
        overlap_track_v3(dt.data(), dE.data(), profile.data(),
                         voltage.data(), bin_centers.data(),
                         n_slices, n_particles,
                         acc_kick, solver, T0, length_ratio,
                         alpha_order, eta0, eta1, eta2,
                         beta, energy, charge, cut_left, cut_right);
    }
    // __itt_detach();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("overlap_track_v3\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0) / n_particles);
    printf("dt: %e\n", accumulate(dt.begin(), dt.end(), 0.0) / n_particles);
    printf("profile: %lf\n", accumulate(profile.begin(), profile.end(), 0.0));

    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}