#include <stdlib.h>
#include <stdio.h>
#include "kick.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>
// #include <PAPIProf.h>
#include <omp.h>
// #include <ittnotify.h>
using namespace std;

int main(int argc, char const *argv[])
{
    int n_turns = 5000;
    int n_particles = 1000000;
    int n_rf = 4;
    int n_threads = 1;

    if (argc > 1) n_turns = atoi(argv[1]);
    if (argc > 2) n_particles = atoi(argv[2]);
    if (argc > 3) n_rf = atoi(argv[3]);
    if (argc > 4) n_threads = atoi(argv[4]);
    omp_set_num_threads(n_threads);

    // setup random engine
    default_random_engine gen;
    uniform_real_distribution<float> d(0.0, 1.0);

    // initialize variables
    vector<float> dE, dt;
    vector<float> voltage, omega_rf, phi_rf;
    float acc_kick;

    dE.resize(n_particles); dt.resize(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        dE[i] = 10e6 * d(gen);
        dt[i] = 10e-6 * d(gen);
    }

    voltage.resize(n_rf);
    omega_rf.resize(n_rf);
    phi_rf.resize(n_rf);
    for (int i = 0; i < n_rf; ++i) {
        voltage[i] = d(gen);
        omega_rf[i] = d(gen);
        phi_rf[i] = d(gen);
    }
    acc_kick = 10e6 * d(gen);

    // auto papiprof = new PAPIProf();
    // papiprof->start_counters("kick_v2");
    // main loop
    // __itt_resume();
    auto start = chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed_time(0.0);
    // start = chrono::system_clock::now();
    for (int i = 0; i < n_turns; ++i) {
        kick_v3(dt.data(), dE.data(), n_rf,
                voltage.data(), omega_rf.data(), phi_rf.data(),
                n_particles, acc_kick);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    printf("kick_v3\ttime(ms)\t%d\t0\t1\n", duration);
    printf("dE: %lf\n", accumulate(dE.begin(), dE.end(), 0.0)/n_particles);
    
    // elapsed_time = chrono::system_clock::now() - start;
    // __itt_pause(); // stop VTune
    // papiprof->stop_counters();
    // papiprof->report_timing();
    // report results

    return 0;
}