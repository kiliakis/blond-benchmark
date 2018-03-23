import numpy as np
import sys
import time
import os

def sync_rad(dt, dE, U0, n_particles,
             sigma_dE, tau_z, energy, n_kicks):

    dE += - (2.0 / tau_z * dE +
             U0 - 2.0 * sigma_dE / np.sqrt(tau_z)
             * energy * np.random.randn(n_particles))

if __name__ == '__main__':
    n_turns = 5000
    n_particles = 1000000
    n_threads = '1'
    n_kicks = 1

    if len(sys.argv) > 1:
        n_turns = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_particles = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_threads = sys.argv[3]

    np.random.seed(0)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    np.random.seed(0)
    dt, dE = np.genfromtxt('../input_files/distribution_10M_particles.txt',
                           unpack=True, skip_header=1)
    dt = dt[:n_particles]
    dE = dE[:n_particles]

    U0 = 754257950.345
    sigma_dE = 0.00142927197106
    tau_z = 232.014940939
    energy = 175000000000.0

    begin = time.time()

    for i in range(n_turns):
        sync_rad(dt, dE, U0, n_particles,
                 sigma_dE, tau_z, energy, n_kicks)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n")
    print("sync_rad_v10\ttime(ms)\t%d\t0\t1\n" % duration)
    print("dE: %lf\n" % np.mean(dE))
