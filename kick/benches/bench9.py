import numpy as np
import sys
import time
import os

def kick(dt, dE, n_rf, voltage, omega_rf, phi_rf, n_particles, acc_kick):

    for i in range(n_rf):
        dE += voltage[i] * np.sin(omega_rf[i]*dt + phi_rf[i])

    dE += acc_kick


if __name__ == '__main__':
    n_turns = 5000
    n_particles = 1000000
    n_rf = 4
    n_threads = '1'

    if len(sys.argv) > 1:
        n_turns = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_particles = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_rf = int(sys.argv[3])
    if len(sys.argv) > 4:
        n_threads = sys.argv[4]

    os.environ['OMP_NUM_THREADS'] = n_threads

    np.random.seed(0)
    dE = 10e6 * np.random.randn(n_particles)
    dt = 10e-6 * np.random.randn(n_particles)
    
    voltage = np.random.randn(n_rf)
    omega_rf = np.random.randn(n_rf)
    phi_rf = np.random.randn(n_rf)

    acc_kick = 10e6 * np.random.rand()

    begin = time.time()

    for i in range(n_turns):
        kick(dt, dE, n_rf, voltage, omega_rf,
             phi_rf, n_particles, acc_kick)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n")
    print("kick_v9\ttime(ms)\t%d\t0\t1\n" % duration)
    print("dE: %lf\n" % (np.sum(dE) / n_particles))
