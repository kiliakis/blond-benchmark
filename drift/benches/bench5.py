import numpy as np
import sys
import time
import os

def drift(dt, dE, solver, T0, length_ratio,
          alpha_order, eta0, eta1,
          eta2, beta, energy, n_particles):

    T = T0 * length_ratio
    if (solver == 'simple'):
        coeff = T * eta0 / (beta * beta * energy)
        dt += coeff * dE
    else:
        coeff = 1. / (beta * beta * energy)
        eta0 = eta0 * coeff
        eta1 = eta1 * coeff**2
        eta2 = eta2 * coeff**3

        if (alpha_order == 1):
            dt += T * (1. / (1. - eta0 * dE) - 1.)
        elif (alpha_order == 2):
            dt += T * (1. / (1. - eta0 * dE - eta1 * dE**2) - 1.)
        else:
            dt += T * (1. / (1. - eta0 * dE - eta1 * dE**2 - eta2 * dE**3) - 1.)


if __name__ == '__main__':
    n_turns = 5000
    n_particles = 1000000
    alpha_order = 0
    n_threads = '1'
    if len(sys.argv) > 1:
        n_turns = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_particles = int(sys.argv[2])
    if len(sys.argv) > 3:
        alpha_order = int(sys.argv[3])
    if len(sys.argv) > 4:
        n_threads = sys.argv[4]

    os.environ['OMP_NUM_THREADS'] = n_threads
    np.random.seed(0)


    dE = 10e6 * np.random.randn(n_particles)
    dt = 10e-6 * np.random.randn(n_particles)
    T0 = np.random.rand()
    length_ratio = np.random.rand()
    eta0 = 1e-3 * np.random.rand()
    eta1 = 1e-3 * np.random.rand()
    eta2 = 1e-3 * np.random.rand()
    beta = np.random.rand()
    energy = np.random.rand()
    if alpha_order > 0:
        solver = 'full'
    else:
        solver = 'simple'

    begin = time.time()

    for i in range(n_turns):
        drift(dt, dE, solver, T0, length_ratio,
              alpha_order, eta0, eta1, eta2,
              beta, energy, n_particles)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n");
    print("drift_v5\ttime(ms)\t%d\t0\t1\n" % duration);
    print("dt: %lf\n" % (np.sum(dt) / n_particles));
