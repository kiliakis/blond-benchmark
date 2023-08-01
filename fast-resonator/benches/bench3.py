import numpy as np
import sys
import time
import os


def fast_resonator(impedance, freq, R_S, Q, freq_R):

    for i in range(len(R_S)):
        impedance[1:] += R_S[i] / (1 + 1j * Q[i]
                                   * (freq[1:] / freq_R[i] -
                                       freq_R[i] / freq[1:]))


if __name__ == '__main__':
    n_turns = 5000
    n_points = 1000000
    n_res = 1000
    n_threads = 1

    if len(sys.argv) > 1:
        n_turns = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_points = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_res = int(sys.argv[3])
    if len(sys.argv) > 4:
        n_threads = int(sys.argv[4])

    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    np.random.seed(0)
    impedance = np.zeros(n_points, np.complex128)
    freq = np.random.randn(n_points)

    R_S = np.random.randn(n_res)
    Q = np.random.randn(n_res)
    freq_R = np.random.randn(n_res)

    begin = time.time()

    for i in range(n_turns):
        fast_resonator(impedance, freq, R_S, Q, freq_R)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n")
    print("fast_resonator_py\ttime(ms)\t%d\t0\t1\n" % duration)
    print("impReal: %lf\n" % (np.sum(impedance.real) / n_points))
    print("impImag: %lf\n" % (np.sum(impedance.imag) / n_points))
