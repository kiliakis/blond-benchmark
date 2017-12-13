# import matplotlib.pyplot as plt
# import ctypes
import os
import numpy as np
from pyprof.timing import report, start_timing, stop_timing
from include.fft_convolution import *
# from pyprof.papiprof import PAPIProf
# papiprof = PAPIProf(metrics=['IPC', 'RESOURCE_STALLS_COST'])


if __name__ == "__main__":
    n_turns = 5000
    n_signal = 10000
    n_kernel = 10000

    import sys
    if(len(sys.argv) > 1):
        n_turns = int(sys.argv[1])
    if(len(sys.argv) > 2):
        n_signal = int(sys.argv[2])
    if(len(sys.argv) > 3):
        n_kernel = int(sys.argv[3])
    if(len(sys.argv) > 4):
        n_threads = sys.argv[4]
        os.environ['OMP_NUM_THREADS'] = n_threads

    np.random.seed(0)
    signal = np.random.randn(n_signal)
    kernel = np.random.randn(n_kernel)

    result = np.zeros(len(signal) + len(kernel) - 1)
    result = convolution_v0(signal, kernel)

    start_timing('convolution_v1')
    for i in range(n_turns):
        result = convolution_v0(signal, kernel)
    stop_timing()

    report()
