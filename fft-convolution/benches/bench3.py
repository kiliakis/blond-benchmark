import numpy as np
from pyprof.timing import report, start_timing, stop_timing
from include.fft_convolution import *
# from pyprof.papiprof import PAPIProf
# papiprof = PAPIProf(metrics=['IPC', 'RESOURCE_STALLS_COST'])


if __name__ == "__main__":
    n_turns = 5000
    n_signal = 10000
    n_kernel = 10000
    n_threads = 1
    import sys
    if(len(sys.argv) > 1):
        n_turns = int(sys.argv[1])
    if(len(sys.argv) > 2):
        n_signal = int(sys.argv[2])
    if(len(sys.argv) > 3):
        n_kernel = int(sys.argv[3])
    if(len(sys.argv) > 4):
        n_threads = int(sys.argv[4])

    np.random.seed(0)
    signal = np.random.randn(n_signal)
    kernel = np.random.randn(n_kernel)

    result = np.zeros(len(signal) + len(kernel) - 1, dtype=float)
    convolution_v1(signal, kernel, result, n_threads)

    start_timing('convolution_v1')
    for i in range(n_turns):
        convolution_v1(signal, kernel, result, n_threads)
    stop_timing()

    report()
