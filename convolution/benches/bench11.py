import numpy as np
import sys
import time
import os

def convolution(signal, kernel):
    return np.convolve(signal, kernel, mode='full') 

if __name__ == '__main__':
    n_turns = 5000
    n_signal = 1000
    n_kernel = 1000
    n_threads = '1'

    if len(sys.argv) > 1:
        n_turns = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_signal = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_kernel = int(sys.argv[3])
    if len(sys.argv) > 4:
        n_threads = sys.argv[4]

    os.environ['OMP_NUM_THREADS'] = n_threads 
    print('turns\tn_signal\tn_kernel\tn_threads')
    print('%d\t%d\t%d\t%s' % (n_turns, n_signal, n_kernel, n_threads))
    np.random.seed(0)
    signal = np.random.randn(n_signal)
    kernel = np.random.randn(n_kernel)
    result = np.empty(n_signal + n_kernel -1)
    
    begin = time.time()

    for i in range(n_turns):
        result = convolution(signal, kernel)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n")
    print("convolution_v11\ttime(ms)\t%d\t0\t1\n" % duration)
    print("result: %lf\n" % np.mean(result))
