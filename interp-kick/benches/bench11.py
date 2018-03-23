import numpy as np
import sys
import time
import os


def interp_kick(dt, dE, voltage, bin_centers, n_slices,
                n_particles, acc_kick):

    inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])


    idx = np.where((dt >= bin_centers[0]) & (dt <= bin_centers[-1]))
    
    fbin = np.array(np.floor((dt[idx] - bin_centers[0]) * inv_bin_width), int)

    voltageKick = np.zeros(len(dE), float)
    voltageKick[idx] = voltage[fbin] + (dt[idx] - bin_centers[fbin]) \
        * (voltage[fbin+1] - voltage[fbin]) * inv_bin_width
    
    dE += voltageKick + acc_kick

    # fbin = np.array((dt[idx] - cut_left) * inv_bin_width, int)

    # return np.histogram(fbin, bins=n_slices, range=(0., n_slices-1))[0]

    # for i in range(n_particles):
    #     a = dt[i]
    #     fbin = int(np.floor((a-bin_centers[0]) * inv_bin_width))
    #     if (a < bin_centers[0]) or (a > bin_centers[-1]):
    #         voltageKick = 0.
    #     else:
    #         voltageKick = voltage[fbin] + (a - bin_centers[fbin]) \
    #             * (voltage[fbin+1] - voltage[fbin]) * inv_bin_width
    #     dE[i] += voltageKick + acc_kick


if __name__ == '__main__':
    n_turns = 5000
    n_particles = 1000000
    n_slices = 1000
    n_threads = 1

    if len(sys.argv) > 1:
        n_turns = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_particles = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_slices = int(sys.argv[3])
    if len(sys.argv) > 4:
        n_threads = int(sys.argv[4])

    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    np.random.seed(0)
    dt, dE = np.genfromtxt('../input_files/distribution_10M_particles.txt',
                           unpack=True, skip_header=1)
    dt = dt[:n_particles]
    dE = dE[:n_particles]
    voltage = np.random.randn(n_slices)
    cut_left = 1.05 * min(dt)
    cut_right = 0.95 * max(dt)
    acc_kick = np.random.rand()

    edges = np.linspace(cut_left, cut_right, n_slices + 1)
    bin_centers = (edges[:-1] + edges[1:])/2

    begin = time.time()

    for i in range(n_turns):
        interp_kick(dt, dE, voltage, bin_centers, n_slices,
                    n_particles, acc_kick)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n")
    print("interp_kick_v11\ttime(ms)\t%d\t0\t1\n" % duration)
    print("dE: %lf\n" % (np.sum(dE) / n_particles))
