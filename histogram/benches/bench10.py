import numpy as np
import sys
import time
import os


def histogram(dt, cut_left, cut_right, n_slices,
              n_particles):

    inv_bin_width = n_slices / (cut_right - cut_left)

    # profile = np.zeros(n_slices, float)
    idx = np.where((dt >= cut_left) & (dt <= cut_right))

    fbin = np.array((dt[idx] - cut_left) * inv_bin_width, int)

    return np.histogram(fbin, bins=n_slices, range=(0., n_slices-1))[0]
    # for i in range(n_particles):
    #     a = dt[i]
    #     if (a < cut_left) or (a > cut_right):
    #         continue
    #     fbin = int((a - cut_left) * inv_bin_width)
    #     profile[fbin] += 1.


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

    dt, _ = np.genfromtxt('../input_files/distribution_10M_particles.txt',
                          unpack=True, skip_header=1)
    dt = dt[:n_particles]

    profile = np.zeros(n_slices, float)

    cut_left = 1.05 * min(dt)
    cut_right = 0.95 * max(dt)

    begin = time.time()

    for i in range(n_turns):
        profile = histogram(dt, cut_left, cut_right, n_slices,
                            n_particles)

    end = time.time()
    duration = (end - begin) * 1000
    print("function\tcounter\taverage_value\tstd(%%)\tcalls\n")
    print("histogram_v11\ttime(ms)\t%d\t0\t1\n" % duration)
    print("profile: %lf\n" % (np.sum(profile) / n_slices))
