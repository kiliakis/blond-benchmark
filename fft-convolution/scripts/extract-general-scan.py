#!/usr/bin/python
import os
import csv
import sys
import numpy as np
from extract.extract_utilities import *

application = 'fft_convolution'
header = ['version', 'cc', 'vec', 'tcm', 'turns', 'signalLen', 'kernelLen',
          'threads', 'time(ms)', 'std(%)']


def extract_results(input, outfile):
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    records = []
    for dirs, subdirs, files in os.walk(input):
        for file in files:
            if('.txt' not in file):
                continue
            times = []
            print(dirs+'/'+file)
            turns = string_between(file, 'i', '-')
            signalLen = string_between(file, 's', '-')
            kernelLen = string_between(file, 'k', '-')
            threads = string_between(file, 't', '-')
            cc = file.split('-')[4]
            vec = file.split('-')[5]
            tcm = file.split('-')[6].split('.txt')[0]
            for line in open(os.path.join(dirs, file), 'r'):
                line = get_line_matching(line, [application, 'convolution'])
                if not line:
                    continue
                line = line.split('\t')
                app = line[0].split(application)[1][1:]
                time = line[2]
                times.append(float(time))
            if times:
                records.append([app, cc, vec, tcm, turns, signalLen, kernelLen,
                                threads, '%.1lf' % np.mean(times),
                                '%.1lf' % (100 * np.std(times) / np.mean(times))])
    # print(records)
    records.sort(key=lambda a: (a[0], a[1], a[2], a[3],
                                int(a[4]), int(a[5]), int(a[6]), int(a[7])))
    out = open(outfile, 'w')
    writer = csv.writer(out, delimiter='\t')
    writer.writerow(header)
    writer.writerows(records)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("You should specify input directory and output file")
        exit(-1)
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    extract_results(input_dir, output_file)
