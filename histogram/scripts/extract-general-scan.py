#!/usr/bin/python
import os
import csv
import sys
import numpy as np
from extract.extract_utilities import *


def extract_results(input, outfile):
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    header = ['version', 'turns', 'points', 'slices', 'threads',
              'time(ms)', 'std(%)']
    records = []
    for dirs, subdirs, files in os.walk(input):
        for file in files:
            if('.txt' not in file):
                continue
            times = []
            print(file)
            turns = string_between(file, 'i', '-')
            points = string_between(file, 'p', '-')
            slices = string_between(file, 's', '-')
            threads = string_between(file, 't', '.')
            for line in open(os.path.join(dirs, file), 'r'):
                line = get_line_matching(line, ['histogram'])
                if not line:
                    continue
                line = line.split('\t')
                app = line[0]
                time = line[2]
                times.append(float(time))
            if times:
                records.append([app, turns, points, slices, threads,
                                '%.2lf' % np.mean(times),
                                '%.2lf' % (100 * np.std(times) / np.mean(times))])
    # print(records)
    records.sort(key=lambda a: (a[0], int(a[1]),
                                int(a[2]), int(a[3]), int(a[4])))
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
