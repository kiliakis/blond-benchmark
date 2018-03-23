#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import numpy as np
# import fnmatch
from plot.plotting_utilities import *

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/'

rooftops = home + '/results/rooftops.csv'

testcases = ['kick', 'drift', 'interp-kick', 'convolution',
             'histogram', 'fft-convolution', 'synchrotron-radiation']
result_dir = home + '{}/results/csv/roofline2/'
image_name = home + 'results/plots/roofline2/roofline.pdf'
xlabel = 'Operational Intensity (Flops/byte)'
ylabel = 'Performance (GFlops/sec)'
title = 'Roofline Model'

show = 0
xlims = [0.01, 10]
ylims = [0.01, 100]
limits = {
    'DP Vector FMA': [0.1677, 100],
    'DP Vector Add': [0.0418, 100],
    'Scalar Add': [0.0106, 100],
    'L1': [0, 0.1677],
    'L2': [0, 0.547],
    'L3': [0, 1.457],
    'DRAM': [0, 3.71],
}

mem_text_xy = [2**(-6), 2**(-5)]


config = {
    'fft-convolution': {
        'short': 'FFTConv',
        'marker': 'o'
    },
    'synchrotron-radiation': {
        'short': 'SR',
        'marker': 'v'
    },
    'histogram': {
        'short': 'Hist',
        'marker': '^'
    },
    'convolution': {
        'short': 'Conv',
        'marker': 's'
    },
    'drift': {
        'short': 'Drift',
        'marker': '*'
    },
    'kick': {
        'short': 'Kick',
        'marker': '+'
    },
    'interp-kick': {
        'short': 'LIKick',
        'marker': 'x'
    }
}


def get_flops(tc, file):
    print(file)
    data = np.genfromtxt(file, dtype=float, delimiter=',',
                         skip_header=1, names=True)
    if '.exe' in file:
        bench = file.split('bench')[2].split('.exe')[0]
    elif '.py' in file:
        bench = file.split('bench')[2].split('.py')[0]

    return [tc+'-'+bench, data]


def get_roofs(tc, file):
    print(file)
    data = np.genfromtxt(file, dtype=str, delimiter=',',
                         skip_header=2)
    # print(data)
    return data

from collections import OrderedDict
def plot_roofline(roofs, flops, x_key, y_key):
    # Generate the figure
    # plt.figure(figsize=(6.5, 4))
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xscale('log', basex=10)
    plt.yscale('log', basey=10)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.grid(True, which='major', alpha=0.5)
    # For each roof plot an horizontal or slanded line
    for roof in roofs:
        name = roof[0]
        value = roof[1]
        roof_type = roof[3]
        if 'single-threaded' not in name:
            continue
        if 'SP' in name:
            continue
        name = name.replace('(single-threaded)', '').strip()
        bw = float(value)/1e9
        if roof_type == 'compute':
            x = np.linspace(limits[name][0], limits[name][1], 100)
            c = 'b'
            y = [bw] * len(x)
            plt.text(1, 1.05*bw, name + ("%.1f GFlops" % bw),
                     fontsize=8)
        elif roof_type == 'memory':
            x = np.linspace(limits[name][0], limits[name][1], 100)
            y = bw * x
            c = 'r'
            plt.text(mem_text_xy[0], bw * mem_text_xy[1],
                     name + (' %.1f GB/s' % bw),
                     rotation=27, fontsize=8)
        else:
            print('Error invalid roof type')
            exit(-1)
        plt.plot(x, y, linestyle='-', color=c, linewidth=1.4, alpha=0.5)

    # For each flop plot a point
    for tc, flop in flops.items():
        bench = tc.split('-')[-1]
        tc = '-'.join(tc.split('-')[:-1])
        plt.plot(flop[x_key], flop[y_key], 'o',
                 # label='%s (%.2f, %.1f) ' % (config[tc]['short'] + bench,
                 #                             flop[x_key], flop[y_key]),
                 label=config[tc]['short'],
                 markersize=8, marker=config[tc]['marker'])

    plt.legend(loc='lower right', ncol=2, fancybox=True,
               fontsize=8.5, framealpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
        loc='lower right', ncol=2, fancybox=True,
               fontsize=8.5, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(image_name, bbox_inches='tight')
    plt.show()

    plt.close()


if __name__ == '__main__':
    flops = {}
    roofs = np.genfromtxt(rooftops, str, skip_header=1, delimiter=',')

    for tc in testcases:
        for dirpath, _, files in os.walk(result_dir.format(tc)):
            if 'flops.csv' not in files:
                continue
            flops_file = os.path.join(dirpath, 'flops.csv')
            data = get_flops(tc, flops_file)
            if data:
                flops[data[0]] = data[1]
    # roofs[1] = roofs[1] / len(testcases)
    print(flops)
    plot_roofline(roofs, flops, 'total_ai', 'total_gflops')
