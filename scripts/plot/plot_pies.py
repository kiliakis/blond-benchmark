#!/usr/bin/python
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
import os
import numpy as np
# import sys
# import csv
import fnmatch
from plot.plotting_utilities import *

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/'
testcases = ['kick', 'drift', 'interp-kick', 'convolution',
             'histogram', 'fft-convolution', 'synchrotron-radiation']
result_dir = home + '{}/results/csv/run0/'
image_name = home + '/results/plots/run0/{}_{}.pdf'
key_names = ['metric']
prefixes = ['']
to_keep = ['value']
show = 0
# xlabel = ['Threads']
# ylabel = ['Run time %']
title = '{} CPI:{}'

names = {
}

serie = ['FE_Bound', 'Bad_Speculation', 'MEM_Bound', 'Core_Bound', 'Retiring']


def plot_pie(tc, input_file):
    print(input_file)
    data = np.genfromtxt(input_file, dtype=str, delimiter='\t')
    header = data[0]
    data = data[1:]
    keys = data[:, 0].tolist()
    values = data[:, 1].tolist()
    CPI = values[keys.index('CPI')]
    del values[keys.index('CPI')]
    keys.remove('CPI')
    values = np.array(values, float)

    # plt.figure(figsize=(6.5, 4))
    plt.figure()
    # plt.grid(True, which='major', alpha=0.5)
    plt.xlabel(title.format(tc, CPI), fontsize=11)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0., 1., len(keys)))
    explode = [0] * len(keys)
    patches, texts, autotexts = plt.pie(values, shadow=False, colors=colors,
                                        counterclock=False,
                                        autopct='%1.1f%%',
                                        textprops={'fontsize': '10'},
                                        startangle=0,
                                        explode=explode)
    for t in autotexts:
        if(float(t.get_text().split('%')[0]) < 3):
            t.set_text('')
    # autotexts[0].set_color('w')
    plt.axis('equal')
    # plt.subplot(grid[0, 0])
    plt.legend(keys, loc='upper center', fancybox=True,
               framealpha=0.4, ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.05))
    # plt.legend(labels, loc='upper center', bbox_to_anchor=(-0.6, 2.4), ncol=5,
    #            fancybox=True, fontsize=8, framealpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        img = image_name.format(
            tc, input_file.split('/')[-1].split('.csv')[0])
        plt.savefig(img, bbox_inches='tight')

    plt.close()


if __name__ == '__main__':
    for tc in testcases:
        files = fnmatch.filter(os.listdir(result_dir.format(tc)), '*.csv')
        plot_pie(tc, result_dir.format(tc)+files[0])
