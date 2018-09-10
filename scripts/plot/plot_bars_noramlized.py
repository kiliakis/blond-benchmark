#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
from plot.plotting_utilities import *


hatces=['x', '\\', '/']
colors=['.25', '.5', '.75']

home_dir = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/'
csv_file = home_dir + 'results/pyintel_vs_pynormal.csv'
image_name = home_dir + '/results/plots/pyintel_vs_normal.pdf'

y_label = 'Normalized Time (Anaconda)'
x_label = 'Applications'
title = 'Runtime comparison of the Intel and Anaconda distribution'
y_name = 'time(ms)'
names = {
    # 'word_count': 'WC',
    # 'linear_regression': 'LR',
    # 'mr_matrix_multiply': 'MM',
    # 'pca': 'PCA',
    # 'histogram': 'Hist',
    # 'kmeans': 'KMeans'
}


# def plot(x, y, label='', yerr=None):

#     plt.grid(True, which='major', alpha=1)
#     # plt.grid(True, which='minor', alpha=0.8)
#     # plt.minorticks_on()
#     plt.errorbar(x, y, yerr=yerr, marker='o', linewidth='1', label=label)



if __name__ == '__main__':
    data = np.genfromtxt(csv_file, delimiter='\t', dtype=str)
    header = list(data[0])
    data = data[1:]

    plots_dir = get_plots(header, data, {'cc': ['intel', 'normal']})

    print(plots_dir)

    ref = plots_dir['normal']

    plt.figure()
    plt.grid('on')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    xticks = np.append(ref[:,0], ['mean'])
    N = len(xticks)
    width = 0.4
    ind = np.linspace(0, N, N)

    space = 0
    for k in ['normal', 'intel']:
        v = plots_dir[k]
        y = 100.0 * np.array(v[:, 2], float) / np.array(ref[:, 2], float)
        y = np.append(y, [np.mean(y)])
        a = plt.bar(ind + space, y, width, label=k)
        if k == 'intel':
            autolabel(plt.gca(), a, 1)
        space += width

    plt.legend(loc='lower right', fancybox=True)
    plt.xticks(ind+width/2, xticks, rotation=30)
    plt.tight_layout()
    plt.savefig(image_name, bbox_inches='tight', dpi=300)
    plt.show()

    # fig, ax = plt.subplots(figsize=(8, 4.5))
    # # plt.xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.set_title(title)
    # # ax.grid(True, which='major', alpha=1)
    # # if(x_lims):
    # #     plt.xlim(x_lims)
    # # if(y_lims):
    # #     plt.ylim(y_lims)
    # # ax.set_ylim(0, 2.)
    # xticks = []
    # normalized = []
    # for dirpath, dirnames, filenames in os.walk(input_folder):
    #     for file in filenames:
    #         if('timings.csv' not in file):
    #             continue
    #         # print file
    #         # print reference_folder + file
    #         values = import_results(dirpath + file)
    #         header = values[0]
    #         values = np.array(values[1:])
    #         c = header.index(y_name)
    #         real = values[0, c].astype(float)
    #         # c = header.index(yerr_name)
    #         # real_err = values[0, c].astype(float)

    #         values = import_results(reference_folder + file)
    #         header = values[0]
    #         values = np.array(values[1:])
    #         c = header.index(y_name)
    #         reference = values[0, c].astype(float)
    #         # c = header.index(yerr_name)
    #         # reference_err = values[0, c].astype(float)
    #         xticks.append(names[file.split('_timings')[0]])
    #         normalized.append(reference/real)
    # xticks = np.array(xticks)
    # normalized = np.array(normalized)
    # # print xticks
    # # print normalized
    # # for x in xticks:
    # #     if(x in names):
    # #         xticks[xticks.index(x)] = names[x]
    # args = xticks.argsort()
    # xticks = xticks[args]
    # normalized = normalized[args]
    # N = len(xticks)
    # width = 0.4
    # ind = np.linspace(0.15, 1.3 * N, N)
    # opacity = 0.9
    # # ind = np.arange(N)
    # dynamic = ax.bar(ind, normalized, width, label='boost-dynamic', color=colors.pop(), hatch=hatces.pop(),
    #                  alpha=opacity)
    # static = ax.bar(ind + width, [1.0] * N,
    #                 width, label='boost-static', color=colors.pop(), hatch=hatces.pop(), alpha=opacity)

    # ax.set_xticks(ind + width)
    # ax.set_xticklabels(xticks)
    # ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize='11')

    # autolabel(dynamic)
    # # autolabel(static)

    # if show == 'show':
    #     plt.show()
    # else:
    #     fig.tight_layout()
    #     for image_name in image_names:
    #         fig.savefig(image_name, bbox_inches='tight')
    # plt.close()
