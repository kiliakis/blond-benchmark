#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os

from plot.plotting_utilities import *

application = 'convolution'
project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/convolution1/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

csv_file = res_dir + 'csv/convolution1/all_results.csv'

plots_config = {
    'plot1': {'lines': {'version': ['v0', 'v1', 'v3'],
                        'vec': ['vec'],
                        'tcm': ['tcm', 'notcm'],
                        'cc': ['icc']},
              'exclude': [['v3', 'notcm']],
              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (4k points/thread X 4k)',
              'ylabel': 'Run-time (ms)',
              'title': 'Convolution Optimizations',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'convolution_opts.pdf'
              },

    'plot2': {'lines': {'version': ['v3', 'v5', 'v9'],
                        'vec': ['vec'],
                        'tcm': ['tcm', 'notcm'],
                        'cc': ['icc']},
              'exclude': [['v9', 'notcm']],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (4k points/thread X 4k)',
              'ylabel': 'Run-time (ms)',
              'title': 'Parallel MKL convolution',
              'extra': ['plt.xscale(\'log\', basex=2)',
                        'plt.yscale(\'log\', basex=10)'],
              'image_name': images_dir + 'parallel_mkl.pdf'
              },

    'plot3': {'lines': {'version': ['v0','v7'],
                        'vec': ['vec'],
                        'tcm': ['notcm'],
                        'cc': ['g++', 'icc']},
              'exclude': [],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (4k points/thread X 4k)',
              'ylabel': 'Run-time (ms)',
              'title': 'Custom Convolution Single VS Double precision',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'custom_single_vs_double.pdf'
              },

    'plot4': {'lines': {'version': ['v5', 'v8'],
                        'vec': ['vec'],
                        'tcm': ['tcm'],
                        'cc': ['icc']},
              'exclude': [],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (4k points/thread X 4k)',
              'ylabel': 'Run-time (ms)',
              'title': 'Parallel MKL Single VS Double Precision',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'mkl_single_vs_double.pdf'
              }
}

if __name__ == '__main__':
    data = np.genfromtxt(csv_file, delimiter='\t', dtype=str)
    header = list(data[0])
    data = data[1:]
    for plot_key, config in plots_config.items():
        print(plot_key)
        plots_dir = get_plots(
            header, data, config['lines'], exclude=config['exclude'])
        # print(plots_dir)
        plt.figure()
        plt.grid('on')
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        if 'extra' in config:
            for c in config['extra']:
                exec(c)
        for label, values in plots_dir.items():
            # print(values)
            x = np.array(values[:, header.index(config['x_name'])], float)
            y = np.array(values[:, header.index(config['y_name'])], float)
            y_err = np.array(
                values[:, header.index(config['y_err_name'])], float)
            y_err = y_err * y / 100.
            print(label, x, y)
            plt.errorbar(x, y, yerr=y_err, label=label, capsize=2, marker='o')
        plt.legend(loc='best', fancybox=True)
        plt.tight_layout()
        plt.savefig(config['image_name'])
        plt.show()
        plt.close()
