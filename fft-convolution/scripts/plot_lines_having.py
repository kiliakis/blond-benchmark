#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os

from plot.plotting_utilities import *

application = 'fft-convolution'
project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/fft-convolution1/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

csv_file = res_dir + 'csv/fft-convolution1/all_results.csv'

plots_config = {
    'plot1': {'lines': {'version': ['v1', 'v2'],
                        'vec': ['vec', 'novec', 'na'],
                        'tcm': ['tcm', 'na'],
                        'cc': ['g++', 'na']},
              'exclude': [],
              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points (50k signalLen/thread X 50k kernelLen)',
              'ylabel': 'Run-time (ms)',
              'title': 'Scipy VS FFTW',
              'extra': ['plt.xscale(\'log\', basex=2)',
                        'plt.yscale(\'log\', basex=10)'],

              'image_name': images_dir + 'scipy_vs_fftw.pdf'
              },

    'plot2': {'lines': {'version': ['v2', 'v4'],
                        'vec': ['vec'],
                        'tcm': ['tcm', 'notcm'],
                        'cc': ['g++', 'icc']},
              'exclude': [['v2', 'icc']],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points (50k signalLen/thread X 50k kernelLen)',
              'ylabel': 'Run-time (ms)',
              'title': 'FFTW vs MKL',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'fftw_vs_mkl.pdf'
              },

    'plot3': {'lines': {'version': ['v4', 'v6'],
                        'vec': ['vec'],
                        'tcm': ['notcm'],
                        'cc': ['icc']},
              'exclude': [],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points (50k signalLen/thread X 50k kernelLen)',
              'ylabel': 'Run-time (ms)',
              'title': 'Single VS Double Precision',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'single_vs_double.pdf'
              },
    'plot4': {'lines': {'version': ['v7', 'v8', 'v9', 
                                    'v7-p100', 'v8-p100', 'v9-p100'],
                        'cc': ['nvcc']},
              'exclude': [],
              'x_name': 'signalLen',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points (50k signalLen/thread X 50k kernelLen)',
              'ylabel': 'Run-time (ms)',
              'title': 'All GPU Versions',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'all_gpu_versions.pdf'
              },
    'plot5': {'lines': {'version': ['v72', 'v82', 'v92'],
                        'cc': ['nvcc']},
              'exclude': [],
              'x_name': 'signalLen',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points (50k X 50k) points/thread',
              'ylabel': 'Run-time (ms)',
              'title': 'All GPU Versions',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'all_gpu_versions_2.pdf'
              },
    'plot6': {'lines': {'version': ['v9', 'v4', 'v9-p100'],
                        'cc': ['nvcc', 'icc'],
                        'tcm': ['notcm', 'na'],
                        'vec': ['vec', 'na']},
              'exclude': [],
              'x_name': 'signalLen',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points (50k signalLen/thread X 50k kernelLen)',
              'ylabel': 'Run-time (ms)',
              'title': 'GPU vs CPU',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'gpu_vs_cpu.pdf'
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
        for label, values in plots_dir.items():
            # print(values)
            x = np.array(values[:, header.index(config['x_name'])], float)
            y = np.array(values[:, header.index(config['y_name'])], float)
            y_err = np.array(
                values[:, header.index(config['y_err_name'])], float)
            y_err = y_err * y / 100.
            print(label, x, y)
            plt.errorbar(x, y, yerr=y_err, label=label, capsize=2, marker='o')
        if 'extra' in config:
            for c in config['extra']:
                exec(c)
        if plot_key == 'plot6':
            plt.gca().get_lines()
            for p in plt.gca().get_lines()[::3]:
                annotate(plt.gca(), p.get_xdata(), p.get_ydata(), fontsize='8')
        plt.legend(loc='best', fancybox=True)
        plt.tight_layout()
        plt.savefig(config['image_name'])
        plt.show()
        plt.close()

    # plt.legend(loc='best', fancybox=True, fontsize='11')
    # plt.axvline(700.0, color='k', linestyle='--', linewidth=1.5)
    # plt.axvline(1350.0, color='k', linestyle='--', linewidth=1.5)
    # plt.annotate('Light\nCombine\nWorkload', xy=(
    #     200, 6.3), textcoords='data', size='16')
    # plt.annotate('Moderate\nCombine\nWorkload', xy=(
    #     800, 6.3), textcoords='data', size='16')
    # plt.annotate('Heavy\nCombine\nWorkload', xy=(
    #     1400, 8.2), textcoords='data', size='16')
