#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os

from plot.plotting_utilities import *

application = 'interp-kick'
project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/interp-kick1/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

csv_file = res_dir + 'csv/interp-kick1/all_results2.csv'

plots_config = {
    'plot1': {'lines': {'version': ['v0', 'v2', 'v4'],
                        'vec': ['vec'],
                        'tcm': ['tcm', 'notcm'],
                        'cc': ['g++']},
              'exclude': [['v1', 'notcm'], ['v2', 'notcm'], ['v4', 'notcm']],
              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (500k points/thread)',
              'ylabel': 'Run-time (ms)',
              'title': 'interp-kick optimizations',
              # 'ylim': [0, 16000],
              'image_name': images_dir + 'v0vsv1vs2vsv4.pdf'
              },

    'plot2': {'lines': {'version': ['v4'],
                        'vec': ['vec'],
                        'tcm': ['tcm'],
                        'cc': ['g++', 'icc']},
              'exclude': [],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (500k points/thread)',
              'ylabel': 'Run-time (ms)',
              'title': 'icc VS gcc',
              # 'ylim': [0, 16000],
              'image_name': images_dir + 'iccVSgcc.pdf'
              },

    'plot3': {'lines': {'version': ['v5', 'v6'],
                        'vec': ['vec'],
                        'tcm': ['tcm'],
                        'cc': ['g++']},
              'exclude': [],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (500k points/thread)',
              'ylabel': 'Run-time (ms)',
              'title': 'float VS double precision',
              'image_name': images_dir + 'float_vs_double.pdf'
              },

    'plot4': {'lines': {'version': ['v4'],
                        'vec': ['vec', 'novec'],
                        'tcm': ['tcm', 'notcm'],
                        'cc': ['g++']},
              'exclude': [],

              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (500k points/thread)',
              'ylabel': 'Run-time (ms)',
              'title': 'tcm and vec effects',
              'image_name': images_dir + 'tcm_and_vec_effects.pdf'
              },
    'plot5': {'lines': {'version': ['v7', 'v8', 'v9', 'v10',
                                    'v7-p100', 'v8-p100', 'v9-p100', 'v10-p100'],
                        'cc': ['nvcc']},
              'exclude': [],
              'x_name': 'points',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points',
              'ylabel': 'Run-time (ms)',
              'title': 'All GPU versions',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'all_gpu_versions.pdf'
              },
    'plot6': {'lines': {'version': ['v9', 'v4', 'v9-p100'],
                        'cc': ['nvcc', 'g++'],
                        'tcm': ['tcm', 'na'],
                        'vec': ['vec', 'na']},
              'exclude': [],
              'x_name': 'points',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Points',
              'ylabel': 'Run-time (ms)',
              'title': 'All GPU versions',
              'extra': ['plt.xscale(\'log\', basex=2)'],
              'image_name': images_dir + 'gpu_vs_gpu.pdf'
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
        # plt.xscale('log', basex=2)
        if 'ylim' in config:
            plt.ylim(config['ylim'])
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
