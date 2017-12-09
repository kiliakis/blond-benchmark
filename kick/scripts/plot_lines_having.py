#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import sys

from plot.plotting_utilities import *

application = 'kick'
project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'results/plots/kick1/'

csv_file = res_dir + 'csv/kick1/all_results.csv'

plots_config = {
    'plot3': {'lines': {'version': ['v0', 'v2'],
                        'vec': ['vec'],
                        'cc': ['icc', 'g++']},
              # 'labels': ['version', 'cc'],
              'x_name': 'threads',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Threads (500k points/thread)',
              'ylabel': 'Run-time (ms)',
              'title': 'std::sin VS vdt::sin',
              'image_name': 'plot3.pdf'
              }
}

if __name__ == '__main__':
    data = np.genfromtxt(csv_file, delimiter='\t', dtype=str)
    header = list(data[0])
    data = data[1:]
    for plot_key, config in plots_config.items():
        plots_dir = get_plots(header, data, config['lines'])
        # print(plots_dir)
        plt.figure()
        plt.grid('on')
        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.xscale('log', basex=2)
        for label, values in plots_dir.items():
            print(values)
            x = np.array(values[:, header.index(config['x_name'])], float)
            y = np.array(values[:, header.index(config['y_name'])], float)
            y_err = np.array(
                values[:, header.index(config['y_err_name'])], float)
            y_err = y_err * y / 100.
            print(label, x, y)
            plt.errorbar(x, y, yerr=y_err, label=label)
        plt.legend(loc='best', fancybox=True)
        plt.tight_layout()
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

    # for image_name in image_names:
    #     plt.savefig(image_name, bbox_inches='tight')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
