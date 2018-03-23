#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from plot.plotting_utilities import *

application = 'kick'
project_dir = './'
res_dir = project_dir + 'results/'
images_dir = res_dir + 'plots/'

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

csv_file = res_dir + 'csv/kick3.csv'

plots_config = {
    'plot3': {'lines': {'version': ['v0', 'v2'],
                        'vec': ['vec', 'novec'],
                        'cc': ['icc', 'g++']},
              'normalize':  'v0-vec-g++',
              'x_name': 'points',
              'y_name': 'time(ms)',
              'y_err_name': 'std(%)',
              'xlabel': 'Macro-particles',
              'ylabel': 'Normalized time',
              'title': '',
              'width': 0.22,
              'step1': 0.22,
              'step2': 1,
              'extra': [],
              'order': ['v0-vec-g++', 'v2-vec-g++', 'v2-vec-icc', 'v0-vec-icc'],
              'names': {'v0-vec-g++': 'std-gcc',
                        'v2-vec-g++': 'vdt-gcc',
                        'v2-vec-icc': 'vdt-icc',
                        'v0-vec-icc': 'std-icc', 
                        'v0-novec-icc': 'std-icc-novec'},
              # 'ylim': [0, 16000],
              'image_name': images_dir + 'kick-benchmark.pdf'
              }

}

if __name__ == '__main__':
    data = np.genfromtxt(csv_file, delimiter='\t', dtype=str)
    header = list(data[0])
    data = data[1:]
    for plot_key, config in plots_config.items():
        print(plot_key)
        plots_dir = get_plots(header, data, config['lines'])
        normalize = plots_dir[config['normalize']]
        print(plots_dir)
        print(normalize)

        plt.figure(figsize=(5,2.))
        # plt.grid('on')
        plt.grid(True, which='major', axis='y', alpha=0.7)

        plt.title(config['title'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        # plt.xscale('log', basex=10)
        if 'ylim' in config:
            plt.ylim(config['ylim'])
        step1 = 0
        for label in config['order']:
            # if label == 'v0-vec-g++':
            #     continue
            values = plots_dir[label]

        # for label, values in plots_dir.items():
            # print(values)
            x = np.array(values[:, header.index(config['x_name'])], int)
            y = np.array(values[:, header.index(config['y_name'])], float)
            norm = np.array(normalize[:, header.index(config['y_name'])], float)
            y = y / norm
            mean = np.mean(y)
            y = np.append(y, mean)
            # y_err = np.array(
            #     values[:, header.index(config['y_err_name'])], float)
            # y_err = y_err * y / 100.
            print(label, x, y)
            # plt.errorbar(x, y, label=label, marker='o')
            plt.bar(np.arange(len(x)+1) + step1, y,
                    width=config['width'], label=config['names'][label])
            if (label == 'v0-vec-icc') or (label == 'v2-vec-icc'):
                plt.gca().annotate('%.1fx' % (1. / mean), xy=(len(x)+1.2*step1, mean + 0.03), 
                    textcoords='data', ha='center', fontsize=9)
            step1 += config['step1']
            # plt.bar(x, y, yerr=y_err, label=label, capsize=2, marker='o')
        plt.xticks(np.arange(len(x)+1), [human_format(i) for i in x] + ['Mean'])
        plt.yticks(np.arange(0, 1.01, 0.25), np.arange(0, 1.01, 0.25))
        if 'extra' in config:
            for c in config['extra']:
                exec(c)
        if plot_key == 'plot6':
            plt.gca().get_lines()
            for p in plt.gca().get_lines()[::3]:
                annotate(plt.gca(), p.get_xdata(), p.get_ydata())
        plt.legend(loc='upper left', fancybox=True, framealpha=0.4, ncol=1, fontsize=10)
        plt.tight_layout()
        plt.savefig(config['image_name'], dpi=300, bbox_inches='tight')
        import subprocess
        subprocess.call(
            ['pdfcrop', config['image_name'], config['image_name']])
        plt.show()
        plt.close()
