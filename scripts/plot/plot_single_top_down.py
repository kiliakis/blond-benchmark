#!/usr/bin/python
import matplotlib.pyplot as plt
import subprocess
import os
import numpy as np
import sys
import csv
from matplotlib import colors as colors
from cycler import cycler
from plot.plotting_utilities import *
import argparse
import matplotlib.gridspec as gridspec

colors = ['#b2182b', '#ef8a62', '#fddbc7', '#2166ac', '#67a9cf', '#d1e5f0']

parser = argparse.ArgumentParser(
    description='Plot the calculated metrics into a bar graph.',
    usage='plot_bars.py infile -o outfile')

parser.add_argument('infile', action='store', type=str,
                    help='The input file that contains the metrics.')

parser.add_argument('-o', '--outfile', action='store', type=str,
                    default='bars.pdf', help='The output image file name.')


args = parser.parse_args()

project_dir = './'
res_dir = project_dir + 'results/'
# images_dir = res_dir + 'plots/'


plots_config = {
    'plot1': {'bars': ['front_bound%',
                       'bad_speculation%',
                       'retiring%',
                       # 'be_bound%'
                       'core_bound%',
                       'mem_bound%'
                       ],
              'pairs': {'front_bound%': 'FEB%',
                        'bad_speculation%': 'BS%',
                        'retiring%': 'RET%',
                        'be_bound%': 'BEB%',
                        'core_bound%': 'CB%',
                        'mem_bound%': 'MB%'
                        },
              'names': {'drift': 'drift',
                        'kick': 'kick',
                        'interp_kick': 'LIkick',
                        'slices': 'hist',
                        'sync_rad': 'SR',
                        'other': 'other',
                        'fft': 'fft',
                        'statistics': 'stats'},
              # 'order': ['interp_kick', 'slices', 'drift', 'fft', 'other'],
              'xticks': ['Time', 'FEB %',
                         'BS %', 'RET %',
                         'CB %', 'MB %'],
              'colors': {'drift': 'tab:blue',
                         'kick': 'tab:green',
                         'interp_kick': 'tab:red',
                         'slices': 'tab:purple',
                         'sync_rad': 'tab:olive',
                         'other': 'tab:grey',
                         'fft': 'tab:orange',
                         'statistics': 'tab:brown'},
              # 'bench_seq' : [''],
              'xlabel': '',
              'ylabel': 'Metric%',
              'title': '',
              'extra': [],
              'image_name': args.outfile
              }
}


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


if __name__ == '__main__':

    data = np.genfromtxt(args.infile, dtype=str, delimiter='\t')

    header = data[0].tolist()
    data = data[1:]
    metrics = data[:, 0].tolist()
    data = np.array(data[:, 1:], float)
    header = header[1:]

    # data[:, [header.index('other'), -1]] = data[:, [-1, header.index('other')]]
    # header[header.index(
    #     'other')], header[-1] = header[-1], header[header.index('other')]
    print('header', header)
    print('metrics', metrics)
    print('data', data)

    for plot, config in plots_config.items():

        f = plt.figure(figsize=(5., 0.8))
        # gs = gridspec.GridSpec(1, 7)
        # ax1 = plt.subplot(gs[0, 0])
        # ax2 = plt.subplot(gs[0, 1:])

        # plt.axes(ax1)
        plt.yticks([], [])
        plt.xticks(np.arange(0, 101, 20), np.arange(0, 101, 20), fontsize=9)
        plt.xlim((0, 100))
        plt.ylabel('Pipeline\nSlots %', rotation=90, fontsize=10)
        # plt.ylabel('Contribution %', fontsize=10, labelpad=-5)

        # plt.axes(ax2)

        # plt.xticks(np.arange(1, len(config['xticks'])+1, 1),
        #            config['xticks'], rotation=0, color='black', fontsize=11)
        # # plt.gca().xaxis.get_major_ticks()[0].label1.set_color('b')
        # plt.yticks(np.arange(0, 76, 10), np.arange(0, 76, 10))

        # plt.xlabel(config['xlabel'], labelpad=0)
        # plt.title(config['title'])
        # plt.grid(True, which='major', axis='y', alpha=0.7)

        x = 1
        width = 0.35
        bottom = 0
        for m, r in zip(metrics, data):
            if m not in config['bars']:
                continue
            plt.barh(x, r, width, left=bottom, label=config['pairs'][m],
                     color=colors.pop())
            # color=config['colors'][h])
            bottom += r
        # plt.show()

        # sys.exit()

        # color_y_axis(ax1, 'blue')
        # color_y_axis(ax2, 'green')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 2.5),
                   ncol=5,
                   fancybox=True, framealpha=0.5,
                   fontsize=10, handletextpad=0.15,
                   columnspacing=1.)
        # plt.legend(title='', loc='upper left',
        #            ncol=5,
        #            # bbox_to_anchor=(0, 1),
        #            fontsize=11, framealpha=0.4)
        plt.tight_layout()
        plt.savefig(config['image_name'], bbox_inches='tight', dpi=300)
        subprocess.call(['pdfcrop', config['image_name'], config['image_name']])
        plt.show()
        plt.close()
