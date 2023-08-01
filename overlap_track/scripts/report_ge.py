#!/usr/bin/python
import os
import csv
import sys
import numpy as np
import subprocess


def string_between(string, before, after):
    temp = string.split(before)[1]
    temp = temp.split(after)[0]
    return temp


home = './'
result_dir = home + 'results/raw/roofline2/'
csv_dir = home + '/results/csv/roofline2/'
# testcases = ['kick', 'drift', 'interp-kick', 'convolution',
#              'histogram', 'fft-convolution', 'synchrotron-radiation']


def process_line(line):
    if('CPI Rate:' in line):
        metric = line.split('CPI Rate:')[1].strip()
        return ['CPI', metric]
    elif('MUX Reliability' in line):
        metric = line.split('MUX Reliability')[1].strip()
        if float(metric) < 0.7:
            print('Warning, MUX Reliability too low:', float(metric))
    elif('Front-End Bound' in line):
        metric = line.split('Front-End Bound')[1].split('%')[0].strip()
        return ['FE_Bound', metric]
    elif('Bad Speculation' in line):
        metric = line.split('Bad Speculation')[1].split('%')[0].strip()
        return ['Bad_Speculation', metric]
    elif('Memory Bound' in line):
        metric = line.split('Memory Bound')[1].split('%')[0].strip()
        return ['MEM_Bound', metric]
    elif('Core Bound' in line):
        metric = line.split('Core Bound')[1].split('%')[0].strip()
        return ['Core_Bound', metric]
    elif('Retiring' in line):
        metric = line.split('Retiring')[1].split('%')[0].strip()
        return ['Retiring', metric]


# First we have to create a dictionary
# one key per application
# one key per event
# one key for map/ combine
# a list for every run to extract mean/ std
def extract_results(tc):
    input_dir = result_dir.format(tc)
    header = ['metric', 'value']
    for dirs, subdirs, files in os.walk(input_dir):
        for file in files:
            if('summary.txt' not in file):
                continue
            out_file = csv_dir.format(tc, dirs.split('/')[-1])
            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))
            # records = []
            print(out_file)
            # for line in open(os.path.join(dirs, file), 'r'):
            records = [process_line(line)
                       for line in open(os.path.join(dirs, file), 'r')
                       if process_line(line) is not None]
            print(records)
            writer = csv.writer(open(out_file, 'w'), delimiter='\t')
            writer.writerow(header)
            # combos.sort(key=lambda a: (a[0]))
            writer.writerows(records)
    #         exit(0)
    #     outdir = os.path.abspath(outdir)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)
    #     os.chdir(input)
    #     header = ['app', 'event_count', 'std']
    #     app_dict = {}
    #     for dirs, subdirs, files in os.walk('./'):
    #         # if subdirs:
    #         #     continue
    #         for file in files:
    #             if('.report' not in file):
    #                 continue
    #             # print dirs, subdirs, files
    #             events = {}
    #             print dirs + '/' + file
    #             app = dirs.split('/')[1]
    #             if(app not in app_dict):
    #                 app_dict[app] = {}
    #             for line in open(os.path.join(dirs, file), 'r'):
    #                 process_line(line, events)
    #             for event, workers in events.items():
    #                 if(event not in app_dict[app]):
    #                     app_dict[app][event] = {}
    #                 for worker, counters in workers.items():
    #                     if(worker not in app_dict[app][event]):
    #                         app_dict[app][event][worker] = []
    #                     app_dict[app][event][worker].append([app,
    #                                                          np.mean(counters),
    #                                                          np.std(counters)])
    #             # print app_dict[app]
    #     for app, events in app_dict.items():
    #         for event, workers in events.items():
    #             for worker, combos in workers.items():
    #                 if not os.path.exists(outdir + '/' + app + '/'):
    #                     os.makedirs(outdir + '/' + app + '/')
    #                 out = open(
    #                     outdir + '/' + app + '/' + event + '_' + worker + '.csv', 'w')
    #                 writer = csv.writer(out, delimiter=' ')
    #                 writer.writerow(header)
    #                 combos.sort(key=lambda a: (a[0]))
    #                 writer.writerows(combos)


def extract_reports(tc):
    input_dir = result_dir.format(tc)
    d = {}
    for dirs, subdirs, files in os.walk(input_dir):
        if('r0' not in dirs.split('/')[-1]):
            continue
        print(dirs)
        bench = dirs.split('/')[-2]
        if bench not in d:
            d[bench] = [[], []]
        exe_list = ['amplxe-cl', '-report', 'summary',
                    '-format=csv', '-csv-delimiter=comma',
                    '-result-dir=' + dirs]
        output = subprocess.check_output(exe_list).decode()
        start = output.find('Clockticks')
        end = output.find('Collection and Platform Info')
        output = output[start:end]
        output = output.splitlines()
        data = []
        for line in output:
            if line:
                data += [line.split(',')]
        data = np.array(data, str)
        d[bench][0] = data[:, 0].tolist()
        d[bench][1] += data[:, 1].tolist()
        print(d)
        # print(data[:, 0])
        # print(np.array(data, str))
        exit(0)
        # try:
        #     app = dirs.split('/')[-2]
        # except IndexError as e:
        #     continue
        # exe_list = ['amplxe-cl', '-report', 'summary',
        #             '-format=csv' '-csv-delimiter=comma',
        #             '-result-dir=' + dirs,
        #             '-report-output=']
        # null = open('/dev/null', 'w')
        # output = subprocess.check_output(exe_list)
        # out_file = open(dirs.split('run')[0] + app + '.report', 'a')
        # out_file.write(output)


if __name__ == '__main__':
    for tc in testcases:
        # extract_reports(tc)
        extract_results(tc)
