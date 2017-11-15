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


home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/'
result_dir = home + '{}/results/raw/run0/'
csv_dir = home + '{}/results/csv/run0/{}.csv'
testcases = ['kick', 'drift', 'interp_kick', 'convolution', 'histogram']

app = ''

delimiter = ';'


def process_line(line, events):
    global event_list
    if('Function' in line) and ('Event Count' in line):
        event_line = line.split(delimiter)[1:]
        event_list = []
        for e in event_line:
            event_list.append(e.split('Event Count:')[1].strip())
            if(e.split('Event Count:')[1].strip() not in events):
                events[e.split('Event Count:')[1].strip()] = {}
    elif('MapReduce' in line) and ('map_worker' in line):
        counters = line.split(delimiter)[1:]
        for i in range(len(counters)):
            event = event_list[i]
            if('map_worker' not in events[event]):
                events[event]['map_worker'] = []
            events[event]['map_worker'].append(int(counters[i]))
    elif('MapReduce' in line) and ('combine_worker' in line):
        counters = line.split(delimiter)[1:]
        for i in range(len(counters)):
            event = event_list[i]
            if('combine_worker' not in events[event]):
                events[event]['combine_worker'] = []
            events[event]['combine_worker'].append(int(counters[i]))
    return


# First we have to create a dictionary
# one key per application
# one key per event
# one key for map/ combine
# a list for every run to extract mean/ std
# def extract_results(input, outdir):
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
        extract_reports(tc)
        # extract_results(input_dir, output_dir)
