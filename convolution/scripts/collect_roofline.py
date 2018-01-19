import subprocess
import os
from functools import reduce
from operator import mul


home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/convolution/'
custom_template = home + '../scripts/custom.tmpl'
result_dir = home + 'results/raw/roofline2/{}/'
csv_dir = home + 'results/csv/roofline2/{}/'
# exe_form = home + '{}/benches/{}'
exe_form = home + 'exe_{}_{}_{}/{}'
result_dir = result_dir + 'i{}-s{}-k{}-t{}-{}-{}-{}/'
csv_dir = csv_dir + 'i{}-s{}-k{}-t{}-{}-{}-{}/'
# out_file_name = result_dir + 'i{}-s{}-k{}-t{}-{}-{}-{}.txt'

configs = {
    # 'bench0.exe': {'sizes': [['5000', '10000', '10000', str(x)]
    #                     for x in [1]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['icc']},
    # 'bench1.exe': {'sizes': [['5000', '10000', '10000', str(x)]
    #                     for x in [1]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['icc']},
    'bench3.exe': {'sizes': [['5000', '10000', '10000', str(x)]
                        for x in [1]],
              'vec': ['vec'],
              'tcm': ['notcm'],
              'cc': ['icc']},
    'bench5.exe': {'sizes': [['5000', '10000', '10000', str(x)]
                        for x in [1]],
              'vec': ['vec'],
              'tcm': ['notcm'],
              'cc': ['icc']},



}

advixe1_args = ['advixe-cl', '-collect', 'survey', '-quiet', '-project-dir']
advixe2_args = ['advixe-cl', '-collect', 'tripcounts', '-flops-and-masks',
                '-ignore-app-mismatch', '-project-dir']
advixe3_args = ['advixe-cl', '-report', 'custom', '-format=csv',
                '--report-template', custom_template, '-report-output', '',
                '-project-dir']
advixe4_args = ['advixe-cl', '-report', 'roofs', '-format=csv',
                '-report-output', '', '-project-dir']

# proclist = ''
# for i in range(28):
#     if(i < 14):
#         proclist += str(i) + ',' + str(i + 14) + ','
#     else:
#         proclist += str(i + 14) + ',' + str(i + 28) + ','
# proclist = proclist[:-1]

# os.environ['GOMP_CPU_AFFINITY'] = proclist
# os.environ['KMP_AFFINITY'] = "granularity=fine,proclist=[" + \
#     proclist + "],explicit"


repeats = 1

total_sims = repeats * \
    sum([reduce(mul, [len(x) for x in y.values()])
         for y in configs.values()])

print("Total runs: ", total_sims)
current_sim = 0

for app, config in configs.items():
    for cc in configs[app]['cc']:
        for tcm in configs[app]['tcm']:
            for vec in configs[app]['vec']:
                subprocess.call('make clean', shell=True)
                vec_value = 1
                tcm_value = 0
                if vec == 'vec':
                    vec_value = 0
                if tcm == 'tcm':
                    tcm_value = 1
                make_string = 'make -k CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(
                    cc, tcm_value, vec_value, cc, vec, tcm)
                subprocess.call(make_string, shell=True)
                for size in configs[app]['sizes']:

                    results = result_dir.format(app, size[0], size[1],
                                                size[2], size[3], cc, vec, tcm)
                    csvs = csv_dir.format(app, size[0], size[1],
                                          size[2], size[3], cc, vec, tcm)
                    if not os.path.exists(results):
                        os.makedirs(results)
                    if not os.path.exists(csvs):
                        os.makedirs(csvs)

                    stdout = open(results + 'summary.txt', 'w')
                    flops = open(csvs + 'flops.csv', 'w')
                    roofs = open(csvs + 'roofs.csv', 'w')
                    # stdout = open(out_file_name.format(
                    #     app, size[0], size[1], size[2], size[3],
                    #     cc, vec, tcm), 'w')
                    exe = exe_form.format(cc, vec, tcm, app)
                    exe_list = [results, '--', exe] + size
                    for i in range(repeats):
                        print(cc, vec, app, size, i)
                        advixe3_args[-2] = csvs + 'flops.csv'
                        advixe4_args[-2] = csvs + 'roofs.csv'
                        subprocess.call(advixe1_args+exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        subprocess.call(advixe2_args+exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        subprocess.call(advixe3_args+exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        subprocess.call(advixe4_args+exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
                    stdout.close()
                    flops.close()
                    roofs.close()
