import subprocess
import os
from functools import reduce
from operator import mul

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/synchrotron-radiation/'
result_dir = home + 'results/raw/synch-rad1/{}/'
# exe_form = home + 'benches/{}'
vec_list = ['vec', 'novec']
tcm_list = ['tcm', 'notcm']
cc_list = ['g++', 'icc']
exe_form = home + 'exe_{}_{}_{}/{}'

out_file_name = result_dir + 'i{}-p{}-t{}-{}-{}-{}.txt'

configs = {
    'bench0': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec'],
               'tcm': ['notcm'],
               'cc': ['icc', 'g++']},

    'bench1': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec', 'novec'],
               'tcm': ['notcm'],
               'cc': ['icc', 'g++']},

    'bench2': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec', 'novec'],
               'tcm': ['notcm'],
               'cc': ['icc', 'g++']},

    'bench3': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec', 'novec'],
               'tcm': ['tcm', 'notcm'],
               'cc': ['icc']},

    'bench4': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec'],
               'tcm': ['tcm'],
               'cc': ['icc']},

    'bench5': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec'],
               'tcm': ['notcm'],
               'cc': ['icc', 'g++']},

    'bench6': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec'],
               'tcm': ['notcm'],
               'cc': ['icc', 'g++']},

    'bench7': {'sizes': [['500', str(500000 * x), str(x)]
                         for x in [1, 2, 4, 8, 14, 28, 56]],
               'vec': ['vec', 'novec'],
               'tcm': ['tcm', 'notcm'],
               'cc': ['icc']}
}


# proclist = 'proclist=['
proclist = ''
for i in range(28):
    if(i < 14):
        proclist += str(i) + ',' + str(i + 14) + ','
    else:
        proclist += str(i + 14) + ',' + str(i + 28) + ','
proclist = proclist[:-1]

os.environ['GOMP_CPU_AFFINITY'] = proclist
os.environ['KMP_AFFINITY'] = "granularity=fine,proclist=[" + \
    proclist + "],explicit"
# print(os.environ['KMP_AFFINITY'])

repeats = 3

total_sims = repeats * \
    sum([reduce(mul, [len(x) for x in y.values()])
         for y in configs.values()])


print("Total runs: ", total_sims)
current_sim = 0
os.chdir(home)

for app, config in configs.items():
    for cc in configs[app]['cc']:
        for tcm in configs[app]['tcm']:
            for vec in configs[app]['vec']:
                subprocess.call('make clean', shell=True)
                if tcm == 'tcm':
                    tcm_value = 1
                else:
                    tcm_value = 0
                if vec == 'vec':
                    vec_value = 0
                else:
                    vec_value = 1
                make_string = 'make -k CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(
                    cc, tcm_value, vec_value, cc, vec, tcm)
                subprocess.call(make_string, shell=True)
                for size in configs[app]['sizes']:
                    results = result_dir.format(app)
                    if not os.path.exists(results):
                        os.makedirs(results)

                    stdout = open(out_file_name.format(
                        app, size[0], size[1], size[2], cc, vec, tcm), 'w')
                    exe = exe_form.format(cc, vec, tcm, app)
                    exe_list = [exe] + size
                    for i in range(repeats):
                        print(app, cc, tcm, vec, size, i)
                        subprocess.call(exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
