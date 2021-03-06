import subprocess
import os
from functools import reduce
from operator import mul

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/convolution/'
result_dir = home + 'results/raw/convolution2/{}/'
# exe_form = home + 'benches/{}'
exe_form = home + 'exe_{}_{}_{}/{}'

out_file_name = result_dir + 'i{}-s{}-k{}-t{}-{}-{}-{}.txt'

intelpy = '/cvmfs/projects.cern.ch/intelsw/python/linux/intelpython3/bin/python'
normalpy = '/afs/cern.ch/work/k/kiliakis/install/anaconda3/bin/python'

configs = {
    # 'bench0.exe': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [4]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['icc']},
    'bench11.py': {'sizes': [['500', str(4000 * x), '4000', str(x)]
                             for x in [1]],
                   'vec': ['na'],
                   'tcm': ['na'],
                   'cc': ['intel', 'normal']},

    # 'bench1': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['icc', 'g++']},

    # 'bench2': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['icc']},

    # 'bench3': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec', 'novec'],
    #           'tcm': ['notcm', 'tcm'],
    #           'cc': ['icc']},

    # 'bench5': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec', 'novec'],
    #           'tcm': ['notcm', 'tcm'],
    #           'cc': ['icc']},

    # 'bench7': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec', 'novec'],
    #           'tcm': ['notcm'],
    #           'cc': ['g++', 'icc']},

    # 'bench8': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec', 'novec'],
    #           'tcm': ['notcm', 'tcm'],
    #           'cc': ['icc']},
    # 'bench9': {'sizes': [['500', str(4000 * x), '4000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm', 'tcm'],
    #           'cc': ['icc']}
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

repeats = 5

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
                # subprocess.call('make clean', shell=True)
                if tcm == 'tcm':
                    tcm_value = 1
                else:
                    tcm_value = 0
                if vec == 'vec':
                    vec_value = 0
                else:
                    vec_value = 1
                if '.py' not in app:
                    make_string = 'make -k CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(
                        cc, tcm_value, vec_value, cc, vec, tcm)
                else:
                    if cc == 'intel':
                        py = intelpy
                    else:
                        py = normalpy

                # subprocess.call(make_string, shell=True)
                for size in configs[app]['sizes']:
                    results = result_dir.format(app)
                    if not os.path.exists(results):
                        os.makedirs(results)

                    stdout = open(out_file_name.format(
                        app, size[0], size[1], size[2], size[3], cc, vec, tcm), 'w')
                    exe = exe_form.format(cc, vec, tcm, app)
                    if '.py' in app:
                        exe_list = [py, 'benches/'+app] + size
                    else:
                        exe_list = [exe] + size
                    for i in range(repeats):
                        print(app, cc, tcm, vec, size, i)
                        subprocess.call(exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
