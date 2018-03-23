import subprocess
import os
from functools import reduce
from operator import mul

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/drift/'
# home = '/home/kiliakis/git/blond-benchmark/drift/'

result_dir = home + 'results/raw/drift4/{}/'
exe_form = home + 'exe_{}_{}_{}/{}'
cuexe_form = home + 'exe_cuda/{}'
intelpy = '/cvmfs/projects.cern.ch/intelsw/python/linux/intelpython3/bin/python'
normalpy = '/afs/cern.ch/work/k/kiliakis/install/anaconda3/bin/python'

out_file_name = result_dir + 'i{}-p{}-a{}-t{}-{}-{}-{}.txt'

configs = {
    'bench0.exe': {'sizes': [['1000', str(x), '0', '1']
                             for x in [100000, 200000, 500000, 1000000, 2000000]],
                   'vec': ['vec'],
                   'tcm': ['notcm'],
                   'cc': ['g++']},
    # 'bench1': {'sizes': [['1000', str(500000 * x), '0', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['g++']},
    # 'bench2': {'sizes': [['1000', str(500000 * x), '0', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm'],
    #           'cc': ['g++']},

    # 'bench3': {'sizes': [['1000', str(500000 * x), '0', str(x)]
    #                      for x in [1, 2, 4, 8, 14, 28, 56]]+
    #                      [['1000', str(500000 * x), '1', str(x)]
    #                      for x in [1, 2, 4, 8, 14, 28, 56]],
    #            'vec': ['vec'],
    #            'tcm': ['notcm'],
    #            'cc': ['g++']}
    # 'bench4.cu.exe': {'sizes': [['1000', str(500000 * x), '0', '512', '64']
    #                             for x in [1, 2, 4, 8, 14, 28, 56]] +
    #                   [['1000', str(500000 * x), '1', '512', '64']
    #                       for x in [1, 2, 4, 8, 14, 28, 56]],
    #                   'vec': ['na'],
    #                   'tcm': ['na'],
    #                   'cc': ['nvcc']},
    # 'bench5.py': {'sizes': [['1000', str(500000 * x), '0', str(x)]
    #                         for x in [1]] +
    #               [['1000', str(500000 * x), '1', str(x)]
    #                for x in [1]],
    #               'vec': ['na'],
    #               'tcm': ['na'],
    #               'cc': ['intel', 'normal']}
}


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
                if 'cu' in app:
                    make_string = 'make cuda CUDEBUG='
                else:
                    make_string = 'make -k CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(
                        cc, tcm_value, vec_value, cc, vec, tcm)
                if '.py' not in app:
                    subprocess.call(make_string, shell=True)
                else:
                    if cc == 'intel':
                        py = intelpy
                    else:
                        py = normalpy
                for size in configs[app]['sizes']:
                    results = result_dir.format(app)
                    if not os.path.exists(results):
                        os.makedirs(results)

                    if 'cu' in app:
                        stdout = open(out_file_name.format(
                            app, size[0], size[1], size[2],
                            '{}X{}'.format(size[3], size[4]),
                            cc, vec, tcm), 'w')
                        exe = cuexe_form.format(app)
                    else:
                        stdout = open(out_file_name.format(
                            app, size[0], size[1], size[2], size[3],
                            cc, vec, tcm), 'w')
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
