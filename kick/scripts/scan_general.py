import subprocess
import os
from functools import reduce
from operator import mul

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/kick/'
# home = '/home/kiliakis/git/blond-benchmark/kick/'
result_dir = home + 'results/raw/kick3/{}/'
exe_form = home + 'exe_{}_{}_{}/{}'
cuexe_form = home + 'exe_cuda/{}'
intelpy = '/cvmfs/projects.cern.ch/intelsw/python/linux/intelpython3/bin/python'
normalpy = '/afs/cern.ch/work/k/kiliakis/install/anaconda3/bin/python'

out_file_name = result_dir + 'i{}-p{}-s{}-t{}-{}-{}-{}.txt'
configs = {
    # To show performance, scalability, icc, sp vs dp
    'bench0.exe': {'sizes': [['5000', str(100000*x), '1', '1']
                            for x in [1, 2, 5, 10, 20, 50]],
                  'vec': ['vec'],
                  'tcm': ['notcm'],
                  'cc': ['g++']
                  },
    'bench2.exe': {'sizes': [['5000', str(100000*x), '1', '1']
                            for x in [1, 2, 5, 10, 20, 50]],
                  'vec': ['vec'],
                  'tcm': ['notcm'],
                  'cc': ['icc', 'g++']
                  }
    # 'bench0.exe': {'sizes': [['5000', str(100000*x), '1', '1']
    #                         for x in [1, 2, 5, 10, 20, 50]],
    #               'vec': ['novec'],
    #               'tcm': ['notcm'],
    #               'cc': ['icc']
    #               }

    # 'bench0': [['500', str(1000000*x), '1', str(x)]
    #            for x in [1, 2, 4, 8, 14, 28, 56]],
    # # To show performance, scalability, icc, sp vs dp
    # 'bench1': [['500', str(1000000*x), '1', str(x)]
    #            for x in [1, 2, 4, 8, 14, 28, 56]],
    # # To show performance, scalability, gcc, sp vs dp
    # 'bench2': [['500', str(1000000*x), '1', str(x)]
    #            for x in [1, 2, 4, 8, 14, 28, 56]],
    # # To show performance, scalability, gcc, sp vs dp
    # 'bench3': [['500', str(1000000*x), '1', str(x)]
    #            for x in [1, 2, 4, 8, 14, 28, 56]],
    # # To show that loop-tiling works
    # # 'bench4': [['500', str(1000000*x), '1', str(x)]
    # #            for x in [1, 2, 4, 8, 14]],
    # # To show performance, scalability, how tiling effects cache misses, gcc
    # 'bench5': [['500', str(1000000*x), '1', str(x)]
    #            for x in [1, 2, 4, 8, 14, 28, 56]],
    # # To show performance, scalability, how tiling effects cache misses, icc
    # 'bench6': [['500', str(1000000*x), '1', str(x)]
    #            for x in [1, 2, 4, 8, 14, 28, 56]],
    # 'bench7.cu.exe': {'sizes': [['500', str(1000000*x), '1', '512', '64']
    #                             for x in [1, 2, 4, 8, 14, 28, 56]],
    #                   'vec': ['na'],
    #                   'tcm': ['na'],
    #                   'cc': ['nvcc']
    #                   },
    # 'bench8.cu.exe': {'sizes': [['500', str(1000000*x), '1', '512', '64']
    #                             for x in [1, 2, 4, 8, 14, 28, 56]],
    #                   'vec': ['na'],
    #                   'tcm': ['na'],
    #                   'cc': ['nvcc']
    #                   },
    # 'bench9.py': {'sizes': [['500', str(1000000*x), '1', str(x)]
    #                         for x in [1]],
    #               'vec': ['na'],
    #               'tcm': ['na'],
    #               'cc': ['intel', 'normal']
    #               }
}

# proclist = ''
# for i in range(28):
#     if(i < 14):
#         proclist += str(i) + ',' + str(i+14) + ','
#     else:
#         proclist += str(i+14) + ',' + str(i+28) + ','
# proclist = proclist[:-1]

# os.environ['GOMP_CPU_AFFINITY'] = proclist
# os.environ['KMP_AFFINITY'] = "granularity=fine,proclist=["+proclist+"],explicit"
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
                if vec == 'vec':
                    vec_value = 0
                else:
                    vec_value = 1
                if tcm == 'tcm':
                    tcm_value = 1
                else:
                    tcm_value = 0
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
                        print(cc, vec, app, size, i)
                        subprocess.call(exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
