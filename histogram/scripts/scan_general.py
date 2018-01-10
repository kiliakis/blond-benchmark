import subprocess
import os
from functools import reduce
from operator import mul


home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/histogram/'
home = '/home/kiliakis/git/blond-benchmark/histogram/'
result_dir = home + 'results/raw/histo2/{}/'
exe_form = home + 'benches/{}'
cuexe_form = home + 'exe_cuda/{}'

out_file_name = result_dir + 'i{}-p{}-s{}-t{}.txt'
configs = {
    # To use as basis
    # 'bench0': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    # To show that it doesn't scale
    # 'bench1': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    # # To show that is scales better, plus that the serial reduction is a problem
    # 'bench2': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]] +
    #           [['1000', '5000000', str(500*x), '14']
    #            for x in [1, 10, 100, 1000]],

    # # To show that the parallel reductin works
    # 'bench3': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]] +
    #           [['1000', '5000000', str(500*x), '14']
    #            for x in [1, 10, 100, 1000]],
    # # To show that loop-tiling works
    # 'bench4': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    # # To show that parallel allocation with TCMalloc works
    # 'bench5_tcm': [['1000', str(500000*x), str(100*x), str(x)]
    #                for x in [1, 2, 4, 8, 14]],
    # # To show that heavier optimization works
    # 'bench6_tcm': [['1000', str(500000*x), str(100*x), str(x)]
    #                for x in [1, 2, 4, 8, 14]],

    'bench7.cu.exe': {'sizes': [['1000', str(500000*x), str(100*x), '512', '1024']
                                for x in [1, 2, 4, 8, 14]],
                      'vec': ['na'],
                      'tcm': ['na'],
                      'cc': ['nvcc']
                      },
    'bench8.cu.exe': {'sizes': [['1000', str(500000*x), str(100*x), '512', '1024']
                                for x in [1, 2, 4, 8, 14]],
                      'vec': ['na'],
                      'tcm': ['na'],
                      'cc': ['nvcc']
                      },
    'bench9.cu.exe': {'sizes': [['1000', str(500000*x), str(100*x), '512', '1024']
                                for x in [1, 2, 4, 8, 14]],
                      'vec': ['na'],
                      'tcm': ['na'],
                      'cc': ['nvcc']
                      }


}

# proclist = 'proclist=['
# for i in range(28):
#     if(i < 14):
#         proclist += str(i) + ',' + str(i+14) + ','
#     else:
#         proclist += str(i+14) + ',' + str(i+28) + ','
# proclist = proclist[:-1] + ']'

# os.environ['KMP_AFFINITY'] = "granularity=fine,"+proclist+",explicit"
# print(os.environ['KMP_AFFINITY'])


repeats = 5

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
                if vec == 'vec':
                    vec_value = 0
                else:
                    vec_value = 1
                if 'cu' in app:
                    make_string = 'make cuda CUDEBUG='
                else:
                    make_string = 'make {} -k CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(
                        cc, tcm_value, vec_value, cc, vec, tcm)
                subprocess.call(make_string, shell=True)
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

                    exe_list = [exe] + size
                    for i in range(repeats):
                        print(cc, vec, app, size, i)
                        subprocess.call(exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
