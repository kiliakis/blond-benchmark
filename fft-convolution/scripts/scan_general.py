import subprocess
import os
from functools import reduce
from operator import mul

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/fft-convolution/'
result_dir = home + 'results/raw/fft-convolution1/{}/'
# exe_form = home + 'benches/{}'
exe_form = home + 'exe_{}_{}_{}/{}'
cuexe_form = home + 'exe_cuda/{}'

out_file_name = result_dir + 'i{}-s{}-k{}-t{}-{}-{}-{}.txt'

configs = {
    # 'bench1': {'sizes': [['500', str(50000 * x), '50000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28, 56]],
    #           'vec': ['na'],
    #           'tcm': ['na'],
    #           'cc': ['na']},

    # Parallelization not working, need to fix this
    # 'bench2': {'sizes': [['500', str(50000 * x), '50000', str(x)]
    #                      for x in [56]],
    #            'vec': ['novec'],
    #            'tcm': ['tcm', 'notcm'],
    #            'cc': ['g++']}


    # 'bench4': {'sizes': [['500', str(50000 * x), '50000', str(x)]
    #                     for x in [56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm', 'notcm'],
    #           'cc': ['icc']},

    # 'bench5': {'sizes': [['500', str(50000 * x), '50000', str(x)]
    #                     for x in [1, 2, 4, 8, 14, 28]],
    #           'vec': ['vec', 'novec'],
    #           'tcm': ['notcm', 'notcm'],
    #           'cc': ['icc']},

    # 'bench6': {'sizes': [['500', str(50000 * x), '50000', str(x)]
    #                     for x in [56]],
    #           'vec': ['vec'],
    #           'tcm': ['notcm', 'notcm'],
    #           'cc': ['icc']}
    'bench7.cu.exe': {'sizes': [['500', str(50000 * x), str(50000 * x), str(x)]
                                for x in [1, 2, 4, 8, 14, 28, 56]],
                      'vec': ['na'],
                      'tcm': ['na'],
                      'cc': ['nvcc']},
    'bench8.cu.exe': {'sizes': [['500', str(50000 * x), str(50000 * x), str(x)]
                                for x in [1, 2, 4, 8, 14, 28, 56]],
                      'vec': ['na'],
                      'tcm': ['na'],
                      'cc': ['nvcc']},
    'bench9.cu.exe': {'sizes': [['500', str(50000 * x), str(50000 * x), str(x)]
                                for x in [1, 2, 4, 8, 14, 28, 56]],
                      'vec': ['na'],
                      'tcm': ['na'],
                      'cc': ['nvcc']}
}


# proclist = 'proclist=['
proclist = ''
for i in range(28):
    if(i < 14):
        proclist += str(i) + ',' + str(i + 14) + ','
    else:
        proclist += str(i + 14) + ',' + str(i + 28) + ','
proclist = proclist[:-1]

# os.environ['GOMP_CPU_AFFINITY'] = proclist
# os.environ['KMP_AFFINITY'] = "granularity=fine,proclist=[" + \
#     proclist + "],explicit"
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
                if app == 'bench2' and cc == 'icc' and vec == 'vec' and tcm == 'notcm':
                    continue
                # subprocess.call('make clean', shell=True)
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
                    make_string = 'make {} -k CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(
                        cc, tcm_value, vec_value, cc, vec, tcm)
                # if app != 'bench1':
                    # subprocess.call(make_string, shell=True)
                for size in configs[app]['sizes']:
                    results = result_dir.format(app)
                    if not os.path.exists(results):
                        os.makedirs(results)
                    if 'cu' in app:
                        stdout = open(out_file_name.format(
                            app, size[0], size[1], size[2], size[3],
                            cc, vec, tcm), 'w')
                        exe = cuexe_form.format(app)
                    else:
                        stdout = open(out_file_name.format(
                            app, size[0], size[1], size[2], size[3],
                            cc, vec, tcm), 'w')
                        exe = exe_form.format(cc, vec, tcm, app)
                    if app == 'bench1':
                        exe_list = ['python', 'benches/bench1.py'] + size
                    else:
                        exe_list = [exe] + size
                    for i in range(repeats):
                        print(app, cc, tcm, vec, size, i)
                        subprocess.call(exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
