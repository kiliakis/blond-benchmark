import subprocess
import os

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/interp-kick/'
result_dir = home + 'results/raw/interp-kick1/{}/'
# exe_form = home + 'benches/{}'
vec_list = ['vec']
tcm_list = ['tcm']
cc_list = ['g++']
exe_form = home + 'exe_{}_{}_{}/{}'

out_file_name = result_dir + 'i{}-p{}-s{}-t{}-{}-{}-{}.txt'
testcases = {
    # To show performance, scalability, icc, sp vs dp
    # 'bench0': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],

    # # To show performance, scalability, icc, sp vs dp
    # 'bench1': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    # # To show performance, scalability, gcc, sp vs dp
    # 'bench2': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    # # To show performance, scalability, gcc, sp vs dp
    # 'bench3': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    # To show that loop-tiling works
    'bench4': [['1000', str(500000*x), str(100*x), str(x)]
               for x in [1, 2, 4, 8, 14]],
    # To show performance, scalability, how tiling effects cache misses, gcc
    'bench5': [['1000', str(500000*x), str(100*x), str(x)]
               for x in [1, 2, 4, 8, 14]],
    # To show performance, scalability, how tiling effects cache misses, icc
    # 'bench6': [['1000', str(500000*x), str(100*x), str(x)]
    #            for x in [1, 2, 4, 8, 14]],
    '': []
}

# proclist = 'proclist=['
proclist = ''
for i in range(28):
    if(i < 14):
        proclist += str(i) + ',' + str(i+14) + ','
    else:
        proclist += str(i+14) + ',' + str(i+28) + ','
proclist = proclist[:-1]

os.environ['GOMP_CPU_AFFINITY'] = proclist
os.environ['KMP_AFFINITY'] = "granularity=fine,proclist=["+proclist+"],explicit"
# print(os.environ['KMP_AFFINITY'])

repeats = 3


total_sims = sum(len(x) for x in testcases.values()) * repeats * 8
print("Total runs: ", total_sims)
current_sim = 0
os.chdir(home)
for cc in cc_list:
    for tcm in tcm_list:
        for vec in vec_list:
            if (cc=='icc' and vec=='novec'):
                continue
            subprocess.call('make clean', shell=True)
            if tcm=='tcm':
                tcm_value=1
            else:
                tcm_value=0
            if vec=='vec':
                vec_value=0
            else:
                vec_value=1
            make_string = 'make CC={} TCM={} NOVEC={} PROGS_DIR=exe_{}_{}_{}'.format(cc,
                            tcm_value, vec_value, cc, vec, tcm)
            subprocess.call(make_string, shell=True)    
            for app, sizes in testcases.items():
                for size in sizes:
                    results = result_dir.format(app)
                    if not os.path.exists(results):
                        os.makedirs(results)

                    stdout = open(out_file_name.format(
                        app, size[0], size[1], size[2], size[3], 
                        cc, vec, tcm), 'w')
                    exe = exe_form.format(cc, vec, tcm, app)
                    exe_list = [exe] + size
                    for i in range(repeats):
                        print(cc, tcm, vec, app, size, i)
                        subprocess.call(exe_list, stdout=stdout,
                                        stderr=stdout, env=os.environ.copy())
                        current_sim += 1
                        print("%lf %% is completed" % (100.0 * current_sim /
                                                       total_sims))
