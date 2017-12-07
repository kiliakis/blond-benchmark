import subprocess
import os

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/histogram/'
result_dir = home + 'results/raw/histo1/{}/'
exe_form = home + 'benches/{}'
out_file_name = result_dir + 'i{}-p{}-s{}-t{}.txt'
testcases = {
    # To use as basis
    'bench0': [['1000', str(500000*x), str(100*x), str(x)]
               for x in [1, 2, 4, 8, 14]],
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
    '': []
}

proclist = 'proclist=['
for i in range(28):
    if(i < 14):
        proclist += str(i) + ',' + str(i+14) + ','
    else:
        proclist += str(i+14) + ',' + str(i+28) + ','
proclist = proclist[:-1] + ']'

os.environ['KMP_AFFINITY'] = "granularity=fine,"+proclist+",explicit"
print(os.environ['KMP_AFFINITY'])

# amplxe_args = ['amplxe-cl', '-collect',
#                'general-exploration', '-no-allow-multiple-runs',
#                '-discard-raw-data', '-quiet']

repeats = 5


total_sims = sum(len(x) for x in testcases.values()) * repeats
print("Total runs: ", total_sims)
current_sim = 0

for app, sizes in testcases.items():
    for size in sizes:
        results = result_dir.format(app)
        if not os.path.exists(results):
            os.makedirs(results)

        stdout = open(out_file_name.format(
            app, size[0], size[1], size[2], size[3]), 'w')
        exe = exe_form.format(app)
        exe_list = [exe] + size
        for i in range(repeats):
            print(app, size, i)
            subprocess.call(exe_list, stdout=stdout,
                            stderr=stdout, env=os.environ.copy())
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
