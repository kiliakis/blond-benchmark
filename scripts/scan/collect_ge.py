import subprocess
import os

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/'
result_dir = home + '{}/results/raw/ge1/{}/'
exe_form = home + '{}/benches/{}'
testcases = {
    'kick': [['bench1', '5000', '4000000', '2', '1']],
    'drift': [['bench1', '5000', '4000000', '1', '1']],
    'interp-kick': [['bench2', '5000', '4000000', '4000', '1']],
    'histogram': [['bench1', '5000', '4000000', '4000', '1']],
    'convolution': [['bench3', '5000', '10000', '10000', '1']],
    'fft-convolution': [['bench4', '5000', '1000000', '1000000', '1']],
    'synchrotron-radiation': [['bench2', '5000', '4000000', '1']],
    '': []
}

amplxe_args = ['amplxe-cl', '-collect',
               'general-exploration', '-no-allow-multiple-runs',
               '-discard-raw-data', '-quiet']

repeats = 1


total_sims = sum(len(x) for x in testcases.values()) * repeats
print("Total runs: ", total_sims)
current_sim = 0

for app, sizes in testcases.items():
    for size in sizes:
        results = result_dir.format(app, '_'.join(size))
        if not os.path.exists(results):
            os.makedirs(results)
        stdout = open(results + 'summary.txt', 'w')
        exe = exe_form.format(app, size[0])
        exe_list = amplxe_args + \
            ['-user-data-dir', results, '--', exe] + size[1:]
        for i in range(repeats):
            print(app, size, i)
            subprocess.call(exe_list, stdout=stdout, stderr=stdout)
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
