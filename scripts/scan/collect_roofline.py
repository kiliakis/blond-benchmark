import subprocess
import os

home = '/afs/cern.ch/work/k/kiliakis/git/blond-benchmark/'
custom_template = home + 'scripts/custom.tmpl'
result_dir = home + '{}/results/raw/roofline1/{}/'
csv_dir = home + '{}/results/csv/roofline1/{}/'
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

advixe1_args = ['advixe-cl', '-collect', 'survey', '-quiet', '-project-dir']
advixe2_args = ['advixe-cl', '-collect', 'tripcounts', '-flops-and-masks',
                '-ignore-app-mismatch', '-project-dir']
advixe3_args = ['advixe-cl', '-report', 'custom', '-format=csv',
                '--report-template', custom_template, '-report-output', '',
                '-project-dir']
advixe4_args = ['advixe-cl', '-report', 'roofs', '-format=csv',
                '-report-output', '', '-project-dir']

repeats = 1


total_sims = sum(len(x) for x in testcases.values()) * repeats
print("Total runs: ", total_sims)
current_sim = 0

for app, sizes in testcases.items():
    for size in sizes:
        results = result_dir.format(app, '_'.join(size))
        csvs = csv_dir.format(app, '_'.join(size))
        if not os.path.exists(results):
            os.makedirs(results)
        if not os.path.exists(csvs):
            os.makedirs(csvs)

        stdout = open(results + 'summary.txt', 'w')
        flops = open(csvs + 'flops.csv', 'w')
        roofs = open(csvs + 'roofs.csv', 'w')
        exe = exe_form.format(app, size[0])
        exe_list = [results, '--', exe] + size[1:]
        for i in range(repeats):
            print(app, size, i)
            advixe3_args[-2] = csvs + 'flops.csv'
            advixe4_args[-2] = csvs + 'roofs.csv'
            subprocess.call(advixe1_args+exe_list, stdout=stdout, stderr=stdout)
            subprocess.call(advixe2_args+exe_list, stdout=stdout, stderr=stdout)
            subprocess.call(advixe3_args+exe_list, stdout=stdout, stderr=stdout)
            subprocess.call(advixe4_args+exe_list, stdout=stdout, stderr=stdout)
            current_sim += 1
            print("%lf %% is completed" % (100.0 * current_sim /
                                           total_sims))
        stdout.close()
        flops.close()
        roofs.close()
