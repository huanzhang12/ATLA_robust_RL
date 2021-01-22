import os
import multiprocessing
from multiprocessing import Process, JoinableQueue
import sys
import time
import stat
import argparse
import traceback
import subprocess
from glob import glob
from os import path
from run import main, add_common_parser_opts, override_json_params
import json

q = JoinableQueue()

parser = argparse.ArgumentParser(description='Run all .json config files')
parser.add_argument('config_path', type=str, nargs='+',
                    help='json path to run all json files inside')
parser.add_argument('--out-dir-prefix', type=str, default="", required=False,
                    help='prefix for output log path')
parser.add_argument('-n', '--num-splits', type=int, default=1, required=False, help='split the total jobs into N parts')
parser.add_argument('-i', '--index', type=int, default=1, required=False, help='this is the i-th part of the N parts')
parser.add_argument('--start', type=int, default=-1, required=False, help='job start index (zero based)')
parser.add_argument('--end', type=int, default=-1, required=False, help='job end index (non-inclusive)')
parser.add_argument('--load-model', type=str, default=None, required=False, help='load pretrained model and optimizer states before training')
parser.add_argument('--ncpus', type=int, default=1, required=False, help='number of physical CPUs to assign for each task.')
parser.add_argument('--start-cpuid', type=int, default=0, required=False, help='first physical CPU ID to use')
parser.add_argument('-t', '--threads', type=int, default=-1, required=False, help='number of threads for training.')
parser.add_argument('--attack_threads', type=int, default=-1, required=False, help='number of threads used for attacks')
parser.add_argument('--run-attack', action='store_true', required=False, help='auto run attack after training finishes.')
parser.add_argument('--deterministic', type=int, default=1, help='disable Gaussian noise in action for --adv-policy-only mode')
parser.add_argument('--no-load-adv-policy', action='store_true', required=False, help='Do not load adversary policy and value network from pretrained model.')
parser.add_argument('--no-smt', action='store_true', required=False, help='assume no SMT cores in CPU.')
parser = add_common_parser_opts(parser)
args = parser.parse_args()
agent_configs = args.config_path
args_params = vars(args)
if args.no_smt:
    NUM_THREADS = multiprocessing.cpu_count()
    ATTACK_THREADS = multiprocessing.cpu_count()
else:
    NUM_THREADS = multiprocessing.cpu_count() // 2
    ATTACK_THREADS = multiprocessing.cpu_count() // 2
if args.threads > 0:
    NUM_THREADS = args.threads
del args_params['threads']
if args.attack_threads > 0:
    ATTACK_THREADS = args.attack_threads
del args_params['attack_threads']
run_attack = args_params['run_attack']
del args_params['run_attack']

attack_template = """#!/bin/bash
trap "kill 0" SIGINT  # exit cleanly when pressing control+C
export nthreads={threads}
source scan_attacks.sh
echo $pwd
semaphorename=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
export ATTACK_FOLDER_NO_WAIT=1
export ATTACK_MODEL_NO_STOCHASTIC=1
{cmdlines}
sem --wait --semaphorename $semaphorename
"""


def eprint(*args, **kwargs):
    # Print to both stdout and stderr
    if not sys.stdout.isatty():  # Don't double print
        print(*args, file=sys.stdout, **kwargs)
    print(*args, file=sys.stderr, **kwargs)


def generate_attack_script(configs):
    cmdlines = ""
    generated = {}
    for config in configs:
        key = os.path.dirname(config)
        if not key in generated:
            generated[key] = True
        else:
            continue
        print(f'Using {config} to generate attack script')
        json_params = json.load(open(config))
        out_dir = json_params['out_dir']
        if args.out_dir_prefix:
            out_dir = path.join(args.out_dir_prefix, out_dir)
        cmdline = "scan_exp_folder {config} {path} $semaphorename\n".format(config=config, path=out_dir)
        cmdlines += cmdline
    attack_cmd = attack_template.format(threads=ATTACK_THREADS, cmdlines=cmdlines)
    script_name = time.strftime("./attack_%m%d_%H%M%S.sh", time.gmtime())
    with open(script_name, "w") as f:
        f.write(attack_cmd)
    os.chmod(script_name, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    eprint(f'attack script generated at {script_name}')
    return script_name


def run_single_config(queue, index):
    # Set CPU affinity (assuming SMT)
    if args.no_smt:
        ncpus = multiprocessing.cpu_count()
    else:
        ncpus = multiprocessing.cpu_count() // 2
    cpu_list = []
    for i in range(args.ncpus):
        # main core
        cpu_list.append((args.start_cpuid + (args.ncpus * index + i)) % ncpus)
        if not args.no_smt:
            # SMT core
            cpu_list.append((args.start_cpuid + (args.ncpus * index + i)) % ncpus + ncpus)
    eprint(f'thread {index} using cpu {cpu_list}')
    os.sched_setaffinity(0, cpu_list)
    error_counts = 0
    time.sleep(2)
    while True:
        ret = 0
        conf_path = queue.get()
        json_params = json.load(open(conf_path))
        params = override_json_params(args_params, json_params, ['config_path', 'out_dir_prefix', 'num_splits', 'index', 'load_model', 'deterministic', 'no_load_adv_policy', 'start', 'end', 'ncpus', 'start_cpuid', 'no_smt'])
        # Append a prefix for output path.
        if args.out_dir_prefix:
            params['out_dir'] = path.join(args.out_dir_prefix, params['out_dir'])
            eprint(f"setting output dir to {params['out_dir']}")
        try:
            print("Running config:", params)
            ret = main(params)
        except Exception as e:
            eprint("ERROR", e)
            traceback.print_exc()
            error_counts += 1
            # raise e
        if ret == -1:
            error_counts += 1
        queue.task_done()
        eprint(f'worker {index} total errors: {error_counts}')

filelist = []
for c in agent_configs:
    filelist.extend(glob(path.join(c, "**/*.json"), recursive=True))
# De-duplicate.
filelist=sorted(set(filelist))
# Run humanoid and ant first.
sorted_filelist = list(sorted(filelist, key=lambda x: - (int('humanoid' in x or 'ant' in x)*10 + int('robust' in x) * 5 + int('lstm' in x) * 2)))

total_jobs = len(sorted_filelist)

if args.start == -1 and args.end == -1:
    jobs_per_split = total_jobs // args.num_splits
    job_start = jobs_per_split * (args.index - 1)
    if args.index == args.num_splits:
        job_end = total_jobs
    else:
        job_end = jobs_per_split * args.index
    eprint(f'Job split into {args.num_splits} part, this is part {args.index}')
else:
    # forced start and end index.
    job_start = args.start
    job_end = args.end

njobs = job_end - job_start
eprint(f'Total {total_jobs} jobs. Job start {job_start + 1}, end {job_end}')
eprint(f'We will handle {njobs} jobs.')
if njobs % NUM_THREADS != 0 and njobs // NUM_THREADS != 0:
    eprint(f'WARNING: the number of jobs {njobs} cannot be divided by number of threads {NUM_THREADS}!')
    eprint(f'You will have to wait longer than usual for the last job to finish.')

time.sleep(2)

sorted_filelist = sorted_filelist[job_start:job_end]

for fname in sorted_filelist:
    eprint("python run.py --config-path {}".format(fname))

if run_attack:
    attack_script = generate_attack_script(sorted_filelist)

time.sleep(5)

for i in range(min(njobs, NUM_THREADS)):
    worker = Process(target=run_single_config, args=(q, i))
    worker.daemon = True
    worker.start()

for fname in sorted_filelist:
    q.put(fname)

q.join()

if run_attack:
    subprocess.check_call(attack_script)

