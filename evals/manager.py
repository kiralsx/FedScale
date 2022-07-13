# Submit job to the remote cluster

import yaml
import sys
import time
import random
import os, subprocess
import pickle, datetime

def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def process_cmd(yaml_file,local = False):
    yaml_conf = load_yaml_conf(yaml_file)

    ps_ip = yaml_conf['ps_ip']
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    executor_configs = "=".join(yaml_conf['worker_ips'])
    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))
    
    print(worker_ips)
    print(total_gpus)

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'recent'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    # job_conf = {'time_stamp':time_stamp,
    #             'ps_ip':ps_ip,
    #             }

    # for conf in yaml_conf['job_conf']:
    #     job_conf.update(conf)

    # conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "


    # for conf_name in job_conf:
    #     conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
    #     if conf_name == "job_name":
    #         job_name = job_conf[conf_name]
    #     if conf_name == "log_path":
    #         log_path = os.path.join(job_conf[conf_name], 'log', job_name, time_stamp)

    total_gpu_processes =  sum([sum(x) for x in total_gpus])

    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    # ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} "
    # ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']}\
    #           --yaml_file={yaml_file} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} ps_ip={ps_ip} --log_path=$FEDSCALE_HOME/evals/multi_logs --job_name=femnist1 --time_stamp={time_stamp} "
    ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} \
              --yaml_file={yaml_file} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} --ps_ip={ps_ip} --time_stamp={time_stamp} "
    
    print(ps_cmd)

    with open(f"{job_name}_logging", 'wb') as fout:
        pass


    print(f"Starting aggregator on {ps_ip}...")
    with open(f"{job_name}_logging", 'a') as fout:
        if local:
            print("run local")
            subprocess.Popen(f'{ps_cmd}',shell=True, stdout=fout, stderr=fout)
        else:
            subprocess.Popen(f'ssh {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
                            shell=True, stdout=fout, stderr=fout)

    time.sleep(10)
    print("after sleep")
    # =========== Submit job to each worker ============
    rank_id = 1
    
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")

        for cuda_id in range(len(gpu)):
            for _  in range(gpu[cuda_id]):
                # worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} --cuda_device=cuda:{cuda_id} "
                worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} \
                              --yaml_file={yaml_file} --this_rank={rank_id} --num_executors={total_gpu_processes} --cuda_device=cuda:{cuda_id} --time_stamp={time_stamp} "
                rank_id += 1
                
                with open(f"{job_name}_logging", 'a') as fout:
                    time.sleep(2)
                    if local:
                        subprocess.Popen(f'{worker_cmd}',
                                         shell=True, stdout=fout, stderr=fout)
                    else:
                        subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                                        shell=True, stdout=fout, stderr=fout)
                

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, job_name)
    with open(job_name, 'wb') as fout:
        job_meta = {'user':submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    # print(f"Submitted job, please check your logs $HOME/{job_conf['model']}/{time_stamp} for status")

def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
        print(f"Shutting down job on {vm_ip}")
        with open(f"{job_name}_logging", 'a') as fout:
            subprocess.Popen(f'ssh {job_meta["user"]}{vm_ip} "python {current_path}/shutdown.py {job_name}"',
                            shell=True, stdout=fout, stderr=fout)

        # _ = os.system(f"ssh {job_meta['user']}{vm_ip} 'python {current_path}/shutdown.py {job_name}'")

if sys.argv[1] == 'submit' or sys.argv[1] == 'start':
    os.system("python shutdown.py all")
    process_cmd(sys.argv[2], False if sys.argv[1] =='submit' else True)
elif sys.argv[1] == 'stop':
    terminate(sys.argv[2])
else:
    print("Unknown cmds ...")


