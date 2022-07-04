import argparse
from fedscale.core import events
import yaml
import shlex
import os


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

parser = argparse.ArgumentParser()

# parse input args from manager.py
parser.add_argument('--yaml_file', type=str)
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--num_executors', type=int, default=1)
parser.add_argument('--executor_configs', type=str, default="127.0.0.1:[1]")
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--time_stamp', type=str)
parser.add_argument('--cuda_device', type=str, default=None)
input_args = parser.parse_args()

datasetCategories = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                    }

# Profiled relative speech w.r.t. Mobilenet
model_factor = {'shufflenet': 0.0644/0.0554,
    'albert': 0.335/0.0554,
    'resnet': 0.135/0.0554,
}

def parse_job_conf(conf_str):
    job_parser = argparse.ArgumentParser(prog="job config parser")

    # job_parser.add_argument('--job_name', type=str, default='demo_job') # in args
    job_parser.add_argument('--app_name', type=str, default='demo_job')
    job_parser.add_argument('--log_path', type=str, default='./', 
            help="default path is ../log") # in args

    # The basic configuration of the cluster
    job_parser.add_argument('--ps_ip', type=str, default='127.0.0.1')  # in args
    job_parser.add_argument('--ps_port', type=str, default='29501')
    job_parser.add_argument('--this_rank', type=int, default=1)
    job_parser.add_argument('--connection_timeout', type=int, default=60)
    job_parser.add_argument('--experiment_mode', type=str, default=events.SIMULATION_MODE)
    job_parser.add_argument('--engine', type=str, default=events.PYTORCH, 
                        help="Tensorflow or Pytorch for cloud aggregation")
    job_parser.add_argument('--num_executors', type=int, default=1)
    job_parser.add_argument('--executor_configs', type=str, default="127.0.0.1:[1]")  # seperated by ;
    job_parser.add_argument('--total_worker', type=int, default=4) # in args
    job_parser.add_argument('--data_map_file', type=str, default=None) # in args
    job_parser.add_argument('--use_cuda', type=str, default='True')
    job_parser.add_argument('--cuda_device', type=str, default=None)
    job_parser.add_argument('--time_stamp', type=str, default='logs') # in args
    job_parser.add_argument('--task', type=str, default='cv')
    job_parser.add_argument('--device_avail_file', type=str, default=None) # in args
    job_parser.add_argument('--clock_factor', type=float, default=1.0, 
                        help="Refactor the clock time given the profile")

    # The configuration of model and dataset
    job_parser.add_argument('--data_dir', type=str, default='~/cifar10/') # in args
    job_parser.add_argument('--device_conf_file', type=str, default='/tmp/client.cfg') # in args
    job_parser.add_argument('--model', type=str, default='shufflenet_v2_x2_0') # in args
    job_parser.add_argument('--data_set', type=str, default='cifar10') # in args
    job_parser.add_argument('--sample_mode', type=str, default='random')
    job_parser.add_argument('--filter_less', type=int, default=32) # in args
    job_parser.add_argument('--filter_more', type=int, default=1e15)
    job_parser.add_argument('--train_uniform', type=bool, default=False)
    job_parser.add_argument('--conf_path', type=str, default='~/dataset/')
    job_parser.add_argument('--overcommitment', type=float, default=1.3)
    job_parser.add_argument('--model_size', type=float, default=65536)
    job_parser.add_argument('--round_threshold', type=float, default=30)
    job_parser.add_argument('--round_penalty', type=float, default=2.0)
    job_parser.add_argument('--clip_bound', type=float, default=0.9)
    job_parser.add_argument('--blacklist_rounds', type=int, default=-1)
    job_parser.add_argument('--blacklist_max_len', type=float, default=0.3)
    job_parser.add_argument('--embedding_file', type=str, default = 'glove.840B.300d.txt')


    # The configuration of different hyper-parameters for training
    job_parser.add_argument('--rounds', type=int, default=50) # in args
    job_parser.add_argument('--local_steps', type=int, default=20)
    job_parser.add_argument('--batch_size', type=int, default=30)
    job_parser.add_argument('--test_bsz', type=int, default=128)
    job_parser.add_argument('--backend', type=str, default="gloo")
    job_parser.add_argument('--upload_step', type=int, default=20)
    job_parser.add_argument('--learning_rate', type=float, default=5e-2)
    job_parser.add_argument('--min_learning_rate', type=float, default=5e-5)
    job_parser.add_argument('--input_dim', type=int, default=0)
    job_parser.add_argument('--output_dim', type=int, default=0)
    job_parser.add_argument('--dump_epoch', type=int, default=1e10)
    job_parser.add_argument('--decay_factor', type=float, default=0.98)
    job_parser.add_argument('--decay_round', type=float, default=10)
    job_parser.add_argument('--num_loaders', type=int, default=2) # in args
    job_parser.add_argument('--eval_interval', type=int, default=5) # in args
    job_parser.add_argument('--sample_seed', type=int, default=233) #123 #233
    job_parser.add_argument('--test_ratio', type=float, default=1.0)
    job_parser.add_argument('--loss_decay', type=float, default=0.2)
    job_parser.add_argument('--exploration_min', type=float, default=0.3)
    job_parser.add_argument('--cut_off_util', type=float, default=0.05) # 95 percentile

    job_parser.add_argument('--gradient_policy', type=str, default=None) # in args

    # for yogi
    job_parser.add_argument('--yogi_eta', type=float, default=3e-3) # in args
    job_parser.add_argument('--yogi_tau', type=float, default=1e-8)
    job_parser.add_argument('--yogi_beta', type=float, default=0.9)
    job_parser.add_argument('--yogi_beta2', type=float, default=0.99)


    # for prox
    job_parser.add_argument('--proxy_mu', type=float, default=0.1)

    # for detection
    job_parser.add_argument('--cfg_file', type=str, default='./utils/rcnn/cfgs/res101.yml')
    job_parser.add_argument('--test_output_dir', type=str, default='./logs/server')
    job_parser.add_argument('--train_size_file', type=str, default='')
    job_parser.add_argument('--test_size_file', type=str, default='')
    job_parser.add_argument('--data_cache', type=str, default='')
    job_parser.add_argument('--backbone', type=str, default='./resnet50.pth')


    # for malicious
    job_parser.add_argument('--malicious_factor', type=int, default=1e15)

    # for differential privacy
    job_parser.add_argument('--noise_factor', type=float, default=0.1)
    job_parser.add_argument('--clip_threshold', type=float, default=3.0)
    job_parser.add_argument('--target_delta', type=float, default=0.0001)

    # for Oort
    job_parser.add_argument('--pacer_delta', type=float, default=5)
    job_parser.add_argument('--pacer_step', type=int, default=20)
    job_parser.add_argument('--exploration_alpha', type=float, default=0.3)
    job_parser.add_argument('--exploration_factor', type=float, default=0.9)
    job_parser.add_argument('--exploration_decay', type=float, default=0.98)
    job_parser.add_argument('--sample_window', type=float, default=5.0)

    # for albert
    job_parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    job_parser.add_argument('--clf_block_size', type=int, default=32)


    job_parser.add_argument(
        "--mlm", type=bool, default=False, help="Train with masked-language modeling loss instead of language modeling."
    )
    job_parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    job_parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    job_parser.add_argument(
        "--block_size",
        default=64,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )


    job_parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
    job_parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    # for tag prediction
    job_parser.add_argument("--vocab_token_size", type=int, default=10000, help="For vocab token size")
    job_parser.add_argument("--vocab_tag_size", type=int, default=500, help="For vocab tag size")

    # for rl example
    job_parser.add_argument("--epsilon", type=float, default=0.9, help="greedy policy")
    job_parser.add_argument("--gamma", type=float, default=0.9, help="reward discount")
    job_parser.add_argument("--memory_capacity", type=int, default=2000, help="memory capacity")
    job_parser.add_argument("--target_replace_iter", type=int, default=15, help="update frequency")
    job_parser.add_argument("--n_actions", type=int, default=2, help="action number")
    job_parser.add_argument("--n_states", type=int, default=4, help="state number")



    # for speech
    job_parser.add_argument("--num_classes", type=int, default=35, help="For number of classes in speech")


    # for voice
    job_parser.add_argument('--train-manifest', metavar='DIR',
                        help='path to train manifest csv', default='data/train_manifest.csv')
    job_parser.add_argument('--test-manifest', metavar='DIR',
                        help='path to test manifest csv', default='data/test_manifest.csv')
    job_parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
    job_parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
    job_parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
    job_parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
    job_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    job_parser.add_argument('--hidden-size', default=256, type=int, help='Hidden size of RNNs')
    job_parser.add_argument('--hidden-layers', default=7, type=int, help='Number of RNN layers')
    job_parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
    job_parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='Finetune the model from checkpoint "continue_from"')
    job_parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true',
                        help='Use random tempo and gain perturbations.')
    job_parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                        help='Use simple spectral augmentation on mel spectograms.')
    job_parser.add_argument('--noise-dir', default=None,
                        help='Directory to inject noise into audio. If default, noise Inject not added')
    job_parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
    job_parser.add_argument('--noise-min', default=0.0,
                        help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
    job_parser.add_argument('--noise-max', default=0.5,
                        help='Maximum noise levels to sample from. Maximum 1.0', type=float)
    job_parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                        help='Turn off bi-directional RNNs, introduces lookahead convolution')
    
    conf = job_parser.parse_args(conf_str)
    conf.use_cuda = eval(conf.use_cuda)
    conf.num_class = datasetCategories.get(conf.data_set, 10)
    for model_name in model_factor:
        if model_name in conf.model:
            conf.clock_factor = conf.clock_factor * model_factor[model_name]
            break
    return conf



# parse job config
yaml_conf = load_yaml_conf(input_args.yaml_file)

args_dict = {}
task_set = set()
for job_name in yaml_conf['job_conf']:
    conf = yaml_conf['job_conf'][job_name]

    dict = {}
    for entry in conf:
        dict.update(entry)

    conf_script = ""
    for conf_name in dict:
        if isinstance(dict[conf_name], str) and '$FEDSCALE_HOME' in dict[conf_name]:
            dict[conf_name] = dict[conf_name].replace('$FEDSCALE_HOME', os.environ["FEDSCALE_HOME"])
        conf_script += (f' --{conf_name}={dict[conf_name]}')
    
    res = parse_job_conf(shlex.split(conf_script))
    res.this_rank = input_args.this_rank
    res.num_executors = input_args.num_executors
    res.executor_configs = input_args.executor_configs
    res.ps_ip = input_args.ps_ip
    res.time_stamp = input_args.time_stamp
    res.cuda_device = input_args.cuda_device
    task_set.add(res.task)
    args_dict[job_name] = res










"""
parser = argparse.ArgumentParser()

parser.add_argument('--job_name', type=str, default='demo_job') # in args
parser.add_argument('--log_path', type=str, default='./', 
        help="default path is ../log") # in args

# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')  # in args
parser.add_argument('--ps_port', type=str, default='29501')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--connection_timeout', type=int, default=60)
parser.add_argument('--experiment_mode', type=str, default=events.SIMULATION_MODE)
parser.add_argument('--engine', type=str, default=events.PYTORCH, 
                    help="Tensorflow or Pytorch for cloud aggregation")
parser.add_argument('--num_executors', type=int, default=1)
parser.add_argument('--executor_configs', type=str, default="127.0.0.1:[1]")  # seperated by ;
parser.add_argument('--total_worker', type=int, default=4) # in args
parser.add_argument('--data_map_file', type=str, default=None) # in args
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--cuda_device', type=str, default=None)
parser.add_argument('--time_stamp', type=str, default='logs') # in args
parser.add_argument('--task', type=str, default='cv')
parser.add_argument('--device_avail_file', type=str, default=None) # in args
parser.add_argument('--clock_factor', type=float, default=1.0, 
                    help="Refactor the clock time given the profile")

# The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='~/cifar10/') # in args
parser.add_argument('--device_conf_file', type=str, default='/tmp/client.cfg') # in args
parser.add_argument('--model', type=str, default='shufflenet_v2_x2_0') # in args
parser.add_argument('--data_set', type=str, default='cifar10') # in args
parser.add_argument('--sample_mode', type=str, default='random')
parser.add_argument('--filter_less', type=int, default=32) # in args
parser.add_argument('--filter_more', type=int, default=1e15)
parser.add_argument('--train_uniform', type=bool, default=False)
parser.add_argument('--conf_path', type=str, default='~/dataset/')
parser.add_argument('--overcommitment', type=float, default=1.3)
parser.add_argument('--model_size', type=float, default=65536)
parser.add_argument('--round_threshold', type=float, default=30)
parser.add_argument('--round_penalty', type=float, default=2.0)
parser.add_argument('--clip_bound', type=float, default=0.9)
parser.add_argument('--blacklist_rounds', type=int, default=-1)
parser.add_argument('--blacklist_max_len', type=float, default=0.3)
parser.add_argument('--embedding_file', type=str, default = 'glove.840B.300d.txt')


# The configuration of different hyper-parameters for training
parser.add_argument('--rounds', type=int, default=50) # in args
parser.add_argument('--local_steps', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--test_bsz', type=int, default=128)
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--upload_step', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=5e-2)
parser.add_argument('--min_learning_rate', type=float, default=5e-5)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--dump_epoch', type=int, default=1e10)
parser.add_argument('--decay_factor', type=float, default=0.98)
parser.add_argument('--decay_round', type=float, default=10)
parser.add_argument('--num_loaders', type=int, default=2) # in args
parser.add_argument('--eval_interval', type=int, default=5) # in args
parser.add_argument('--sample_seed', type=int, default=233) #123 #233
parser.add_argument('--test_ratio', type=float, default=1.0)
parser.add_argument('--loss_decay', type=float, default=0.2)
parser.add_argument('--exploration_min', type=float, default=0.3)
parser.add_argument('--cut_off_util', type=float, default=0.05) # 95 percentile

parser.add_argument('--gradient_policy', type=str, default=None) # in args

# for yogi
parser.add_argument('--yogi_eta', type=float, default=3e-3) # in args
parser.add_argument('--yogi_tau', type=float, default=1e-8)
parser.add_argument('--yogi_beta', type=float, default=0.9)
parser.add_argument('--yogi_beta2', type=float, default=0.99)


# for prox
parser.add_argument('--proxy_mu', type=float, default=0.1)

# for detection
parser.add_argument('--cfg_file', type=str, default='./utils/rcnn/cfgs/res101.yml')
parser.add_argument('--test_output_dir', type=str, default='./logs/server')
parser.add_argument('--train_size_file', type=str, default='')
parser.add_argument('--test_size_file', type=str, default='')
parser.add_argument('--data_cache', type=str, default='')
parser.add_argument('--backbone', type=str, default='./resnet50.pth')


# for malicious
parser.add_argument('--malicious_factor', type=int, default=1e15)

# for differential privacy
parser.add_argument('--noise_factor', type=float, default=0.1)
parser.add_argument('--clip_threshold', type=float, default=3.0)
parser.add_argument('--target_delta', type=float, default=0.0001)

# for Oort
parser.add_argument('--pacer_delta', type=float, default=5)
parser.add_argument('--pacer_step', type=int, default=20)
parser.add_argument('--exploration_alpha', type=float, default=0.3)
parser.add_argument('--exploration_factor', type=float, default=0.9)
parser.add_argument('--exploration_decay', type=float, default=0.98)
parser.add_argument('--sample_window', type=float, default=5.0)

# for albert
parser.add_argument(
    "--line_by_line",
    action="store_true",
    help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument('--clf_block_size', type=int, default=32)


parser.add_argument(
    "--mlm", type=bool, default=False, help="Train with masked-language modeling loss instead of language modeling."
)
parser.add_argument(
    "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
)
parser.add_argument(
    "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
)
parser.add_argument(
    "--block_size",
    default=64,
    type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).",
)


parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

# for tag prediction
parser.add_argument("--vocab_token_size", type=int, default=10000, help="For vocab token size")
parser.add_argument("--vocab_tag_size", type=int, default=500, help="For vocab tag size")

# for rl example
parser.add_argument("--epsilon", type=float, default=0.9, help="greedy policy")
parser.add_argument("--gamma", type=float, default=0.9, help="reward discount")
parser.add_argument("--memory_capacity", type=int, default=2000, help="memory capacity")
parser.add_argument("--target_replace_iter", type=int, default=15, help="update frequency")
parser.add_argument("--n_actions", type=int, default=2, help="action number")
parser.add_argument("--n_states", type=int, default=4, help="state number")



# for speech
parser.add_argument("--num_classes", type=int, default=35, help="For number of classes in speech")


# for voice
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/test_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=256, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=7, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true',
                    help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')

args, unknown = parser.parse_known_args()
args.use_cuda = eval(args.use_cuda)


datasetCategories = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                    }

# Profiled relative speech w.r.t. Mobilenet
model_factor = {'shufflenet': 0.0644/0.0554,
    'albert': 0.335/0.0554,
    'resnet': 0.135/0.0554,
}


args.num_class = datasetCategories.get(args.data_set, 10)
for model_name in model_factor:
    if model_name in args.model:
        args.clock_factor = args.clock_factor * model_factor[model_name]
        break
"""
