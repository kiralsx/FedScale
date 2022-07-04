# package for aggregator
from fedscale.core.fllibs import *
# import os

# logDir = os.path.join(list(args_dict.values())[0].log_path, list(args_dict.values())[0].time_stamp, 'aggregator')
logDir = os.path.join(list(args_dict.values())[0].log_path, 'recent', 'aggregator')

logFile = os.path.join(logDir, 'log')

def init_logging():
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)
    logging.basicConfig(
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='(%m-%d) %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logFile, mode='a'),
                        logging.StreamHandler()
                    ])

    logging.info(f"logDir {logDir}")

def dump_ps_ip(args):
    hostname_map = {}
    with open('ipmapping', 'rb') as fin:
        hostname_map = pickle.load(fin)

    ps_ip = str(hostname_map[str(socket.gethostname())])
    args.ps_ip = ps_ip

    with open(os.path.join(logDir, 'ip'), 'wb') as fout:
        pickle.dump(ps_ip, fout)

    logging.info(f"Load aggregator ip: {ps_ip}")


def initiate_aggregator_setting():
    init_logging()
    #dump_ps_ip()

initiate_aggregator_setting()
