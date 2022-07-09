# -*- coding: utf-8 -*-

from cmath import log
from concurrent.futures import thread

from tomlkit import string
# from fedscale.core import response
from fedscale.core.fl_aggregator_libs import *
from fedscale.core.resource_manager import ResourceManager
from fedscale.core import events, executor
from fedscale.core import job_api_pb2
import fedscale.core.job_api_pb2_grpc as job_api_pb2_grpc

import torch
from torch.utils.tensorboard import SummaryWriter
import threading
import pickle
import grpc
from concurrent import futures
from argparse import Namespace

MAX_MESSAGE_LENGTH = 1*1024*1024*1024 # 1GB

class Aggregator(job_api_pb2_grpc.JobServiceServicer):
    """This centralized aggregator collects training/testing feedbacks from executors"""
    def __init__(self, args_dict):
        for job_name, a in args_dict.items():
            logging.info(f"Job args {job_name}: {a}")
            logging.info("")

        self.args_dict = args_dict
        self.demo_arg = list(args_dict.values())[0]
        self.experiment_mode = self.demo_arg.experiment_mode
        self.device = self.demo_arg.cuda_device if self.demo_arg.use_cuda else torch.device('cpu')

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.round_duration = {job_name: 0. for job_name in self.args_dict}
        self.resource_manager = ResourceManager(self.experiment_mode) # TODO: check if this needs to be modified in multi-job setting
        self.client_manager = self.init_client_manager() # TODO: check if need lock

        # ======== model and data ========
        self.model = {job_name: None for job_name in self.args_dict}
        self.model_in_update = {job_name: 0 for job_name in self.args_dict}
        self.update_lock = {job_name: threading.Lock() for job_name in self.args_dict}
        self.model_weights = {job_name: collections.OrderedDict() for job_name in self.args_dict}  # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.last_gradient_weights = {job_name: [] for job_name in self.args_dict} # only gradient variables
        # self.model_state_dict = None # not used
        # NOTE: if <param_name, param_tensor> (e.g., model.parameters() in PyTorch), then False
        # True, if <param_name, list_param_tensors> (e.g., layer.get_weights() in Tensorflow)
        self.using_group_params = self.demo_arg.engine == events.TENSORFLOW 

        # ======== channels ========
        self.connection_timeout = self.demo_arg.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}    # Unicast
        self.sever_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque() # Broadcast

        # ======== runtime information ========
        self.tasks_round = {job_name: 0 for job_name in self.args_dict}
        self.num_of_clients = {job_name: 0 for job_name in self.args_dict}

        # NOTE: sampled_participants = sampled_executors in deployment,
        # because every participant is an executor. However, in simulation mode,
        # executors is the physical machines (VMs), thus:
        # |sampled_executors| << |sampled_participants| as an VM may run multiple participants
        self.sampled_participants = {job_name: [] for job_name in self.args_dict} # TODO: check this 
        self.sampled_executors = [] # TODO: check if need to use dict

        self.round_stragglers = {job_name: [] for job_name in self.args_dict}
        self.model_update_size = {job_name: 0. for job_name in self.args_dict}

        # self.collate_fn = None # not used
        self.task = {job_name: args_dict[job_name].task for job_name in self.args_dict}
        self.round = {job_name: 0 for job_name in self.args_dict}

        self.start_run_time = time.time()
        self.client_conf = {} # TODO: check if this really unused

        self.stats_util_accumulator = {job_name: [] for job_name in self.args_dict}
        self.loss_accumulator = {job_name: [] for job_name in self.args_dict}
        self.client_training_results = {job_name: [] for job_name in self.args_dict}

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = {job_name: [] for job_name in self.args_dict}
        self.testing_history = {job_name: 
                                    {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                                    'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()} 
                                for job_name, args in self.args_dict.items()} 

        # self.log_writer = SummaryWriter(log_dir=logDir)


        self.virtual_client_clock = {job_name: {} for job_name in self.args_dict}
        self.flatten_client_duration = {job_name: None for job_name in self.args_dict}

        self.stop_lock = threading.Lock()
        self.stop_exectuors = 0
        self.shutdown = False

        # ======== Task specific ============
        self.init_task_context()



    def setup_env(self):
        self.setup_seed(seed=1)
        self.optimizer = {job_name: ServerOptimizer(args.gradient_policy, args, self.device) for job_name, args in self.args_dict.items()}


    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Initiating control plane communication ...")
        if self.experiment_mode == events.SIMULATION_MODE:
            num_of_executors = 0
            for ip_numgpu in self.demo_arg.executor_configs.split("="):
                ip, numgpu = ip_numgpu.split(':')
                for numexe in numgpu.strip()[1:-1].split(','):
                    for _ in range(int(numexe.strip())):
                        num_of_executors += 1
            self.executors = list(range(num_of_executors))
        else:
            self.executors = list(range(self.args_dict.total_worker))

        # initiate a server process
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, self.grpc_server)
        port = '[::]:{}'.format(self.demo_arg.ps_port)

        logging.info(f'%%%%%%%%%% Opening aggregator sever using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()


    def init_data_communication(self):
        """For jumbo traffics (e.g., training results).
        """
        pass


    def init_model(self):
        """Load model"""
        assert self.demo_arg.engine == events.PYTORCH, "Please define model for non-PyTorch models"

        self.model = {job_name: init_model(job_name, args) for job_name, args in self.args_dict.items()}


        # Initiate model parameters dictionary <param_name, param>
        self.model_weights = {job_name : model.state_dict() for job_name, model in self.model.items()}

    def init_task_context(self):
        """Initiate execution context for specific tasks"""

        self.imdb = dict()
        for job_name, args in self.args_dict.items():
            if args.task == "detection":
                cfg_from_file(self.args_dict.cfg_file)
                np.random.seed(self.cfg.RNG_SEED)
                self.imdb[job_name], _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args_dict.data_dir], server=True)
            else:
                self.imdb[job_name] = None # TODO: check if this is correct

        # if self.args_dict.task == "detection":
        #     cfg_from_file(self.args_dict.cfg_file)
        #     np.random.seed(self.cfg.RNG_SEED)
        #     self.imdb, _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args_dict.data_dir], server=True)


    def init_client_manager(self):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Oort sampler
                - Oort prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai
        """

        # sample_mode: random or oort
        client_manager = {}
        for job_name, args in self.args_dict.items():
            client_manager[job_name] = clientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        """For Simulation Mode: load client profiles/traces"""
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def client_register_handler(self, executorId, info_dict):
        """Triggered once receive new executor registration"""
        assert sorted(list(self.args_dict.keys())) == sorted(list(info_dict.keys()))
        for job_name, info in info_dict.items():
            logging.info(f"Loading executor[{executorId}] for jobname {job_name} {len(info['size'])} client traces ...")
            for _size in info['size']:
                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = (self.num_of_clients[job_name]+1)%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})

                clientId = (self.num_of_clients[job_name]+1) if self.experiment_mode == events.SIMULATION_MODE else executorId
                self.client_manager[job_name].registerClient(executorId, clientId, size=_size, speed=systemProfile)
                # TODO: check if multi-job is compatible with oort
                self.client_manager[job_name].registerDuration(clientId, batch_size=self.args_dict[job_name].batch_size,
                    upload_step=self.args_dict[job_name].local_steps, upload_size=self.model_update_size[job_name], download_size=self.model_update_size[job_name])
                self.num_of_clients[job_name] += 1
            logging.info("Info of all feasible clients {}".format(self.client_manager[job_name].getDataInfo()))
            assert self.client_manager[job_name].getDataInfo()['total_feasible_clients'] <= len(info['size'])


    def executor_info_handler(self, executorId, info):

        self.registered_executor_info.add(executorId)
        logging.info(f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == events.SIMULATION_MODE:
            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                # start to sample clients
                self.round_completion_handler()
        else:
            # In real deployments, we need to register for ealsch client
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                self.round_completion_handler()


    def tictak_client_tasks(self, job_name, sampled_clients, num_clients_to_collect):
        if self.experiment_mode == events.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            completionTimes = []
            completed_client_clock = {}
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                # client_cfg = self.client_conf[job_name].get(client_to_run, self.args_dict)
                client_cfg = self.args_dict[job_name]

                exe_cost = self.client_manager[job_name].getCompletionTime(client_to_run,
                                        batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                        upload_size=self.model_update_size[job_name], download_size=self.model_update_size[job_name])

                roundDuration = exe_cost['computation'] + exe_cost['communication']
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager[job_name].isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
            top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            return (clients_to_run, dummy_clients, 
                    completed_client_clock, round_duration, 
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client:{'computation': 1, 'communication':1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock, 
                1, completionTimes)


    def run(self):
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        for job_name in self.args_dict:
            self.save_last_param(job_name)
        self.model_update_size = {job_name: sys.getsizeof(pickle.dumps(model))/1024.0*8. for job_name, model in self.model.items()} # kbits
        self.client_profiles = self.load_client_profile(file_path=self.demo_arg.device_conf_file)
        self.event_monitor()


    def select_participants(self, job_name, select_num_participants, overcommitment=1.3, busy_clients=None):
        return sorted(self.client_manager[job_name].resampleClients(
            int(select_num_participants*overcommitment), 
            cur_time=self.global_virtual_clock,
            busy_clients=busy_clients))


    def client_completion_handler(self, job_name, results):
        """We may need to keep all updates from clients, 
        if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args_dict[job_name].gradient_policy in ['q-fedavg']:
            self.client_training_results[job_name].append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator[job_name].append(results['utility'])
        self.loss_accumulator[job_name].append(results['moving_loss'])

        self.client_manager[job_name].registerScore(results['clientId'], results['utility'],
            auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.round[job_name],
            duration=self.virtual_client_clock[job_name][results['clientId']]['computation']+
                self.virtual_client_clock[job_name][results['clientId']]['communication']
        )

        # ================== Aggregate weights ======================
        self.update_lock[job_name].acquire()

        self.model_in_update[job_name] += 1 
        if self.using_group_params == True:
            self.aggregate_client_group_weights(results)
        else:
            self.aggregate_client_weights(job_name, results)

        self.update_lock[job_name].release()
    
    def aggregate_client_weights(self, job_name, results):
        """May aggregate client updates on the fly"""
        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round

        for p in results['update_weight']:
            param_weight = results['update_weight'][p]
            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(param_weight).to(device=self.device)

            if self.model_in_update[job_name] == 1:
                self.model_weights[job_name][p].data = param_weight
            else:
                self.model_weights[job_name][p].data += param_weight

        if self.model_in_update[job_name] == self.tasks_round[job_name]:
            for p in self.model_weights[job_name]:
                d_type = self.model_weights[job_name][p].data.dtype

                self.model_weights[job_name][p].data = (
                    self.model_weights[job_name][p]/float(self.tasks_round[job_name])).to(dtype=d_type)


    def aggregate_client_group_weights(self, results):
        """Streaming weight aggregation. Similar to aggregate_client_weights, 
        but each key corresponds to a group of weights (e.g., for Tensorflow)"""

        for p_g in results['update_weight']:
            param_weights = results['update_weight'][p_g]
            for idx, param_weight in enumerate(param_weights):
                if isinstance(param_weight, list):
                    param_weight = np.asarray(param_weight, dtype=np.float32)
                param_weight = torch.from_numpy(param_weight).to(device=self.device)

                if self.model_in_update == 1:
                    self.model_weights[p_g][idx].data = param_weight
                else:
                    self.model_weights[p_g][idx].data += param_weight

        if self.model_in_update == self.tasks_round:
            for p in self.model_weights:
                for idx in range(len(self.model_weights[p])):
                    d_type = self.model_weights[p][idx].data.dtype

                    self.model_weights[p][idx].data = (
                        self.model_weights[p][idx].data/float(self.tasks_round)
                    ).to(dtype=d_type)


    # load the last model weights to last_gradient_weights
    def save_last_param(self, job_name):
        if self.demo_arg.engine == events.TENSORFLOW:
            self.last_gradient_weights = [layer.get_weights() for layer in self.model.layers]
        else:
            # self.last_gradient_weights = {job_name: [p.data.clone() for p in model.parameters()] for job_name, model in self.model.items()}
            self.last_gradient_weights[job_name] = [p.data.clone() for p in self.model[job_name].parameters()]


    def round_weight_handler(self, job_name):
        """Update model when the round completes"""
        if self.round[job_name] > 1:
            if self.demo_arg.engine == events.TENSORFLOW:
                for layer in self.model.layers:
                    layer.set_weights([p.cpu().detach().numpy() for p in self.model_weights[layer.name]])
                # TODO: support update round gradient
            else:
                self.model[job_name].load_state_dict(self.model_weights[job_name])
                current_grad_weights = [param.data.clone() for param in self.model[job_name].parameters()]
                self.optimizer[job_name].update_round_gradient(self.last_gradient_weights[job_name], current_grad_weights, self.model[job_name])


    def round_completion_handler(self):
        self.global_virtual_clock += np.max(list(self.round_duration.values()))
        assert not self.resource_manager.has_next_task(), f'round_completion_handler: resource manager has pending tasks, {self.resource_manager.get_queue()}'
        
        self.sampled_participants = {job_name: [] for job_name in self.args_dict}
        for job_name in args_dict:
            self.round[job_name] += 1
            if self.round[job_name] % args_dict[job_name].decay_round == 0:
                self.args_dict[job_name].learning_rate = max(self.args_dict[job_name].learning_rate*self.args_dict[job_name].decay_factor, self.args_dict[job_name].min_learning_rate)

            # handle the global update w/ current and last
            self.round_weight_handler(job_name)

            avgUtilLastround = sum(self.stats_util_accumulator[job_name])/max(1, len(self.stats_util_accumulator[job_name]))
            # assign avg reward to explored, but not finished workers
            for clientId in self.round_stragglers[job_name]:
                self.client_manager[job_name].registerScore(clientId, avgUtilLastround,
                        time_stamp=self.round[job_name],
                        duration=self.virtual_client_clock[job_name][clientId]['computation']+self.virtual_client_clock[job_name][clientId]['communication'],
                        success=False)

            avg_loss = sum(self.loss_accumulator[job_name])/max(1, len(self.loss_accumulator[job_name]))

            logging.info(f"[{job_name}]: Wall clock: {round(self.global_virtual_clock)} s, round: {self.round[job_name]}, Planned participants: " + \
                f"{len(self.sampled_participants[job_name])}, Succeed participants: {len(self.stats_util_accumulator[job_name])}, Training loss: {avg_loss}")
            
            # dump round completion information to tensorboard
            # if len(self.loss_accumulator[job_name]):
            #     self.log_train_result(avg_loss)

            # update select participants
            self.sampled_participants[job_name] = self.select_participants(
                            job_name = job_name,
                            select_num_participants=self.args_dict[job_name].total_worker,
                            overcommitment=self.args_dict[job_name].overcommitment,
                            busy_clients = [client for l in list(self.sampled_participants.values()) for client in l])

            (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
                        job_name, self.sampled_participants[job_name], self.args_dict[job_name].total_worker)

            logging.info(f"[{job_name}]: Selected participants to run: {clientsToRun}, finishes round at {self.global_virtual_clock+round_duration} (duration: {round_duration})")

            # Issue requests to the resource manager; Tasks ordered by the completion time
            self.resource_manager.register_tasks(job_name, clientsToRun)
            self.tasks_round[job_name] = len(clientsToRun)

            # Update executors and participants
            if self.experiment_mode == events.SIMULATION_MODE:
                self.sampled_executors = list(self.individual_client_events.keys())
            else:
                self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]

            self.save_last_param(job_name)
            self.round_stragglers[job_name] = round_stragglers
            self.virtual_client_clock[job_name] = virtual_client_clock # exe_cost (compute_Cost, communication_cost)
            # self.flatten_client_duration[job_name] = numpy.array(flatten_client_duration) # the round time of each client compute + communicatoin
            self.round_duration[job_name] = round_duration
            
            self.model_in_update[job_name] = 0
            self.test_result_accumulator[job_name] = []
            self.stats_util_accumulator[job_name] = []
            self.client_training_results[job_name] = []

            if self.round[job_name] >= self.args_dict[job_name].rounds:
                self.broadcast_aggregator_events(job_name, events.SHUT_DOWN)
            elif self.round[job_name] % self.args_dict[job_name].eval_interval == 0:
                self.broadcast_aggregator_events(job_name, events.UPDATE_MODEL)
                self.broadcast_aggregator_events(job_name, events.MODEL_TEST)
            else:
                self.broadcast_aggregator_events(job_name, events.UPDATE_MODEL)
                self.broadcast_aggregator_events(job_name, events.START_ROUND)

        logging.info(f'after round completion handler: {self.broadcast_events_queue}')



    # def log_train_result(self, avg_loss):
    #     """Result will be post on TensorBoard"""
    #     self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
    #     self.log_writer.add_scalar('FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock/60.)
    #     self.log_writer.add_scalar('FAR/round_duration (min)', self.round_duration/60., self.round)
    #     self.log_writer.add_histogram('FAR/client_duration (min)', self.flatten_client_duration, self.round)

    # def log_test_result(self):
    #     self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
    #     self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
    #     self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
    #                                 self.global_virtual_clock/60.)
    #     self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
    #                                 self.global_virtual_clock/60.)


    def deserialize_response(self, responses):
        return pickle.loads(responses)

    def serialize_response(self, responses):
        return pickle.dumps(responses)

    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset"""

        results = results['results']

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args_dict.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args_dict.task == "detection":
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                }
            else:
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'test_len': accumulator['test_len']
                }


            logging.info("FL Testing in round: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(self.round, self.global_virtual_clock, self.testing_history['perf'][self.round]['top_1'],
                    self.testing_history['perf'][self.round]['top_5'], self.testing_history['perf'][self.round]['loss'],
                    self.testing_history['perf'][self.round]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_test_result()

            self.broadcast_events_queue.append(events.START_ROUND)


    def broadcast_aggregator_events(self, job_name, event):
        """Issue tasks (events) to aggregator worker processes"""
        self.broadcast_events_queue.append((job_name, event))

    def dispatch_client_events(self, job_name, event, clients=None):
        """Issue tasks (events) to clients"""
        if clients is None:
            clients = self.sampled_executors

        for client_id in clients:
            self.individual_client_events[client_id].append((job_name, event))

    def get_client_conf(self, job_name):
        """Training configurations that will be applied on clients"""

        conf = {
            'learning_rate': self.args_dict[job_name].learning_rate,
            'model': None # none indicates we are using the global model
        }
        return conf

    def create_client_task(self, job_name):
        """Issue a new client training task to the executor"""
        task = self.resource_manager.get_next_task(job_name)
        train_config = None
        model = None
        if task is not None:
            job_name, client_id = task
            config = self.get_client_conf(job_name)
            train_config = {'job_name': job_name, 'client_id': client_id, 'task_config': config}
        return train_config, model


    def get_test_config(self, client_id):
        """FL model testing on clients"""

        return {'client_id':client_id}

    def get_global_model(self, job_name):
        """Get global model that would be used by all FL clients (in default FL)"""
        return self.model[job_name]

    def get_shutdown_config(self, client_id):
        return {'client_id': client_id}

    def add_event_handler(self, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first."""
        self.sever_events_queue.append((client_id, event, meta, data))


    def CLIENT_REGISTER(self, request, context):
        """FL Client register to the aggregator"""

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id uses the same executor_id (VMs) in simulations
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        if executor_id not in self.individual_client_events:
            # logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info(f"Previous client: {executor_id} resumes connecting")

        # We can customize whether to admit the clients here
        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(events.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(event=events.DUMMY_EVENT,
                meta=dummy_data, data=dummy_data)


    def CLIENT_PING(self, request, context):
        """Handle client requests"""

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = events.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = events.DUMMY_EVENT
            response_data = response_msg = events.DUMMY_RESPONSE
        else:
            job_name, current_event = self.individual_client_events[executor_id].popleft()
            if current_event == events.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(job_name)
                if response_msg is not None:
                    job_name = response_msg['job_name']
                    job_client_id = response_msg['client_id']
                else:
                    current_event = events.DUMMY_EVENT
                    if self.experiment_mode != events.SIMULATION_MODE:
                        self.individual_client_events[executor_id].appendleft((job_name, events.CLIENT_TRAIN))
            elif current_event == events.MODEL_TEST:
                response_msg = self.get_test_config(executor_id)
            elif current_event == events.UPDATE_MODEL:
                response_msg = {'job_name': job_name}
                response_data = self.get_global_model(job_name)
            elif current_event == events.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)
                self.stop_lock.acquire()
                self.stop_exectuors += 1
                self.stop_lock.release()
                    

        if current_event == events.DUMMY_EVENT:
            logging.info(f"[Ping Handler] Issue TASK {current_event} to EXECUTOR ({executor_id})")
        elif current_event != events.CLIENT_TRAIN:
            logging.info(f"[Ping Handler] Issue TASK ({current_event}, {job_name}) to EXECUTOR ({executor_id})")
        else:
            logging.info(f"[Ping Handler] Issue TASK ({current_event}, {job_name}, {job_client_id}) to EXECUTOR ({executor_id})")

        response_msg, response_data = self.serialize_response(response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        return job_api_pb2.ServerResponse(event=current_event,
                meta=response_msg, data=response_data)


    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task."""

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == events.CLIENT_TRAIN:
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            # meta represents job_name
            job_name = meta_result
            logging.info(f"[Complete Handler] executor[{executor_id}] completes -> event: {event} meta: {meta_result} client: {client_id}")
            if execution_status is False:
                logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")
            if self.resource_manager.has_next_task(executor_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                self.individual_client_events[executor_id].appendleft((job_name, events.CLIENT_TRAIN))

        elif event in (events.MODEL_TEST, events.UPLOAD_MODEL):
            logging.info(f'executor[{executor_id}] completes -> event: {event} meta: {meta_result} client: {client_id}')
            self.add_event_handler(executor_id, event, meta_result, data_result)
            
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")
        return self.CLIENT_PING(request, context)


    def event_monitor(self):
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if self.shutdown and self.stop_exectuors == len(self.individual_client_events):
                break 
            elif len(self.broadcast_events_queue) > 0:
                job_name, current_event = self.broadcast_events_queue.popleft()

                if current_event in (events.UPDATE_MODEL, events.MODEL_TEST):
                    self.dispatch_client_events(job_name, current_event)

                elif current_event == events.START_ROUND:
                    self.dispatch_client_events(job_name, events.CLIENT_TRAIN)

                elif current_event == events.SHUT_DOWN:
                    self.shutdown = True
                    self.dispatch_client_events(job_name, events.SHUT_DOWN)
                    # logging.info('Aggregator shuts down')
                    # break

            # Handle events queued on the aggregator
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()

                if current_event == events.UPLOAD_MODEL:
                    job_name = meta
                    assert job_name in self.args_dict, f'{job_name}'
                    self.client_completion_handler(job_name, self.deserialize_response(data))

                    round_completed = True
                    for job_name in self.args_dict:
                        if len(self.stats_util_accumulator[job_name]) != self.tasks_round[job_name]:
                            round_completed = False
                            break
                    if round_completed:
                        self.round_completion_handler()

                elif current_event == events.MODEL_TEST:
                    self.testing_completion_handler(client_id, self.deserialize_response(data))

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)


    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)

if __name__ == "__main__":
    aggregator = Aggregator(args_dict)
    aggregator.run()
