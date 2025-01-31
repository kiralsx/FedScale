# -*- coding: utf-8 -*-
from fedscale.core.fl_client_libs import *
from argparse import Namespace
import gc
import collections
import torch
import pickle

from fedscale.core.client import Client
from fedscale.core.rlclient import RLClient
from fedscale.core import events
from fedscale.core.communication.channel_context import ClientConnections
import fedscale.core.job_api_pb2 as job_api_pb2


class Executor(object):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""
    def __init__(self, args_dict):

        self.args_dict = args_dict
        self.demo_arg = list(args_dict.values())[0]
        self.device = self.demo_arg.cuda_device if self.demo_arg.use_cuda else torch.device('cpu')
        self.num_executors = self.demo_arg.num_executors
        # ======== env information ========
        self.this_rank = self.demo_arg.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.model = self.training_sets = self.test_dataset = None
        # self.temp_model_path = {job_name: os.path.join(logDir, 'model_'+job_name+'-'+str(self.this_rank)+'.pth.tar') for job_name in self.args_dict}
        self.temp_model_path = {job_name: os.path.join(logDir, job_name + '_exe_' + str(self.this_rank) + '.pth.tar') for job_name in self.args_dict}

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(self.demo_arg.ps_ip, self.demo_arg.ps_port)

        # ======== runtime information ========
        self.collate_fn = {job_name: None for job_name in self.args_dict} #TODO: check how to modify this
        self.task = {job_name: args.task for job_name, args in self.args_dict.items()}
        self.round = {job_name: 0 for job_name in self.args_dict}
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        super(Executor, self).__init__()

    def setup_env(self):
        logging.info(f"(EXECUTOR{self.this_rank}): is setting up environ ...")
        self.setup_seed(seed=1)

    def setup_communication(self):
        self.init_control_communication()
        self.init_data_communication()


    def setup_seed(self, seed=1):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages."""
        self.aggregator_communicator.connect_to_server()


    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)
        """
        pass

    def init_model(self):
        """Return the model architecture used in training"""
        assert self.demo_arg.engine == events.PYTORCH, "Please override this function to define non-PyTorch models"
        return {job_name: init_model(job_name, args).to(device=self.device) for job_name, args in self.args_dict.items()}

        # model = init_model()
        # model = model.to(device=self.device)
        # return model

    def init_data(self):
        """Return the training and testing dataset"""
        training_sets = {}
        testing_sets = {}
        for job_name, args in self.args_dict.items():
            train_dataset, test_dataset = init_dataset(job_name=job_name, args=args)
            if args.task == "rl":
                return train_dataset, test_dataset
            # load data partitioner (entire_train_data)
            logging.info(f"(EXECUTOR:{self.this_rank}): Data partitioner starts on {job_name}...")
            
            # TODO: should not use args.total_worker when data_map_file is None
            training_sets[job_name] = DataPartitioner(data=train_dataset, args=args, numOfClass=args.num_class)
            training_sets[job_name].partition_data_helper(num_clients=args.total_worker, data_map_file=args.data_map_file)

            testing_sets[job_name] = DataPartitioner(data=test_dataset, args=args, numOfClass=args.num_class, isTest=True)
            testing_sets[job_name].partition_data_helper(num_clients=self.num_executors)

            logging.info(f"(EXECUTOR:{self.this_rank}): Data partitioner completes ...")

            if self.task == 'nlp':
                self.collate_fn[job_name] = MyCollator(tokenizer=tokenizer[job_name])
            elif self.task == 'voice':
                self.collate_fn[job_name] = voice_collate_fn
            
        return training_sets, testing_sets

        """
        train_dataset, test_dataset = init_dataset()
        if self.task == "rl":
            return train_dataset, test_dataset
        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(data=train_dataset, args = self.args, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(data=test_dataset, args = self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        if self.task == 'nlp':
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets
        """


    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.training_sets, self.testing_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()

    def dispatch_worker_events(self, request):
        """Add new events to worker queues"""
        self.event_queue.append(request)

    def deserialize_response(self, responses):
        return pickle.loads(responses)

    def serialize_response(self, responses):
        return pickle.dumps(responses)

    def UpdateModel(self, job_name, model):
        """Receive the broadcasted global model for current round"""
        self.update_model_handler(job_name=job_name, model=model)

    def Train(self, config):
        """Load train config and data to start training on client """
        job_name, client_id, train_config = config['job_name'], config['client_id'], config['task_config']

        model = None
        if 'model' in train_config and train_config['model'] is not None:
            model = train_config['model']

        client_conf = self.override_conf(job_name, train_config)
        train_res = self.training_handler(job_name=job_name, clientId=client_id, conf=client_conf, model=model)

        # # Report execution completion meta information
        # response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
        #     job_api_pb2.CompleteRequest(
        #         client_id = str(client_id), executor_id = self.executor_id,
        #         event = events.CLIENT_TRAIN, status = True, msg = None,
        #         meta_result = job_name, data_result = None
        #     )
        # )

        # self.dispatch_worker_events(response)
        return client_id, train_res

    def Test(self, job_name):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group"""

        test_res = self.testing_handler(job_name, args=self.args_dict[job_name])
        test_res = {'executorId': self.this_rank, 'results': test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id = self.executor_id, executor_id = self.executor_id,
                event = events.MODEL_TEST, status = True, msg = None,
                meta_result = job_name, data_result = self.serialize_response(test_res)
            )
        )
        self.dispatch_worker_events(response)


    def Stop(self):
        """Stop the current executor"""

        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True


    def report_executor_info_handler(self):
        """Return the statistics of training dataset"""
        return {job_name: self.training_sets[job_name].getSize() for job_name in self.training_sets}
        # return self.training_sets.getSize()


    def update_model_handler(self, job_name, model):
        """Update the model copy on this executor"""
        self.model[job_name] = model
        self.round[job_name] += 1

        # Dump latest model to disk
        # logging.info(f'dump model to {self.temp_model_path[job_name]}')
        with open(self.temp_model_path[job_name], 'wb') as model_out:
            pickle.dump(self.model[job_name], model_out)


    def load_global_model(self, job_name):
        # load last global model
        with open(self.temp_model_path[job_name], 'rb') as model_in:
            model = pickle.load(model_in)
        return model


    def override_conf(self, job_name, config):
        default_conf = vars(self.args_dict[job_name]).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)


    def get_client_trainer(self, conf):
        """Developer can redefine to this function to customize the training:
           API:
            - train(client_data=client_data, model=client_model, conf=conf)
        """
        return Client(conf)


    def training_handler(self, job_name, clientId, conf, model=None):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model(job_name) if model is None else model

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer[job_name] # TODO: change to corresponding tokenizer
        if self.args_dict[job_name].task == "rl":
            client_data = self.training_sets[job_name]
            client = RLClient(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets[job_name], 
                batch_size=conf.batch_size, args = self.args_dict[job_name], 
                collate_fn=self.collate_fn[job_name]
            )

            client = self.get_client_trainer(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)

        return train_res


    def testing_handler(self, job_name, args):
        """Test model"""
        evalStart = time.time()
        device = self.device
        model = self.load_global_model(job_name)
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, model, device=device)
            _, _, _, testResults = test_res
        else:
            data_loader = select_dataset(self.this_rank, self.testing_sets[job_name], 
                batch_size=args.test_bsz, args = args, 
                isTest=True, collate_fn=self.collate_fn[job_name]
            )

            if self.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            if self.demo_arg.engine == events.PYTORCH:
                test_res = test_model(self.this_rank, model, args, data_loader, 
                    device=device, criterion=criterion, tokenizer=tokenizer[job_name]) # TODO: change to corresponding tokenizer
            else:
                raise Exception(f"Need customized implementation for model testing in {self.demo_arg.engine} engine")

            test_loss, acc, acc_5, testResults = test_res
            logging.info("(EXECUTOR:{}): [{}] After aggregation round {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                .format(self.executor_id, job_name, self.round[job_name], round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults

    def client_register(self):
        """Register the executor information to the aggregator"""
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id = self.executor_id,
                        executor_id = self.executor_id,
                        executor_info = self.serialize_response(self.report_executor_info_handler())
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task"""
        response = self.aggregator_communicator.stub.CLIENT_PING(job_api_pb2.PingRequest(
            client_id = self.executor_id,
            executor_id = self.executor_id
        ))
        self.dispatch_worker_events(response)

    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info(f"(EXECUTOR:{self.this_rank}): Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == events.CLIENT_TRAIN:
                    # logging.info(f"executor[{self.executor_id}] receives TRAIN request")
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    train_config['job_name'] = str(train_config['job_name'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    # response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                    response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                        job_api_pb2.CompleteRequest(client_id = str(client_id), executor_id = self.executor_id,
                        event = events.UPLOAD_MODEL, status = True, msg = None,
                        meta_result = train_config['job_name'], data_result = self.serialize_response(train_res)
                    ))

                    self.dispatch_worker_events(response)

                elif current_event == events.MODEL_TEST:
                    # logging.info(f"executor[{self.executor_id}] receives TEST request")
                    meta = self.deserialize_response(request.meta)
                    job_name = str(meta['job_name'])
                    executor_id = str(meta['executor_id'])
                    assert executor_id == self.executor_id, f'executor[{self.executor_id} receives TEST request for executor[{executor_id}]]'
                    self.Test(job_name)

                elif current_event == events.UPDATE_MODEL:
                    # logging.info(f"executor[{self.executor_id}] receives UPDATE request")
                    config = self.deserialize_response(request.meta)
                    job_name = str(config['job_name'])
                    model = self.deserialize_response(request.data)
                    self.UpdateModel(job_name, model)

                elif current_event == events.SHUT_DOWN:
                    logging.info(f"executor[{self.executor_id}] receives SHUTDOWN request")
                    self.Stop()

                elif current_event == events.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()


if __name__ == "__main__":
    executor = Executor(args_dict)
    executor.run()