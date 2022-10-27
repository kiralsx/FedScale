import math
from random import Random
import pickle
import logging

from fedscale.core.helper.client import Client

class clientManager(object):

    def __init__(self, mode, args, sample_seed=233):
        self.Clients = {}
        self.clientOnHosts = {}
        self.mode = mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        self.ucbSampler = None 

        if self.mode == 'oort': 
            import sys,os
            current = os.path.dirname(os.path.realpath(__file__))
            parent = os.path.dirname(current)
            sys.path.append(parent)
            from thirdparty.oort.oort import create_training_selector
            #sys.path.append(current) 
            self.ucbSampler = create_training_selector(args=args)
        elif self.mode == 'multi_aaai':
            from thirdparty.multi_aaai.multi_aaai import MultiSampler
            self.multiSampler = MultiSampler()


        self.feasibleClients = []
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args

        # for multi scheduling aaai paper
        self.client_selection_time = dict()

        if args.device_avail_file is not None:
            with open(args.device_avail_file, 'rb') as fin:
                self.user_trace = pickle.load(fin)
            self.user_trace_keys = list(self.user_trace.keys())

    def registerClient(self, hostId, clientId, size, speed, duration=1):

        uniqueId = self.getUniqueId(hostId, clientId)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(clientId)%len(self.user_trace)]]

        self.Clients[uniqueId] = Client(hostId, clientId, speed, user_trace)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(clientId)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward':min(size, self.args.local_steps*self.args.batch_size),
                            'duration':duration,
                            }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)
            elif self.mode == "multi_aaai":
                self.multiSampler.register_client(clientId)
        else:
            del self.Clients[uniqueId]


    def getAllClients(self):
        return self.feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients)

    def getClient(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)]

    def registerDuration(self, clientId, batch_size, upload_step, upload_size, download_size):
        if self.getUniqueId(0, clientId) not in self.Clients or self.mode=='random':
            return
        exe_cost = self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                batch_size=batch_size, upload_step=upload_step,
                upload_size=upload_size, download_size=download_size
        )
        if self.mode == "oort":
            self.ucbSampler.update_duration(clientId, exe_cost['computation']+exe_cost['communication'])
        elif self.mode == "multi_aaai":
            self.multiSampler.register_round_time(clientId, exe_cost['computation']+exe_cost['communication'])


    def getCompletionTime(self, clientId, batch_size, upload_step, upload_size, download_size):
        
        return self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
                batch_size=batch_size, upload_step=upload_step,
                upload_size=upload_size, download_size=download_size
            )

    def registerSpeed(self, hostId, clientId, speed):
        uniqueId = self.getUniqueId(hostId, clientId)
        self.Clients[uniqueId].speed = speed

    def registerScore(self, clientId, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward
        if self.mode == "oort":
            feedbacks = {
                'reward': reward,
                'duration': duration,
                'status': True,
                'time_stamp': time_stamp
            }

            self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)

    def registerPerf(self, clientId, round, stats_util, duration):
        self.Clients[self.getUniqueId(0, clientId)].register_perf(round=round, perf={'stats util': stats_util, 'duration': duration})

    def getPerf(self):
        res = {}
        for client_id, client in self.Clients.items():
            res[client_id] = client.get_perf()
        return res

    def registerClientScore(self, clientId, reward):
        self.Clients[self.getUniqueId(0, clientId)].registerReward(reward)

    def getScore(self, hostId, clientId):
        uniqueId = self.getUniqueId(hostId, clientId)
        return self.Clients[uniqueId].getScore()

    def getClientsInfo(self):
        clientInfo = {}
        for i, clientId in enumerate(self.Clients.keys()):
            client = self.Clients[clientId]
            clientInfo[client.clientId] = client.distance
        return clientInfo

    def nextClientIdToRun(self, hostId):
        init_id = hostId - 1
        lenPossible = len(self.feasibleClients)

        while True:
            clientId = str(self.feasibleClients[init_id])
            csize = self.Clients[clientId].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(clientId)

            init_id = max(0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

    def getUniqueId(self, hostId, clientId):
        return str(clientId)
        #return (str(hostId) + '_' + str(clientId))

    def clientSampler(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def clientOnHost(self, clientIds, hostId):
        self.clientOnHosts[hostId] = clientIds

    def getCurrentClientIds(self, hostId):
        return self.clientOnHosts[hostId]

    def getClientLenOnHost(self, hostId):
        return len(self.clientOnHosts[hostId])

    def getClientSize(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].size

    def getSampleRatio(self, clientId, hostId, even=False):
        totalSampleInTraining = 0.

        if not even:
            for key in self.clientOnHosts.keys():
                for client in self.clientOnHosts[key]:
                    uniqueId = self.getUniqueId(key, client)
                    totalSampleInTraining += self.Clients[uniqueId].size

            #1./len(self.clientOnHosts.keys())
            return float(self.Clients[self.getUniqueId(hostId, clientId)].size)/float(totalSampleInTraining)
        else:
            for key in self.clientOnHosts.keys():
                totalSampleInTraining += len(self.clientOnHosts[key])

            return 1./totalSampleInTraining

    def getFeasibleClients(self, cur_time):
        if self.user_trace is None:
            clients_online = self.feasibleClients
        else:
            clients_online = [clientId for clientId in self.feasibleClients if self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)]

        logging.info(f"Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " + \
                    f"{len(self.feasibleClients)-len(clients_online)} clients offline")

        return clients_online

    def isClientActive(self, clientId, cur_time):
        return self.Clients[self.getUniqueId(0, clientId)].isActive(cur_time)

    def resampleClients(self, numOfClients, cur_time=0., busy_clients=None):
        self.count += 1

        clients_online = self.getFeasibleClients(cur_time)
        assert len(set(clients_online)) == len(clients_online), f'clients_online contain repeated client_id'

        # only pick not busy clients
        if busy_clients is not None:
            # check the correctness of busy clients
            for client_id, slots in busy_clients.items():
                for i in range(len(slots)-1):
                    for j in range(i+1, len(slots)):
                        assert slots[i][0] >= slots[j][1] or slots[i][1] <= slots[j][0], f'client {client_id} slots overlap {slots[i]} {slots[j]}'

            clients_available = []
            for x in clients_online:
                if x not in busy_clients:
                    clients_available.append(x)
                else:
                    aval = True
                    for slots in busy_clients[x]:
                        if cur_time >= slots[0] and cur_time < slots[1]:
                            aval = False
                            break
                    if aval:
                        clients_available.append(x)
            clients_online = clients_available


        # if busy_clients is not None:
        #     clients_online = [x for x in clients_online if x not in busy_clients]

        if len(clients_online) <= numOfClients:
            return clients_online

        picked_clients = None
        clients_online_set = set(clients_online)
        assert len(clients_online_set) == len(clients_online)


        if self.mode == "oort" and self.count > 1:
            picked_clients = self.ucbSampler.select_participant(numOfClients, feasible_clients=clients_online_set)
        elif self.mode == 'multi_aaai':
            picked_clients = self.multiSampler.sample(clients_online, numOfClients)
        else:
            self.rng.shuffle(clients_online)
            client_len = min(numOfClients, len(clients_online) -1)
            picked_clients = clients_online[:client_len]

        return picked_clients

    def getAllMetrics(self):
        if self.mode == "oort":
            return self.ucbSampler.getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, clientId):
        return self.ucbSampler.get_client_reward(clientId)

    def get_median_reward(self):
        if self.mode == 'oort':
            return self.ucbSampler.get_median_reward()
        return 0.

