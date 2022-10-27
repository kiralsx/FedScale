import cvxpy as cp
import numpy as np
import logging


class MultiSampler(object):
    def __init__(self):
        # for multi scheduling aaai paper
        self.client_freq = dict()
        self.client_round_time = dict()
    

    def register_client(self, client_id):
        # init the selection time of clients 
        self.client_freq[client_id] = 0


    def register_round_time(self, client_id, round_time):
        if client_id not in self.client_freq:
            return
        self.client_round_time[client_id] = round_time


    def sample(self, available_clients, num_clients_to_select):
        assert len(self.client_round_time) == len(self.client_freq)
        
        x = cp.Variable(shape=(len(available_clients)), boolean=True)
        # frequency
        freq = np.array([self.client_freq[cid] for cid in available_clients])
        freq = (x + freq)
        avg_freq = cp.sum(freq) / x.shape[0]
        freq_score = cp.sum(cp.square(freq - avg_freq)) / x.shape[0]

        # round time
        round_time = np.array([self.client_round_time[cid] for cid in available_clients])
        time_score = cp.max(cp.multiply(x,round_time))

        obj_expr = 0.2 * time_score + 0.8 * freq_score
        obj = cp.Minimize(obj_expr)
        constraints = [cp.sum(x) == num_clients_to_select]
        problem = cp.Problem(obj, constraints=constraints)
        problem.solve(solver=cp.SCIP, verbose=False)
        if problem.status != 'optimal':
            logging.info(f"Status: {problem.status}")
            logging.info(f"Problem: {problem}")
            logging.info(f"The optimal value is: {problem.value}")
            logging.info("A solution x is")
            logging.info(f'{np.round(x.value)}')

        x = np.round(x.value, decimals=2)
        logging.info(f'{x}')
        selected_clients = [i for i in range(x.shape[0]) if x[i]==1]
        assert len(selected_clients) == num_clients_to_select
        logging.info(f"obj: {problem.value}, time_score:{time_score.value}, freq_score: {freq_score.value}, selected_clients: {selected_clients}")
        
        # update freq
        for cid in selected_clients:
            self.client_freq[cid] += 1

        return selected_clients