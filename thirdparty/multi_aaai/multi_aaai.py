import cvxpy as cp
import numpy as np
import logging
import math
import copy

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


class MultiSampler(object):
    def __init__(self):
        # for multi scheduling aaai paper
        self.client_freq = dict()
        self.client_round_time = dict()
        self.sample_size = 100
        self.observations = dict()
    

    def register_client(self, client_id):
        # init the selection time of clients 
        self.client_freq[client_id] = 0


    def register_round_time(self, client_id, round_time):
        if client_id not in self.client_freq:
            return
        self.client_round_time[client_id] = round_time


    def sample(self, available_clients, num_clients_to_select):
        assert len(self.client_round_time) == len(self.client_freq)
        
        num_valid_solns = nCr(len(available_clients), num_clients_to_select)
        # num_sampled_solns = int(self.sample_perc * num_valid_solns)
        logging.info(f'[multi_aaai] sampled {self.sample_size} out of {num_valid_solns} solutions')

        observations = copy.copy(self.observations)

        for _ in range(min(num_valid_solns, self.sample_size)):
            sol = np.sort(np.random.choice(available_clients, num_clients_to_select, replace=False))
            assert len(sol) == len(set(sol))

            time_score = np.max([self.client_round_time[cid] for cid in sol])

            freq = copy.copy(self.client_freq)
            for cid in sol:
                freq[cid] += 1

            mean_freq = np.mean(list(freq.values()))

            stats_score = np.mean([(v-mean_freq)**2 for v in freq.values()])

            score = 0.001 * time_score + stats_score

            observations[tuple(sol)] = score
        
        opt_sol, opt_score = min(observations.items(), key=lambda x: x[1]) 

        self.observations[opt_sol] = opt_score

        # update freq
        for cid in opt_sol:
            self.client_freq[cid] += 1 
        
        return list(opt_sol)


    def sample_opt(self, available_clients, num_clients_to_select):
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

        obj_expr = 0.2/60 * time_score + freq_score
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
        logging.info(f"obj: {problem.value}, time_score:{0.2/60 * time_score.value}, freq_score: {freq_score.value}, selected_clients: {selected_clients}")
        
        # update freq
        for cid in selected_clients:
            self.client_freq[cid] += 1

        return selected_clients