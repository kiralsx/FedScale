import cvxpy as cp
import numpy as np

def create_training_selector():
    return _training_selector()

class _training_selector(object):
    def __init__(self):
        pass

    def sample(client_round_time, client_selection_frequency):
        assert len(client_round_time) == len(client_selection_frequency)
        
        avg_freq = np.mean(list(client_selection_frequency.values()))
        freq_score = 



        x = cp.Variable(shape=(3), boolean=True)
        v = np.asarray([1,2,3])
        obj_expr = cp.max(cp.multiply(x,v))
        obj = cp.Minimize(obj_expr)
        constraints = [cp.sum(x) == 2]
        problem = cp.Problem(obj, constraints=constraints)
        problem.solve(solver=cp.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF'}, verbose=False)