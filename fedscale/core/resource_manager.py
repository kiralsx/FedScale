import logging
from fedscale.core import events
import threading
import collections

class ResourceManager(object):
    """Schedule training tasks across GPUs/CPUs"""
    def __init__(self, experiment_mode):
        assert experiment_mode == events.SIMULATION_MODE, 'resource manager does not support deploy mode'
        self.queue = collections.deque()
        self.experiment_mode = experiment_mode
        self.update_lock = threading.Lock()

    def get_queue(self):
        self.update_lock.acquire()
        res = self.queue.copy()
        self.update_lock.release()
        return res

    def get_first(self):
        self.update_lock.acquire()
        res = self.queue[0]
        self.update_lock.release()
        return res

    def register_tasks(self, job_name, clientsToRun):
        self.update_lock.acquire()
        for client_id in clientsToRun:
            # assert len([x for x in self.queue if x[1]==client_id]) == 0, f'resource_manager: {client_id} has already been assigned a task {[x for x in self.queue if x[1]==client_id]}'
            self.queue.append((job_name, client_id))
        self.update_lock.release()

    def has_next_task(self, job_name=None, client_id=None):
        self.update_lock.acquire()
        if self.experiment_mode == events.SIMULATION_MODE:
            exist_next_task = len(self.queue) > 0
        else:
            exist_next_task = (job_name, client_id) in self.queue
        self.update_lock.release()
        return exist_next_task
    
    def get_next_task(self, job_name, client_id=None):
        if self.experiment_mode != events.SIMULATION_MODE:
            raise NotImplementedError(f'must be in simultion mode')

        next_task = None

        if self.has_next_task():
            first_task = self.get_first()
            if first_task[0] == job_name:
                self.update_lock.acquire()
                next_task = self.queue.popleft()
                self.update_lock.release()
            else:
                self.update_lock.acquire()
                assert len([t for t in self.queue if t[0]==job_name]) == 0, f'non-sequential task for {job_name} in resource manager: {self.queue}'
                self.update_lock.release()
        return next_task
