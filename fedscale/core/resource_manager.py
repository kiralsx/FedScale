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
            assert len([x for x in self.queue if x[1]==client_id]) == 0, f'resource_manager: {client_id} has already been assigned a task {[x for x in self.queue if x[1]==client_id]}'
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
        next_task = None
        self.update_lock.acquire()
        if self.experiment_mode == events.SIMULATION_MODE:
            if len(self.queue) > 0:
                next_task = self.queue.popleft()
                if next_task[0] != job_name:
                    self.queue.appendleft(next_task)
                    assert job_name not in [job_name for (job_name, _) in self.queue], f'unsequential {job_name} in queue'
                    next_task = None
        else:
            if (job_name, client_id) in self.queue:
                next_task = (job_name, client_id)
                self.queue.remove((job_name, client_id))
        self.update_lock.release()
        return next_task

        



    


# class ResourceManager2(object):
#     """Schedule training tasks across GPUs/CPUs"""

#     def __init__(self, experiment_mode):

#         self.client_run_queue = []
#         self.client_run_queue_idx = 0
#         self.experiment_mode = experiment_mode
#         self.update_lock = threading.Lock()
    
#     def register_tasks(self, clientsToRun):
#         self.client_run_queue = clientsToRun.copy()
#         self.client_run_queue_idx = 0

#     def remove_client_task(self, client_id):
#         assert(client_id in self.client_run_queue, 
#             f"client task {client_id} is not in task queue")
#         pass

#     def has_next_task(self, client_id=None):
#         exist_next_task = False
#         if self.experiment_mode == events.SIMULATION_MODE:
#             exist_next_task = self.client_run_queue_idx < len(self.client_run_queue)
#         else:
#             exist_next_task = client_id in self.client_run_queue
#         return exist_next_task

#     def get_next_task(self, client_id=None):
#         next_task_id = None
#         self.update_lock.acquire()
#         if self.experiment_mode == events.SIMULATION_MODE:
#             if self.has_next_task(client_id):
#                 next_task_id = self.client_run_queue[self.client_run_queue_idx]
#                 self.client_run_queue_idx += 1
#         else:
#             if client_id in self.client_run_queue:
#                 next_task_id = client_id
#                 self.client_run_queue.remove(next_task_id)

#         self.update_lock.release()
#         return next_task_id
