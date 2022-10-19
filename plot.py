import pickle
import matplotlib.pyplot as plt
import numpy as np

file_path = '/workspace/FedScale/dataset/data/device_info/client_device_capacity'
with open(file_path, 'rb') as fin:
    # {clientId: [computer, bandwidth]}
    global_client_profile = pickle.load(fin)

computation = [v['computation'] for _, v in global_client_profile.items()]
communication = [v['communication'] for _, v in global_client_profile.items()]

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(computation)
 
# show plot
plt.savefig('./computation.png')


fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(communication)
 
# show plot
plt.savefig('./communication.png')