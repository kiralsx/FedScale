from fedscale.dataloaders import divide_data, utils_data
# from fedscale.core.fllibs import *
from fedscale.dataloaders.femnist import FEMNIST
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os



""" ############ check femnist data map ############"""
# train_transform, test_transform = utils_data.get_data_transform('mnist')
# train_dataset = FEMNIST('./dataset/data/femnist', dataset='train', transform=train_transform)
# test_dataset = FEMNIST('./dataset/data/femnist', dataset='test', transform=test_transform)

# job_parser = argparse.ArgumentParser(prog="job config parser")
# args = job_parser.parse_args("")
# args.task = 'cv'


# training_sets = divide_data.DataPartitioner(data=train_dataset, args=args, numOfClass=62)
# training_sets.partition_data_helper(num_clients=None, data_map_file='./dataset/data/femnist/client_data_mapping/train.csv')
# #testing_sets = DataPartitioner(data=test_dataset, args=args, numOfClass=62, isTest=True)
# #testing_sets.partition_data_helper(num_clients=None, data_map_file='./dataset/data/femnist/client_data_mapping/train.csv')

# client_size = training_sets.getSize()
# print(client_size)



""" ############ check google speech data map ############"""
# from torchvision import datasets, transforms
# from fedscale.dataloaders.speech import SPEECH
# from fedscale.dataloaders.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
# from fedscale.dataloaders.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
# from fedscale.dataloaders.speech import BackgroundNoiseDataset

# job_parser = argparse.ArgumentParser(prog="job config parser")
# args = job_parser.parse_args("")
# args.data_dir = '/workspace/FedScale/dataset/data/google_speech'
# args.task='speech'

# bkg = '_background_noise_'
# data_aug_transform = transforms.Compose(
#     [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
# bg_dataset = BackgroundNoiseDataset(os.path.join(args.data_dir, bkg), data_aug_transform)
# add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
# train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
# train_dataset = SPEECH(args.data_dir, dataset= 'train',
#                         transform=transforms.Compose([LoadAudio(),
#                                 data_aug_transform,
#                                 add_bg_noise,
#                                 train_feature_transform]))
# valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
# test_dataset = SPEECH(args.data_dir, dataset='test',
#                         transform=transforms.Compose([LoadAudio(),
#                                 FixAudioLength(),
#                                 valid_feature_transform]))


# training_sets = divide_data.DataPartitioner(data=train_dataset, args=args, numOfClass=62)
# training_sets.partition_data_helper(num_clients=None, data_map_file='/workspace/FedScale/dataset/data/google_speech/client_data_mapping/train.csv')
# #testing_sets = DataPartitioner(data=test_dataset, args=args, numOfClass=62, isTest=True)
# #testing_sets.partition_data_helper(num_clients=None, data_map_file='./dataset/data/femnist/client_data_mapping/train.csv')

# client_size = training_sets.getSize()
# print(len(client_size['size']))



""" ############ check system / stats utility ############"""
# client_perf = json.load(open('/workspace/FedScale/evals/client_perf.txt'))['femnist-1']
client_perf = json.load(open('/workspace/FedScale/evals/client_perf.txt'))['femnist-1']

round_duration = []
for client_id, util in client_perf.items():
    round_duration.append(util['1']['duration'])
round_duration_norm = []
for x in round_duration:
    round_duration_norm.append((x - np.min(round_duration)) / (np.max(round_duration) - np.min(round_duration)))

round_duration = round_duration_norm
round_duration = [1-x for x in round_duration]

num_rounds = len(client_perf['1'])
num_clients = len(client_perf)
stats_util = np.zeros((num_rounds, num_clients))
for i in range(1, num_rounds+1):
    for j in range(1, num_clients+1):
        stats_util[i-1][j-1] = client_perf[str(j)][str(i)]['stats util']

for i in range(num_rounds):
    norm = []
    util = stats_util[i]
    for x in util:
        norm.append((x - np.min(util)) / (np.max(util) - np.min(util)))
    stats_util[i] = norm

stats_util_dict = {}
stats_util_dict['femnist-1'] = stats_util

# at a singel time slot
# fig, ax = plt.subplots()
# time_slot = 0
# ax.plot(range(len(round_duration)), round_duration, 'x--', label="system utility", markevery=1, markersize=3, markerfacecolor = 'none', linewidth = 1)
# ax.plot(range(len(round_duration)), stats_util[time_slot], 'o--', label="stats utility", markevery=1, markersize=3, markerfacecolor = 'none', linewidth = 1)
# ax.set_xlabel("Client ID", fontsize=15)
# ax.set_ylabel("Utility", fontsize=15)
# # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# # ax.grid()
# # ax.set_xlim([0, T])
# # ax.set_ylim(0)
# # ax.tick_params(axis='both', labelsize=13)
# ax.legend(loc="best")
# plt.tight_layout()
# plt.savefig("./util_speech.png")

# trend
# fig, ax = plt.subplots()
# for client_id in range(10):
#     ax.plot(range(num_rounds), stats_util[:,client_id].reshape(-1), 'x--', label=f"client {client_id}", markevery=1, markersize=3, markerfacecolor = 'none', linewidth = 1)
# ax.set_xlabel("Round", fontsize=15)
# ax.set_ylabel("Stats Utility", fontsize=15)
# # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# # ax.grid()
# # ax.set_xlim([0, T])
# # ax.set_ylim(0)
# # ax.tick_params(axis='both', labelsize=13)
# ax.legend(loc="best")
# plt.tight_layout()
# plt.savefig("./util_trend_speech.png")



client_perf = json.load(open('/workspace/FedScale/evals/client_perf.txt'))['speech-1']

round_duration = []
for client_id, util in client_perf.items():
    round_duration.append(util['1']['duration'])
round_duration_norm = []
for x in round_duration:
    round_duration_norm.append((x - np.min(round_duration)) / (np.max(round_duration) - np.min(round_duration)))

round_duration = round_duration_norm
round_duration = [1-x for x in round_duration]

num_rounds = len(client_perf['1'])
num_clients = len(client_perf)
stats_util = np.zeros((num_rounds, num_clients))
for i in range(1, num_rounds+1):
    for j in range(1, num_clients+1):
        stats_util[i-1][j-1] = client_perf[str(j)][str(i)]['stats util']

for i in range(num_rounds):
    norm = []
    util = stats_util[i]
    for x in util:
        norm.append((x - np.min(util)) / (np.max(util) - np.min(util)))
    stats_util[i] = norm

stats_util_dict['speech-1'] = stats_util

time_slot = 2
fig, ax = plt.subplots()
ax.plot(range(num_clients), stats_util_dict['femnist-1'][time_slot], 'x--', label="femnist", markevery=1, markersize=3, markerfacecolor = 'none', linewidth = 1)
ax.plot(range(num_clients), stats_util_dict['speech-1'][time_slot], 'o--', label="speech", markevery=1, markersize=3, markerfacecolor = 'none', linewidth = 1)
ax.set_xlabel("Client ID", fontsize=15)
ax.set_ylabel("Stats Utility", fontsize=15)
# ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# ax.grid()
# ax.set_xlim([0, T])
# ax.set_ylim(0)
# ax.tick_params(axis='both', labelsize=13)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("./multi_util.png")