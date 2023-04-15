# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

import adaptdl
import adaptdl.torch as adl
# from adaptdl.torch.epoch import current_epoch
from adaptdl.torch.epoch import AdaptiveEpoch
from adaptdl.checkpoint import save_all_states

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import time
import os

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--ckpt', default=False,
                    action='store_true', help='use checkpoint after each epoch')
parser.add_argument('--mixed-precision', dest='mixed_precision', default=False,
                    action='store_true', help='use automatic mixed precision')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

global adaptive_epoch
adaptive_epoch = AdaptiveEpoch()

global trainloader
if adaptdl.env.replica_rank() == 0:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    trainloader = adl.AdaptiveDataLoader(trainset, adaptive_epoch, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    dist.barrier()  # We use a barrier here so that non-master replicas would wait for master to download the data
else:
    raise NotImplementedError
    dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

if args.autoscale_bsz:
    trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1980), gradient_accumulation=False)

global validloader
validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
validloader = adl.AdaptiveDataLoader(validset, adaptive_epoch, batch_size=100, shuffle=False, num_workers=2)


def init():


    

    # Model
    print('==> Building model..')
    global net
    net = eval(args.model)()
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    global criterion
    criterion = nn.CrossEntropyLoss()
    global optimizer
    optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
                        lr=args.lr, momentum=0.9, weight_decay=5e-4)
    global lr_scheduler
    lr_scheduler = MultiStepLR(optimizer, [30, 45], 0.1)

    global scaler
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    net = adl.AdaptiveDataParallel(net, optimizer, lr_scheduler, scaler)


def destroy():
    # global trainloader
    # global validloader
    global net

    # del trainloader
    # del validloader
    del net





# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    epoch_start_time = time.time()
    net.train()
    stats = adl.Accumulator(adaptive_epoch)
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        if args.mixed_precision:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        stats["loss_sum"] += loss.detach().item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().detach().item()

    epoch_time = time.time() - epoch_start_time
    global current_time
    current_time += epoch_time

    # global writer
    trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data/")
    net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model/")
    if args.mixed_precision:
        writer.add_scalar("MixedPrecision/scale", scaler.get_scale(), epoch)
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Epoch to Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Epoch Accuracy/Train", stats["accuracy"], epoch)
        writer.add_scalar("Time to Loss/Train", stats["loss_avg"], current_time)
        writer.add_scalar("Time to Accuracy/Train", stats["accuracy"], current_time)
        writer.add_scalar("Time to Epoch", current_time, epoch)
        print("Train:", stats)


def valid(epoch):
    net.eval()
    stats = adl.Accumulator(adaptive_epoch)
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            stats["loss_sum"] += loss.detach().item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().detach().item()

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Epoch to Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Epoch to Accuracy/Valid", stats["accuracy"], epoch)
        writer.add_scalar("Time to Loss/Valid", stats["loss_avg"], current_time)
        writer.add_scalar("Time to Accuracy/Valid", stats["accuracy"], current_time)
        print("Valid:", stats)



current_time = 0



if args.autoscale_bsz:
    tensorboard_dir = "/workspace/FedScale/adapt/cifar/output/autoscale-ckpt" 
    ckpt_dir = f"/workspace/FedScale/adapt/cifar/output/autoscale-ckpt"
else:
    tensorboard_dir = f"/workspace/FedScale/adapt/cifar/output/{args.bs}-ckpt"
    ckpt_dir = f"/workspace/FedScale/adapt/cifar/output/{args.bs}-ckpt"
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

def print_model():
    for name, param in net._state.model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            break


def run(use_ckpt=False):
    
    

    # global first_layer
    # if first_layer is not None:
    #     print(list(net._state.model.state_dict().values())[0] - first_layer)
        # exit()

    not_finished = True

    global writer
    with SummaryWriter(tensorboard_dir) as writer:
        while not_finished:
            print("\n")
            init()
            for epoch in adaptive_epoch.remaining_epochs_until(args.epochs):
                print(f"### memory allocated: {torch.cuda.memory_allocated()*1e-9}, max memory allocated: {torch.cuda.max_memory_allocated()*1e-9}, memory reserved: {torch.cuda.memory_reserved()*1e-9}, max memory reserved: {torch.cuda.max_memory_reserved()*1e-9}")
                # epoch = adaptive_epoch.current_epoch()
                print(f"### lr: {lr_scheduler.get_last_lr()}")
                train(epoch)
                valid(epoch)
                lr_scheduler.step()


                # first_layer = list(net._state.model.state_dict().values())[0] 
                
                if use_ckpt:
                    destroy()
                    break
                    
            if use_ckpt:
                save_all_states(ckpt_dir)
                if adaptive_epoch.finished_epochs() >= args.epochs:
                    not_finished = False 
            else:
                not_finished = False

            

            

# first_layer = None
# for _ in range(args.epochs):
#     run_one_epoch()

run(args.ckpt)





