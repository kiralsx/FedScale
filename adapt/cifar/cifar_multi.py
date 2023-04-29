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
import sys
import gc
import pickle

from models import *

import adaptdl
import adaptdl.torch as adl
# from adaptdl.torch.epoch import current_epoch
from adaptdl.torch.epoch import AdaptiveEpoch
from adaptdl.checkpoint import save_all_states, _STATES_TO_NAMES, _NAMES_TO_STATES
from adaptdl.torch._metrics import MetricsState

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import time
import os
# import multiprocessing
import torch.multiprocessing as mp
from numba import cuda
from queue import Empty


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False



def worker(queue, args, current_time, job_dir):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"### device: {device}")

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


    if adaptdl.env.replica_rank() == 0:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        
        # dist.barrier()  # We use a barrier here so that non-master replicas would wait for master to download the data
    else:
        raise NotImplementedError
        dist.barrier()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

    validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    metrics = MetricsState(ckpt_dir=job_dir)

    # global adaptive_epoch
    adaptive_epoch = AdaptiveEpoch(ckpt_dir=job_dir)

    # global trainloader
    trainloader = adl.AdaptiveDataLoader(trainset, adaptive_epoch, metrics, ckpt_dir=job_dir, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    if args.autoscale_bsz:
        trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1980), gradient_accumulation=False)

    # global validloader
    # validloader = adl.AdaptiveDataLoader(validset, adaptive_epoch, metrics, batch_size=100, shuffle=False, num_workers=2)
    validloader = DataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

    # global model
    # Model
    print('==> Building model..')
    model = eval(args.model)()
    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    # global criterion
    criterion = nn.CrossEntropyLoss()
    # global optimizer
    optimizer = optim.SGD([{"params": [param]} for param in model.parameters()],
                        lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # global lr_scheduler
    lr_scheduler = MultiStepLR(optimizer, [30, 45], 0.1)

    # global scaler
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # global net
    net = adl.AdaptiveDataParallel(model, optimizer, metrics, lr_scheduler, scaler, ckpt_dir=job_dir)


    # train
    def train(epoch, current_time):
        net.train()
        # stats = adl.Accumulator(adaptive_epoch)
        stats = {}
        stats["loss_sum"] = 0
        stats["total"] = 0
        stats["correct"] = 0

        first_time_in_epoch = True
        epoch_start_time = time.time()
        for _, (inputs, targets) in enumerate(trainloader):
            if first_time_in_epoch:
                print(f"###### current bsz: {trainloader.current_batch_size}")
                first_time_in_epoch = False
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
        current_time = current_time + epoch_time

        # global writer
        trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data/")
        net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model/")
        if args.mixed_precision:
            writer.add_scalar("MixedPrecision/scale", scaler.get_scale(), epoch)
        # with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Epoch to Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Epoch Accuracy/Train", stats["accuracy"], epoch)
        writer.add_scalar("Time to Loss/Train", stats["loss_avg"], current_time)
        writer.add_scalar("Time to Accuracy/Train", stats["accuracy"], current_time)
        writer.add_scalar("Time to Epoch", current_time, epoch)
        print("Train:", stats)

        return current_time

    def valid(epoch, current_time):
        net.eval()
        # stats = adl.Accumulator(adaptive_epoch)
        stats = {}
        stats["loss_sum"] = 0
        stats["total"] = 0
        stats["correct"] = 0
        with torch.no_grad():
            for inputs, targets in validloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                stats["loss_sum"] += loss.detach().item() * targets.size(0)
                _, predicted = outputs.max(1)
                stats["total"] += targets.size(0)
                stats["correct"] += predicted.eq(targets).sum().detach().item()

        # with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Epoch to Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Epoch to Accuracy/Valid", stats["accuracy"], epoch)
        writer.add_scalar("Time to Loss/Valid", stats["loss_avg"], current_time)
        writer.add_scalar("Time to Accuracy/Valid", stats["accuracy"], current_time)
        print("Valid:", stats)

    with SummaryWriter(job_dir) as writer:
        for epoch in adaptive_epoch.remaining_epochs_until(args.epochs):
            print(f'\n---------------------- Epoch: {epoch}')
            current_time = train(epoch, current_time)
            valid(epoch, current_time)
            lr_scheduler.step()

            if args.ckpt:
                break

    if args.ckpt:
        save_all_states(job_dir)

    queue.put((current_time, adaptive_epoch.finished_epochs()))


if __name__ == '__main__':


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

    if args.autoscale_bsz:
        # tensorboard_dir = "/workspace/FedScale/adapt/cifar/output/autoscale-ckpt" 
        # ckpt_dir = f"/workspace/FedScale/adapt/cifar/output/autoscale-ckpt"
        job_dir = f"/workspace/FedScale/adapt/cifar/output/autoscale-ckpt-test"
    else:
        # tensorboard_dir = f"/workspace/FedScale/adapt/cifar/output/{args.bs}-ckpt"
        # ckpt_dir = f"/workspace/FedScale/adapt/cifar/output/{args.bs}-ckpt"
        job_dir = f"/workspace/FedScale/adapt/cifar/output/{args.bs}-ckpt-test"

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)


    finished = False

    current_epoch = 0
    current_time = 0
    while current_epoch < args.epochs:
    # for i in range(2):
        # print(f"\n######### Epoch: {current_epoch}")
        # mp.set_start_method('spawn', force=True)
        q = mp.Queue()
        p = mp.Process(target=worker, args=(q, args, current_time, job_dir))
        p.start()
        p.join()        

        print(f"### after process, memory allocated: {torch.cuda.memory_allocated()*1e-9}, max memory allocated: {torch.cuda.max_memory_allocated()*1e-9}, memory reserved: {torch.cuda.memory_reserved()*1e-9}, max memory reserved: {torch.cuda.max_memory_reserved()*1e-9}")
        try:
            current_time, current_epoch = q.get(block=False)
        except Empty:
            print(f"Nothing get")
            exit()