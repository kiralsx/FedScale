import torch
import torch.nn as nn
# from opacus.grad_sample import GradSampleModule
from torch.utils.data import TensorDataset
import adaptdl
import adaptdl.torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
from adaptdl.checkpoint import save_all_states

if torch.cuda.is_available():  
  device = "cuda:0" 
else:  
  device = "cpu"  

n_input, n_hidden, n_out, dataset_size, learning_rate = 10, 15, 1, 1280, 0.01

data_x = torch.randn(dataset_size, n_input)
data_y = (torch.rand(size=(dataset_size, 1)) < 0.5).float()

# data_x = data_x.to(device)
# data_y = data_y.to(device)

batch_size = 32
dataset = TensorDataset(data_x, data_y)
kwargs = {'batch_size': batch_size}
train_loader = adaptdl.torch.AdaptiveDataLoader(dataset, batch_size=batch_size, drop_last=True)
train_loader.autoscale_batch_size(1028, local_bsz_bounds=(32, 128))


model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())
# model = GradSampleModule(model)
model.to(device)
# print(model)



loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
                                    else "gloo") # Changed in step 1
# model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler) # Changed in step 1
model = adaptdl.torch.AdaptiveDataParallel(model, optimizer)



# for idx, param_group in enumerate(optimizer.param_groups):
#     print(param_group)

# exit()


losses = []
def train(model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbsz:{}\tvar:{}\tsqr:{}\tgain:{}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loader.current_batch_size, model.gns.var_avg(), model.gns.sqr_avg(), model.gns.gain(10),loss.item()))

    losses.append(np.mean(epoch_loss))
    print(f"Epoch: {epoch}, loss: {np.mean(epoch_loss)}")


# epochs=1000
# for epoch in adaptdl.torch.remaining_epochs_until(epochs): # Changed in step 4
#     train(model, device, train_loader, optimizer, epoch, loss_function)
#     save_all_states()
    
#     # print(model.gain)
#     # exit()
#     # test(model, device, test_loader)
#     # scheduler.step()\



"""
test checkpoint
"""
for epoch in adaptdl.torch.remaining_epochs_until(2): # Changed in step 4
    train(model, device, train_loader, optimizer, epoch, loss_function)
    save_all_states(ckpt_dir="/workspace/FedScale/adapt/test/ckpt")
    break

train_loader = adaptdl.torch.AdaptiveDataLoader(dataset, batch_size=batch_size, drop_last=True)
train_loader.autoscale_batch_size(1028, local_bsz_bounds=(32, 128))

model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())
model.to(device)


loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
# adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
#                                     else "gloo") # Changed in step 1
# model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler) # Changed in step 1
model = adaptdl.torch.AdaptiveDataParallel(model, optimizer)

for epoch in adaptdl.torch.remaining_epochs_until(2): # Changed in step 4
    train(model, device, train_loader, optimizer, epoch, loss_function)
    save_all_states(ckpt_dir="/workspace/FedScale/adapt/test/ckpt")














# losses = []
# for epoch in range(1):
#     pred_y = model(data_x)
#     loss = loss_function(pred_y, data_y)
#     losses.append(loss.item())

#     model.zero_grad()
#     loss.backward()

#     for p in model.parameters():
#        print(p)
#        print(p.grad_sample[0].shape)
#        print(p.grad_sample)
#        exit()

#     optimizer.step()


import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.savefig('./loss.png')