#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.FSRegDataset import *
from model.RegNet import *
from options import *

import numpy as np
import json
import time
import random
import cv2
import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


# initialization
opt = BaseOptions(useJupyterNotebookArgs=[]).parse()
device = torch.device("cuda:%d"%(opt.cuda_id))
print(device, "is available")

# random seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

with open(opt.if_dataroot + "/id.json", 'r') as f:
    TRAIN_ID_LIST = json.load(f)["train"]
    
train_set = FSRegDataset(opt, id_list=TRAIN_ID_LIST, mod="train")
train_loader = DataLoader(train_set, batch_size=opt.reg_training_batch_size, num_workers=6, pin_memory=True,
                        shuffle=True)
model = UnetRegNet(opt, mod="train").to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.reg_learning_rate)

model.train()

global_step = 0
loss_list = []
start_epoch = 0
end_epoch = opt.reg_training_epoch


def train_sample(item):
    
    volume = item["volume"].to(device).float()
    mv_map = item["mv_map"].to(device).float()
    facial_mask = item["facial_mask"].to(device).float()
    depth_value = item["depth_value"].to(device).float()
    
    optimizer.zero_grad()
    res = model(volume, depth_value, mv_map, facial_mask)
    
    loss = res["loss"]
    loss.backward()
    optimizer.step()
    
    return res

for epoch in range(start_epoch, end_epoch):
    cnt = 0
    for item in train_loader:
        if global_step == 0:
            print("depth_value: ")
            print(item["depth_value"][0])
        cnt += 1
        start_time = time.time()
        res = train_sample(item)
        loss = res["loss"]
        location = item["location"]
        
        loss_list.append(loss.item())    
        global_step += 1
        end_time = time.time()
        print("Epoch %03d/%03d | %04d/%04d | Time: %.2fs | Loss: %.8f | Id: %s" 
                      %(epoch + start_epoch, end_epoch - start_epoch, cnt, train_loader.__len__(), \
                        end_time - start_time, loss, location))
        
    # save per 5 epochs
    if epoch % 5 == 4:
        checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dic": optimizer.state_dict(),
                        "loss_list": loss_list,
                        "epoch": epoch + 1,
                        "global_step" : global_step
                      }
        path_checkpoint = opt.reg_model_path + "/checkpoint_reg_{}_epoch.pkl".format(epoch + 1)
        torch.save(checkpoint, path_checkpoint)

