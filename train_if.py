#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np
import json
import time
import random
import cv2
import matplotlib.pyplot as plt

from utils import *
from model.ConvIFNet import *
from datasets.FSIFDataset import *
from options import *
from loss import select_loss_func

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# initialization
opt = BaseOptions().parse()
device = torch.device("cuda:%d"%(opt.cuda_id))
print(device, "is available")

# random seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

with open(opt.if_dataroot + "/id.json", 'r') as f:
    TRAIN_ID_LIST = json.load(f)["train"]
train_set = FSIFDataset(opt=opt, id_list=TRAIN_ID_LIST, mod="train")
train_loader = DataLoader(train_set, batch_size=opt.if_training_batch_size, num_workers=4, shuffle=True)
model = ConvIFNet(opt=opt, loss_func=select_loss_func(opt.if_loss_func), mod="train").to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.if_learning_rate)

model.train()

global_step = 0
loss_list = []
#validate_loss_list = []
start_epoch = 0
end_epoch = opt.if_training_epoch

def train_sample(item):
    images = item["images"].to(device).float()
    cams = item["cams"].to(device).float()
    # [B, V, H, W, C] => [B, V, C, H, W]
    images = images.transpose(2,4)
    images = images.transpose(3,4)
    images, cams = reshape_multiview_tensors(images, cams)

    points = item["points"].to(device).float()
    labels = item["labels"].to(device).float()
    displacements = item["displacements"].to(device).float()
    
    optimizer.zero_grad()
    res = model(images, cams, points, displacements,
                labels)
    #preds = res["preds"]
    loss = res["loss"]
    
    loss.backward()
    optimizer.step()
    
    return {"loss": loss}

for epoch in range(start_epoch, end_epoch):
    count = 0
    model.train()
    #lr_scheduler.step()
    for item in train_loader:
        start_time = time.time()
        res = train_sample(item)
        loss_list.append(res["loss"].item())

        location = item["progress"]
        end_time = time.time()

        count += 1
        global_step += 1
        print("Epoch %03d/%03d | %04d/%04d | Location: %s | Time: %.2fs | Loss: %.8f" 
                  %(epoch + start_epoch, end_epoch - start_epoch, count, train_loader.__len__(), location[0],
                    end_time - start_time, res["loss"].item()))
    # save per 5 
    if epoch % 5 == 4:
        checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dic": optimizer.state_dict(),
                        "loss_list": loss_list,
                        "epoch": epoch + 1,
                        "global_step" : global_step
                      }
        path_checkpoint = opt.if_model_path + "/checkpoint_if_{}_epoch.pkl".format(epoch + 1)
        torch.save(checkpoint, path_checkpoint)

