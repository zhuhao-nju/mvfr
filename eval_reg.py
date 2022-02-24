#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets.FSRegDataset import *
from model.RegNet import *
from utils import *
from options import *

import os
import numpy as np
import json
import time
import random
import cv2

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

opt = BaseOptions(useJupyterNotebookArgs=[]).parse()
device = torch.device("cuda:%d"%(opt.cuda_id))
# dataset
with open(opt.reg_dataroot + "/id.json", 'r') as f:
    EVAL_ID_LIST = json.load(f)["eval"]
eval_set = FSRegDataset(opt, id_list=EVAL_ID_LIST, mod="eval")
eval_loader = DataLoader(eval_set, batch_size=opt.reg_evaling_batch_size, num_workers=6, pin_memory=True,
                        shuffle=False)

model = UnetRegNet(opt, mod="eval").to(device)

path_checkpoint = "./predefine_data/checkpoint_reg.pkl"
checkpoint = torch.load(path_checkpoint, map_location='cuda:0')
print("load pretrain model from {}".format(path_checkpoint))
model.load_state_dict(checkpoint["model_state_dict"])


def eval_reg_sample(item):
    volume = item["volume"].to(device).float()
    facial_mask = item["facial_mask"].to(device).float()
    depth_value = item["depth_value"].to(device).float()
    res = model(volume, depth_value, None, facial_mask)
    
    return res

model.eval()
for item in eval_loader:
    start_time = time.time()
    res = eval_reg_sample(item)
    facial_mask = item["facial_mask"]
    pos_map = item["pos_map"]
    norm_map = item["norm_map"]
    displacement_scale = item["displacement_scale"]
    location = item["location"]
    
    reg_pred_map = res["pred"].cpu().detach()
    
    # [B,H,W,3]
    reg_pos_map = pos_map + norm_map * reg_pred_map.unsqueeze(-1) * displacement_scale[:,None,None,None]
    reg_pos_map = reg_pos_map * facial_mask.unsqueeze(-1)
    for b in range(opt.reg_evaling_batch_size):
        save_npy(opt.reg_dataroot + "/pred/reg_map/pos_map_%s.npy"% (location[b]), reg_pos_map[b].numpy())
        pred_mesh = uv2mesh(reg_pos_map[b].numpy(), vt=True)
        pred_mesh.update_normals()
        om.write_mesh(opt.reg_dataroot + "/pred/reg_mesh/reg_%s.obj" % (location[b]), 
                      pred_mesh, 
                      vertex_tex_coord=True,
                      vertex_normal=True)
        print(location[b])

