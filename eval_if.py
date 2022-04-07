#!/usr/bin/env python
# coding: utf-8


import torch, json, time, random, cv2, tqdm
import numpy as np
import openmesh as om
from torch.utils.data import DataLoader, Dataset
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
"""
# random seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
"""
mod = "eval" # train or eval here
# dataset
with open(opt.if_dataroot + "/id.json", 'r') as f:
    EVAL_ID_LIST = json.load(f)[mod]
eval_set = FSIFDataset(opt=opt, id_list=EVAL_ID_LIST, mod="eval")
eval_loader = DataLoader(eval_set, batch_size=opt.if_evaling_batch_size, num_workers=4, shuffle=False)
model = ConvIFNet(opt=opt, loss_func=select_loss_func(opt.if_loss_func), mod="eval").to(device)
path_checkpoint = "./predefine_data/checkpoint_if.pkl"
checkpoint = torch.load(path_checkpoint, map_location='cuda:0')
print("load pretrain model from path: {}".format(path_checkpoint))
model.load_state_dict(checkpoint["model_state_dict"])

def eval_if_sample(item, num_sample=300000):
    """ eval one model for implicit function module """
    images = item["images"].to(device).float()
    cams = item["cams"].to(device).float()
    # [B, V, H, W, C] => [B, V, C, H, W]
    images = images.transpose(2,4)
    images = images.transpose(3,4)
    images, cams = reshape_multiview_tensors(images, cams)
    points = item["points"].to(device).float()
    displacements = item["displacements"].to(device).float()
    
    length = displacements.shape[1]
    times = length // num_sample + 1
    
    preds = torch.ones_like(displacements) * -1
    
    reuse_features = model.getFeatures(images)
    
    for time in range(times):
        index1 = 0 + time * num_sample
        index2 = (time + 1) * num_sample
        if time == times-1:
            res = model(images, cams, points[:,:,index1:], displacements[:,index1:],
                        None, reuse_features=reuse_features)
            preds[:,index1:] = res["preds"]
        else:
            res = model(images, cams, points[:,:,index1:index2], displacements[:,index1:index2],
                        None, reuse_features=reuse_features)
            preds[:,index1:index2] = res["preds"]

    return preds, res

model.eval()
with torch.no_grad():
    for count, item in enumerate(eval_loader):
        preds, res = eval_if_sample(item, num_sample=int(100000))
        # post process
        preds = preds.cpu()
        points_index = item["points_index"].long()
        facial_mask = item["facial_mask"]
        pos_map = item["pos_map"]
        norm_map = item["norm_map"]
        displacement_scale = item["displacement_scale"]
        
        location = item["progress"]
        
        volume = []
        for b in range(opt.if_evaling_batch_size):
            v = torch.zeros(opt.d_size, opt.uv_size, opt.uv_size)
            v[(points_index[b,0,:] + opt.d_size//2, points_index[b,1,:], points_index[b,2,:])] = preds[b]
            np.save(opt.if_dataroot + f"/volume/{mod}/ori_volume_%s.npy" % (location[b]), v.numpy().astype(np.float32))
            volume.append(v)
        volume = torch.stack(volume, dim=0)
        print(location)
        # optional
        # implicit function predicts map
        # if_pred_map = torch.argmax(volume, dim=1) - opt.d_size//2
        # if_pos_map = pos_map + norm_map * if_pred_map.unsqueeze(-1) * displacement_scale[:,None,None,None]
        # if_pos_map = if_pos_map * facial_mask.unsqueeze(-1)
        # for b in range(opt.if_evaling_batch_size):
            # pred_mesh = uv2mesh(if_pos_map[b].numpy(), vt=True)
            # om.write_mesh("./test_data/pred_%s_ori.obj" % (location[b]), 
            #               pred_mesh, 
            #               vertex_tex_coord=True)
