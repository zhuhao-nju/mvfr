import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import json
import cv2

from utils import *

class FSRegDataset(Dataset):
    def __init__(self, opt, id_list, mod="train"):
        super(FSRegDataset).__init__()
        self.dataroot = opt.reg_dataroot
        self.id_list = id_list
        with open(self.dataroot + "/Rt_scale_dict.json", 'r') as f:
            self.Rt_scale_dict = json.load(f)
        self.interval_scale = opt.interval_scale
        
        facial_mask = cv2.imread(self.dataroot + "/facial_mask.png")[:,:,0].astype(np.bool)
        self.facial_mask = facial_mask[200-128:200+128, 256-128:256+128]

        self.d_size = opt.d_size
        if mod == "train" or mod == "eval":
            self.mod = mod
        else:
            raise Exception(f"Unknown mod {mod} for Reg dataset")
        
        

    def __getitem__(self, idx):
        sample_id = self.id_list[idx]
        person_id = sample_id.split("_")[0]
        expression_id = sample_id.split("_")[1]
        
        volume = read_npy(self.dataroot + "/volume/{}/ori_volume_{}.npy".format(self.mod, sample_id))
        norm_map = read_npy(self.dataroot + "/maps/{}/norm_map/norm_map_{}.npy".format(self.mod, sample_id))
        pos_map = read_npy(self.dataroot + "/maps/{}/pos_map/pos_map_{}.npy".format(self.mod, sample_id))
        if self.mod == "train":
            mv_map = read_npy(
                self.dataroot + "/maps/{}/mv_map/mv_map_{}.npy".format(self.mod, sample_id)
            )
        Rt_scale = self.Rt_scale_dict[str(int(person_id))][str(int(expression_id))][0]
        displacement_scale = self.interval_scale / Rt_scale
        
        D = self.d_size
        if D % 2 == 1:
            depth_value = np.linspace(-(D//2), D//2, D)
        else:
            depth_value = np.linspace(-D//2, D//2-1, D)
        
        res = {
            "pos_map": pos_map,
            "norm_map": norm_map,
            "volume": np.expand_dims(volume, 0), 
            "displacement_scale": displacement_scale, 
            "depth_value": depth_value,
            "facial_mask": self.facial_mask, 
            "location": sample_id
        }

        if self.mod == "train":
            res.update({
                "mv_map": mv_map
            })

        return res

    
    def __len__(self):
        return len(self.id_list)    
