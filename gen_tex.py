#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import json
import time
import random
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve

from options import *
from utils import *

def get_tex_data(dataroot, mask, pe_id):
    
    p_id = pe_id.split("_")[0]
    e_id = pe_id.split("_")[1]
    
    pos_map = read_npy(dataroot + "/pred/reg_map/pos_map_{}.npy".format(pe_id))
    norm_map = read_npy(dataroot + "/maps/eval/norm_map/norm_map_{}.npy".format(pe_id))
    
    with open(dataroot + "/select_dict.json", 'r') as f:
        select_dict = json.load(f)
    indices = select_dict[str(int(p_id))][str(int(e_id))]
    with open(dataroot + "/images/{}/{}/params.json".format(p_id, e_id), 'r') as f:
        cams_dict = json.load(f)
    
    cams = []
    images = []
    for index in indices:
        K = np.array(cams_dict[str(index) + "_K"])
        Rt = np.array(cams_dict[str(index) + "_Rt"])
        # give to cams
        cams.append(
            read_camera(K, Rt)
        )
        images.append(
            read_image(dataroot + "/images/{}/{}/{}.jpg".format(p_id, e_id, index))
        )
    
    data = {
        "pos_map": pos_map,
        "norm_map": norm_map,
        "images": images,
        "cams": cams  
    }
    
    return data

def gen_tex(opt, pos_map, norm_map, images, cams):
    """ generate texture from images with predicted pos_map """
    
    # 1. upsample pos_map
    map_upsample_scale = opt.tex_uv_size / opt.uv_size
    
    pos_map_updated_linear = cv2.resize(
        pos_map, None, fx=map_upsample_scale, fy=map_upsample_scale, interpolation=cv2.INTER_LINEAR)
    
    pos_map_updated_nearest = cv2.resize(
        pos_map, None, fx=map_upsample_scale, fy=map_upsample_scale, interpolation=cv2.INTER_NEAREST)
    
    pos_map_updated_nearest_norm = np.linalg.norm(pos_map_updated_nearest, axis=-1)
    pos_map_updated_linear_norm = np.linalg.norm(pos_map_updated_linear, axis=-1)
    thres = np.mean(np.abs(pos_map_updated_linear_norm - pos_map_updated_nearest_norm))
    nearest_index = np.where(np.abs(pos_map_updated_linear_norm - pos_map_updated_nearest_norm) > thres)
    pos_map_updated = pos_map_updated_linear
    pos_map_updated[nearest_index[0], nearest_index[1], :] = pos_map_updated_nearest[nearest_index[0], nearest_index[1], :]
    
    pos_map_mask = np.sum(np.abs(pos_map_updated), axis=-1)
    pos_map_mask = np.where(pos_map_mask==0, 0, 1)
    
    # TODO: use pos_map to get norm_map
    norm_map_updated = cv2.resize(
        norm_map, None, fx=map_upsample_scale, fy=map_upsample_scale, interpolation=cv2.INTER_LINEAR)
    kernel_size = 31
    conv_kernel = np.ones((kernel_size,kernel_size,1)) / kernel_size / kernel_size
    norm_map_updated_conv = convolve(norm_map_updated, conv_kernel)
    
    # 2. calculate weights and unwrap image
    weights = []
    textures = []
    for i, image in enumerate(images):
        cam = cams[i]
        # [3, 4]
        P = np.matmul(cam[1,:3,:], cam[0,:,:])
        # [H, W, 3]
        homo = np.matmul(
            P[np.newaxis, np.newaxis, :, :], 
            np.concatenate([pos_map_updated, np.ones(pos_map_updated.shape[:-1] + (1,))], axis=-1)[:,:,:,np.newaxis]
        ).squeeze(axis=-1)
        # 0->x->W 1->y->H
        xy = homo[:,:,0:2] / homo[:,:,2:3]
        # interpolation
        height = image.shape[0]
        width = image.shape[1]
        # normalize to range[-1, 1]
        x_normalized = xy[:,:,0] / ((width-1)/2) - 1
        y_normalized = xy[:,:,1] / ((height-1)/2) - 1
        # [H, W, 2]
        xy_normalized = np.stack([x_normalized, y_normalized], axis=-1)
        # [C, H, W]
        image = image.transpose(2, 0, 1)
        # to torch and add batch dim
        xy_normalized = torch.from_numpy(xy_normalized).unsqueeze(0).float()
        image = torch.from_numpy(image).unsqueeze(0).float()
        texture = F.grid_sample(image, xy_normalized, mode='bilinear', padding_mode='zeros')
        texture = texture.squeeze(0).numpy().transpose(1, 2, 0)
        textures.append(texture)
        
        # camera center
        center = np.array([0, 0, 0, 1])
        P_inv = np.linalg.inv(cam[0,:,:])
        center_world_coor = np.matmul(P_inv, center[:,np.newaxis])
        center_world_coor = (center_world_coor[:3] / center_world_coor[3:4]).squeeze(-1)
        ray_vector = center_world_coor[np.newaxis,np.newaxis,:] - pos_map_updated
        # use dot as a weight of the angle between norms and rays
        dot = np.sum(ray_vector * norm_map_updated_conv, axis=-1)
        dot = dot / np.linalg.norm(ray_vector, axis=-1)
        dot = np.where(dot>=0, dot, 0)
        # default occlusion solved by dot; out of sight solved by tex_filter
        tex_filter = np.where(np.sum(texture, axis=-1) > 0, 1, 0)
        weight = dot * tex_filter
        weights.append(weight) 
        
    textures = np.stack(textures, axis=0)
    weights = np.stack(weights, axis=0)
    
    # 3. blending
    weights_normalized = weights / np.sum(weights, axis=0)
    weights_normalized = np.where(np.isnan(weights_normalized)==True,
                                  1.0/len(images), weights_normalized)
    # sum([V, H, W, 3] * [V, H, W, 1], axis=0)
    texture_blended = np.sum(textures * weights_normalized[:,:,:,np.newaxis], axis=0)
    
    # add mask in case world coor (0,0) multiply norm_conv 
    texture_blended = texture_blended * pos_map_mask[:,:,np.newaxis]
    
    return texture_blended

def relocate(image):
    """ adjust image for dpmap prediction; only support 1024 tex"""
    image_crop = image[300-288:1250-288,550-512:1500-512,:]
    image_relocated = cv2.resize(image_crop, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    return image_relocated

opt = BaseOptions().parse()
with open(opt.reg_dataroot + "/id.json", 'r') as f:
    EVAL_ID_LIST = json.load(f)["eval"]

mask = cv2.imread(opt.reg_dataroot + '/facial_mask.png').astype(np.bool)[200-128:200+128, 256-128:256+128, 0]
#mask_raw = cv2.imread(opt.reg_dataroot + '/facial_mask_raw.png')
#mask_pred = cv2.resize(mask_raw, (2084, 2048), interpolation=cv2.INTER_NEAREST)[800-512:800+512, 1024-512:1024+512, 0].astype(np.bool)

for face_id in EVAL_ID_LIST:
    data = get_tex_data(opt.reg_dataroot, mask, face_id)
    tex = gen_tex(opt, data["pos_map"], data["norm_map"], data["images"], data["cams"])
    print(f"Generating {face_id} texture")
    #plt.figure(), plt.imshow(tex/255)
    #tex = tex * mask_pred[:,:,np.newaxis]
    cv2.imwrite(opt.reg_dataroot + f"/pred/texture/texture_{face_id}.jpg", tex)
    tex_relocated = relocate(tex)
    cv2.imwrite(opt.reg_dataroot + f"/pred/texture_relocated/texture_{face_id}.jpg", tex_relocated)

