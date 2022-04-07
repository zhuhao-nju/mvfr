#!/usr/bin/env python
# coding: utf-8

"""
undist 1-359 people and * 20 expressions images with select_dict.json
and save them to outpath
"""
import cv2
import numpy as np

import glob
import json
import matplotlib.pyplot as plt
import sys
import os
import time

from utils import *

# height = 1728; width = 2592 
#STANDARD_SHAPE = (1296, 864)
STANDARD_MAX_SHAPE = 1296
SAVE = True
PEOPLE_ID_LIST = [i for i in range(1, 359 + 1)]
EXPRESSION_NAMES_LIST = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left', \
                         '6_jaw_right', '7_jaw_forward', '8_mouth_left', '9_mouth_right', '10_dimpler', \
                         '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll', \
                         '16_grin', '17_cheek_blowing', '18_eye_closed', '19_brow_raiser', '20_brow_lower']
MAINPATH = '/media/xyz/RED31/mvfr_released/dev'
# TODO: change ori path here
MAINPATH_READ = '/media/xyz/RED31/share3/facescape_mview/fsmview_trainset/imgs'
MAINPATH_WRITE = MAINPATH + '/images'

# global
with open(MAINPATH + '/select_dict.json', 'r') as f:
    select_dict = json.load(f)
# record whether some images have been upsampled
upsample_list = []

def oneExpressionProcess(person_id, expression_name):
    
    read_path = MAINPATH_READ + "/{}/{}".format(person_id, expression_name)
    names = glob.glob(read_path + "/*.jpg")
    
    # sort
    #names.sort(key=lambda x:int(x.split("/")[-1].split(".")[0]))
    #print(names[:3])
    
    
    """
    view_ids = []
    for name in names:
        view_ids.append(int(name.split('\\')[-1].split(".")[0]))
    view_ids.sort()
    """
    expression_id = expression_name.split("_")[0]
    view_ids = select_dict[str(person_id)][expression_id]
    view_ids.sort()
    #print(view_ids)
    print("%d %s: read %d images" %(person_id, expression_name, len(view_ids)))
    
    
    with open(read_path + "/params.json") as f:
        data = json.load(f)
        
    cams, images, undist_images = [], [], []
    for view_id in view_ids:
        image = read_image(read_path + "/{}.jpg".format(view_id))
        images.append(image)
        K = np.array(data[str(view_id) + "_K"])
        Rt = np.array(data[str(view_id) + "_Rt"])
        dist = np.array(data[str(view_id) + "_distortion"])
        cams.append(
            read_camera(K, Rt)
        )
        if True:
            undist_images.append(
                cv2.undistort(image, K, dist)
            )
            
    new_images, new_cams = [], []
    
    for i, view_id in enumerate(view_ids):
        #image = images[i]
        image = undist_images[i]
        cam = cams[i]
        height = image.shape[0]
        width = image.shape[1]
        # check 
        if not ((height / width == 1.5) or (width / height == 1.5)):
            #raise Exception("Height and Width Dismatch")
            print("Height and Width Dismatch in {}_{}".format(person_id, expression_name))
            continue
        """
        # transpose
        if height < width:
            new_image = transpose_image(image)
            # transpose cam here simplily exchange line1 and line2 of K, and it has no influence on projection matrix
            new_cam = transpose_camera(cam)
            height = new_image.shape[0]
            width = new_image.shape[1]
            #print("%d.jpg transposes" %(view_id))
        else:
            new_image = image
            new_cam = cam
        # resize    
        #print(height)
        if height > STANDARD_SHAPE[0]:
            scale = STANDARD_SHAPE[0] / height
            new_image = resize_image(new_image, scale)
            new_cam = resize_camera(new_cam, scale)
        elif height < STANDARD_SHAPE[0]:
            # up_sample
            # nearly none
            print("%d %s: %s upsampling" %(person_id, expression_name, view_id))
            scale = STANDARD_SHAPE[0] / height
            new_image = resize_image(new_image, scale)
            new_cam = resize_camera(new_cam, scale)
            upsample_list.append("%03d_%s_%02d"%(person_id, expression_id, view_id))
        """
        max_shape = max(height, width)
        if max_shape >= STANDARD_MAX_SHAPE:
            scale = STANDARD_MAX_SHAPE / max_shape
            new_image = resize_image(image, scale)
            new_cam = resize_camera(cam, scale)
        elif max_shape < STANDARD_MAX_SHAPE:
            # up_sample
            # nearly none
            print("%d %s: %s upsampling" %(person_id, expression_name, view_id))
            scale = STANDARD_MAX_SHAPE / max_shape
            new_image = resize_image(image, scale)
            new_cam = resize_camera(cam, scale)
            upsample_list.append("%03d_%s_%02d"%(person_id, expression_id, view_id))
        
        new_images.append(new_image)
        new_cams.append(new_cam)
    
    
    save_path = MAINPATH_WRITE + "/%03d/%02d"%(person_id, int(expression_id))
    if SAVE:
        # create folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_data = {}
        for i, view_id in enumerate(view_ids):
            cv2.imwrite(save_path + "/{}.jpg".format(view_id), new_images[i])
            save_data[str(view_id) + "_K"] = new_cams[i][1,0:3,0:3].tolist()
            save_data[str(view_id) + "_Rt"] = new_cams[i][0,0:3,0:4].tolist()
            save_data[str(view_id) + "_height"] = new_images[i].shape[0]
            save_data[str(view_id) + "_width"] = new_images[i].shape[1]
        with open(save_path + "/params.json", "w") as f:
            json.dump(save_data, f)
        #print("Already save!")

for person_id in PEOPLE_ID_LIST:
    for expression_name in EXPRESSION_NAMES_LIST:
        start_time = time.time()
        oneExpressionProcess(person_id, expression_name)
        end_time = time.time()
        print("%d %s: Time cost: %.2fs" %(person_id, expression_name, end_time - start_time))
        #break
    #break
with open(MAINPATH + "/upsample_dict.json", "w") as f:
    json.dump({"upsample_list": upsample_list}, f)




