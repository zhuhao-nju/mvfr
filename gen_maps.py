#!/usr/bin/env python
# coding: utf-8

import os, json, camera, tqdm, sys, numpy as np
from fitter import fitter
from detector import detector
from uv_rasterizer import uv_rasterizer
from options import *

expressions_name_list = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left', \
                         '6_jaw_right', '7_jaw_forward', '8_mouth_left', '9_mouth_right', '10_dimpler', \
                         '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll', \
                         '16_grin', '17_cheek_blowing', '18_eye_closed', '19_brow_raiser', '20_brow_lower']

# get options
opt = BaseOptions().parse()
mod = "eval" # "train" or "eval" here.
# dataset
with open(opt.if_dataroot + "/id.json", 'r') as f:
    ALL_ID_LIST = json.load(f)
ID_LIST = ALL_ID_LIST[mod]
class dataPathController:
    """ Controlling Data Paths of gen_map """
    def __init__(self, dataroot):
        
        self.dataroot = dataroot
        
        with open(self.dataroot + "/select_dict.json", 'r') as f:
            self.select_dict = json.load(f)
        
    def get_detector_paths(self, face_id):
        """ from processed images """
        p_id, e_id = face_id.split('_')
        img_dir = self.dataroot + '/images/%s/%s'%(p_id, e_id)
        # TODO: remind to change it back
        img_dir = "/media/xyz/RED31/mvfr_released/dev" + '/images/%s/%s'%(p_id, e_id)
        idx_list = self.select_dict[str(int(p_id))][str(int(e_id))]
        img_format = [".jpg"] * len(idx_list)
        
        return img_dir, idx_list, img_format
    
    def get_raw_detector_paths(self, face_id):
        """ from facescape images """
        p_id, e_id = face_id.split('_')
        # TODO: change dir
        #img_dir = '/media/xyz/RED31/share3/facescape_mview/fsmview_trainset/imgs/%d/%s'%(int(p_id), expressions_name_list[int(e_id)-1])
        img_dir = opt.facescape_dataroot + '/imgs/%d/%s'%(int(p_id), expressions_name_list[int(e_id)-1])
        idx_list = self.select_dict[str(int(p_id))][str(int(e_id))]
        img_format = [".jpg"] * len(idx_list)
        
        return img_dir, idx_list, img_format
    

def gen_map(face_id, dataPathController):
    
    p_id, e_id = face_id.split('_')
    # ==================== fit base mesh ====================
    fit_model = fitter(fn_params = "./predefine_data/factor_847_50_52.json",
                       fn_core = "./predefine_data/core_847_50_52.npy")
    img_dir, idx_list, img_format = dataPathController.get_detector_paths(face_id)
    kp_detector = detector(fn_params = img_dir + "/params.json",
                           idx_list = idx_list,
                           img_format = img_format,
                           img_dir = img_dir,
                           debug = False)
    kp3d_list = kp_detector.solve_3d()
    
    verts, rot_matrix, trans, scale = fit_model.fit(kp3d_list)
    faces = fit_model.get_faces()
    
    fit_mesh_dir = opt.fit_dataroot + "/fit_mesh"
    # save out
    with open(fit_mesh_dir + f'/fit_mesh_{face_id}.obj', 'w') as f:
        for vert in verts:
            f.write("v %.6f %.6f %.6f\n" % (vert[0], vert[1], vert[2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0], face[1], face[2]))
    
    # ==================== generate maps ====================
    uvr = uv_rasterizer(uv_size=512, ref_mesh_dirname = "./predefine_data/1_neutral.obj")
    # TODO: gt_dir in dataset
    #gt_path = '/media/xyz/RED31/share3/facescape_mview/fsmview_trainset/mesh/%d/%s.ply'%(int(p_id), expressions_name_list[int(e_id)-1])
    gt_path = opt.facescape_dataroot + '/mesh/%d/%s.ply'%(int(p_id), expressions_name_list[int(e_id)-1])
    scale_tu = Rt_scale_dict["%d"%(int(p_id))]["%d"%(int(e_id))][0]
    max_dist_tu = 50 # 5mm
    this_max_dist = max_dist_tu / scale_tu
    mv_map, pos_map, norm_map = uvr.compute_mvmap(gt_path,
                                                  fit_mesh_dir + f"/fit_mesh_{face_id}.obj",
                                                  max_dist = this_max_dist,
                                                  smooth_norm_kernel = 21)
    
    return mv_map, pos_map, norm_map


with open(opt.fit_dataroot + "/Rt_scale_dict.json", 'r') as f:
    Rt_scale_dict = json.load(f)
dpController = dataPathController(opt.fit_dataroot)
for face_id in ID_LIST:
    p_id, e_id = face_id.split('_')
    mv_map, pos_map, norm_map = gen_map(face_id, dpController)
    np.save(opt.fit_dataroot + f"/maps/{mod}/mv_map/mv_map_{face_id}.npy", mv_map[200-128:200+128, 256-128:256+128])
    np.save(opt.fit_dataroot + f"/maps/{mod}/norm_map/norm_map_{face_id}.npy", norm_map[200-128:200+128, 256-128:256+128, :])
    np.save(opt.fit_dataroot + f"/maps/{mod}/pos_map/pos_map_{face_id}.npy", pos_map[200-128:200+128, 256-128:256+128, :])
    print(face_id, "maps have been generated")
