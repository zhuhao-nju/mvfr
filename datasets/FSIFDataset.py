from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import random
import matplotlib.pyplot as plt

from utils import *

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class FSIFDataset(Dataset):
    def __init__(self, opt, id_list, mod="train"):
        super(FSIFDataset).__init__()

        self.mainpath = opt.if_dataroot
        self.view_num = opt.if_view_num
        self.interval_scale = opt.interval_scale
        self.size = opt.uv_size
        self.d_size = opt.d_size
        self.half_d_size = self.d_size // 2
        
        self.sigma = opt.sigma
        self.points_num = opt.points_num

        self.image_downsample_scale = opt.if_image_downsample_scale
        self.feature_downsample_scale = opt.if_feature_downsample_scale

        if mod == "train" or mod == "eval":
            self.mod = mod
        else:
            raise Exception(f"Unknown mod {mod} for IF dataset")

        # dataset id like "001_01"
        self.id_list = id_list
        print("id list length: %d" %len(self.id_list))
        print("fsDataset: self.feature_downsample_scale: %f | self.image_downsample_scale: %f" 
              %(self.feature_downsample_scale, self.image_downsample_scale))
        
        # make gaussian table for query
        self.gaussian_table = gaussian(np.linspace(0, 2, self.d_size), 1, 0.05)


        # init fetch something
        self._init_fetch()

    def _init_fetch(self):
        with open(self.mainpath + "/Rt_scale_dict.json", 'r') as f:
            # scale location [][][0]
            self.Rt_scale_dict = json.load(f)
            
        with open(self.mainpath + "/select_dict.json", 'r') as f:
            self.select_dict = json.load(f)

    # generate points for evaling
    def _gen_points(self,
        pos_map, norm_map, displacement_scale, facial_mask
    ):
        # someshape
        D, H, W = (self.d_size, self.size, self.size)
        # 1. gen all point
        if D % 2 == 0:
            grid = np.mgrid[-(D//2):D//2, :H, :W]
        else:
            grid = np.mgrid[-(D//2):D//2+1, :H, :W]
        all_points_index = grid.reshape(3,-1)
        d = all_points_index[0,:]
        h = all_points_index[1,:]
        w = all_points_index[2,:]
        # 2. filter points in mask [DHW]
        points_mask = facial_mask[(h,w)]
        points_mask = np.where(points_mask==True)
        # [3, POINTS_NUM=>P] "d, h, w" order
        points_index = all_points_index[:,points_mask[0]]

        d = points_index[0,:]
        h = points_index[1,:]
        w = points_index[2,:]
        
        # 3. [3, P] in ori coordinate
        points_ori_coor = pos_map[(h,w)] + norm_map[(h,w)] * displacement_scale * (d[:,np.newaxis])
        points_ori_coor = points_ori_coor.T

        # [-1~1]
        displacements = d / self.d_size * 2

        return {
            "points": points_ori_coor,
            "points_index": points_index,
            "displacements": displacements
        }
    
    # sample points for training
    def _sample_points(self, 
        pos_map, norm_map, 
        mv_map, displacement_scale, 
        facial_mask):

        # 1. change mv_map to interval degree
        mv_map = mv_map / displacement_scale
        # facial mask
        mask_index = np.where(facial_mask>0)

        # 2. sample 
        # 2.1 surface based
        mask_index_length = mask_index[0].shape[0]
        # four times for more points
        #surface_sample_index = np.array(random.sample(range(mask_index_length), 4 * self.num_sample_inout))
        surface_sample_index = np.random.choice(range(mask_index_length), 4 * self.points_num, replace=True)
        h = mask_index[0][surface_sample_index]
        w = mask_index[1][surface_sample_index]
        # [4*N, 3] add some gaussian movement along normal => "h, w, d" order
        surface_points = np.stack([h, w, mv_map[(h, w)] + np.random.normal(loc=0, scale=self.sigma, size=surface_sample_index.shape[0])]
            , axis=1)
        # 2.2 uniform random based TODO: real uniform random(in 3 dim) 
        #random_sample_index = np.array(random.sample(range(mask_index_length), self.num_sample_inout // 4))
        random_sample_index = np.random.choice(range(mask_index_length), self.points_num // 4, replace=True)
        h2 = mask_index[0][random_sample_index]
        w2 = mask_index[1][random_sample_index]
        # [N//4, 3] => "h, w, d" order
        random_points = np.stack([h2, w2, np.random.rand(random_sample_index.shape[0]) * self.d_size + (-self.d_size/2)] 
            , axis=1)
        # concat
        sample_points = np.concatenate([surface_points, random_points], 0)
        np.random.shuffle(sample_points)

        # 3. judge inside or outside
        h3 = sample_points[:,0].astype(np.int32)
        w3 = sample_points[:,1].astype(np.int32)
        m3 = sample_points[:,2].astype(np.int32)
        # TODO  once np.where is enough
        # 
        inside_index = np.where(m3-mv_map[(h3, w3)]<0)[0]
        outside_index = np.where(m3-mv_map[(h3, w3)]>0)[0]
        inside_points = sample_points[inside_index]
        outside_points = sample_points[outside_index]
        nin = inside_index.shape[0]
        nout = outside_index.shape[0]
        """
        # PRINT
        print("nin %d; nout %d"%(nin, nout))
        """

        if (nin > self.points_num // 2) and (nout > self.points_num // 2):
            inside_points = inside_points[:self.points_num // 2]
            outside_points = outside_points[:self.points_num // 2]
        elif nin > nout:
            inside_points = inside_points
            outside_points = outside_points[:(self.points_num - nin)]
        else:
            inside_points = inside_points[:(self.points_num - nout)]
            outside_points = outside_points

        # [3, N]
        points = np.concatenate([inside_points, outside_points], 0).T
        #labels = np.concatenate([np.ones((inside_points.shape[0])), np.zeros((outside_points.shape[0]))], 0)

        # change to ori coordinate
        h4 = points[0,:].astype(np.int32)
        w4 = points[1,:].astype(np.int32)
        m4 = points[2,:]
        points = pos_map[(h4, w4)] + norm_map[(h4, w4)] * displacement_scale * m4[:,np.newaxis]
        # [3, N]
        points = points.T
        # normalize displacements # [-1~1]
        displacements = m4 / self.d_size * 2

        #save_sampled_points('../data/sampled_points_{}_{}.ply'.format(person_id, expression_id), points, labels)

        # make gaussian labels
        labels = np.zeros(len(points[0]))
        for i in range(len(points[0])):
            position = int(np.round(m4[i] - mv_map[h4[i], w4[i]]))
            if (position>=-self.half_d_size) and (position<=self.half_d_size):
                labels[i] = self.gaussian_table[position + self.half_d_size]
                #print(labels[i])

        return {
            "points": points,
            "labels": labels,
            "displacements": displacements
        }


    def __getitem__(self, idx):

        sample_id = self.id_list[idx]
        person_id = sample_id.split("_")[0]
        expression_id = sample_id.split("_")[1]

        # read
        norm_map = read_npy(
            self.mainpath + "/maps/{}/norm_map/norm_map_{}_{}.npy".format(self.mod, person_id, expression_id)
        )
        pos_map = read_npy(
            self.mainpath + "/maps/{}/pos_map/pos_map_{}_{}.npy".format(self.mod, person_id, expression_id)
        )
        if self.mod == "train":
            mv_map = read_npy(
                self.mainpath + "/maps/{}/mv_map/mv_map_{}_{}.npy".format(self.mod, person_id, expression_id)
            )
        facial_mask = read_image(
            self.mainpath + "/facial_mask.png"
        ).astype(np.bool)[200-128:200+128, 256-128:256+128, 0]
        with open(self.mainpath + "/images/{}/{}/params.json".format(person_id, expression_id), 'r') as f:
            cams_dict = json.load(f)
        
        indices = self.select_dict[str(int(person_id))][str(int(expression_id))][:self.view_num]
        Rt_scale = self.Rt_scale_dict[str(int(person_id))][str(int(expression_id))][0]
        displacement_scale = self.interval_scale / Rt_scale
        #print("images indices", indices)
        cams = []
        images = []
        for index in indices:
            image = read_image(
                self.mainpath + "/images/{}/{}/{}.jpg".format(person_id, expression_id, index)
            )
            h = image.shape[0]
            w = image.shape[1]
            if h > w:
                # transpose
                image = transpose_image(image)
            # image downsample
            image = resize_image(image, scale=self.image_downsample_scale)
            
            K = np.array(cams_dict[str(index) + "_K"])
            Rt = np.array(cams_dict[str(index) + "_Rt"])
            # give to cams
            cam = read_camera(K, Rt)
            if h > w:
                # transpose
                cam = transpose_camera(cam)
            # feature downsample
            cam = resize_camera(cam, scale=self.feature_downsample_scale)
            # image downsample
            cam = resize_camera(cam, scale=self.image_downsample_scale)
            
            cams.append(cam)
            images.append(image)
            
        
        # [V, 2, 4, 4]
        cams = np.stack(cams, axis=0)
        # [V, H, W, C]
        images = np.stack(images, axis=0)

        res = {
            "cams" : cams,
            "images" : images,
            "progress" : sample_id,
            "displacement_scale" : displacement_scale,
            "pos_map" : pos_map,
            "norm_map" : norm_map,
            "facial_mask": facial_mask
        }
        if self.mod == "train":
            res.update(
                self._sample_points( 
                    pos_map, norm_map, mv_map, displacement_scale, facial_mask
                )
            )
        else:
            res.update(
                self._gen_points(
                    pos_map, norm_map, displacement_scale, facial_mask
                )
            )
        
        return res

    def __len__(self):

        return len(self.id_list)