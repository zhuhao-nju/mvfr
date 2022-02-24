import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import *
from .ConvFilter import *
from .SurfaceClassifier import *

class ConvIFNet(nn.Module):
    '''
    ConvPIFu network uses conv network as the image filter.
    '''

    def __init__(self, opt, loss_func=nn.MSELoss(), mod="train"):
        super(ConvIFNet, self).__init__()

        self.name = "conv"
        
        self.opt = opt
        self.loss_func = loss_func
        self.view_num = self.opt.if_view_num
        self.image_filter = FeatureNet(norm=opt.norm)
        self.surface_classifier = SurfaceClassifier(
                filter_channels=self.opt.mlp_dim,
                last_op=nn.Sigmoid())
        self.mod = mod
        if self.mod == "train":
            self.batch_size = self.opt.if_training_batch_size
        elif self.mod == "eval":
            self.batch_size = self.opt.if_evaling_batch_size
        else:
            raise Exception("Mod name {} mismatches 'train' or 'eval'".format(mod))

        # save some result
        self.im_feat_list = []
        self.features = None
        
        self.preds = None
        self.labels = None
    
    def getFeatures(self, images):
        res = {}
        self.im_feat_list = self.image_filter(images)["outputs"]
        res.update({
            "im_feat_list":self.im_feat_list
        })
        return res
    
    def forward(self, 
        images, cams, points, movements, 
        labels, reuse_features=None
    ):
        '''
        :param images [BV, C, H, W]
        :param cams [BV, 2, 4, 4]
        :param movements [B, N] 
        :param points [B, 3, N] 3=>(H_i, W_i, Movement_in_ori_coordinate)
        '''
        # util shapes
        B = self.batch_size
        V = self.view_num
        N = points.shape[-1]

        # 1. images feature extraction
        # [BV, C', H', W']
        if reuse_features == None:
            self.im_feat_list = self.image_filter(images)["outputs"]
            #print("calculating features")
        else:
            self.im_feat_list = reuse_features["im_feat_list"]
        
        # 2. query
        # 2.1 perspective projection
        # [BV, 3, 4]
        KT = torch.matmul( cams[:,1,:3,:], cams[:,0,:,:] )
        rot = KT[:,:3,:3]
        trans = KT[:,:3,3:4]
        # [B, 3, N] => [B, V, 3, N] => [BV, 3, N]
        points = points.contiguous()
        """print(points.shape)"""
        points = points.unsqueeze(1).repeat(1, V, 1, 1).view(B*V, 3, N)
        # [BV, 3, 3] tensordot [BV, 3, N] => [BV, 3, N]
        homo = torch.einsum('bxy, byn -> bxn', rot, points) + trans
        """
        # BEE10 changed
        # [BV, 3, 4] tensordot [BV, 4, N] => [BV, 3, N]
        BV = points.shape[0]
        N = points.shape[-1]
        homo = torch.einsum('bxy, byn -> bxn', KT[:,:3,:],
                            torch.cat([points, torch.zeros((BV, 1, N), device=torch.device("cuda:0"))], dim=1))
        """
        """print("homo.shape:", homo.shape)"""
        # image coordinate [BV, 2, N]
        xy = homo[:,0:2,:] / homo[:,2:3,:]

        # 2.2 sample xy in feature
        height = self.im_feat_list[-1].shape[2]
        width = self.im_feat_list[-1].shape[3]
        # normalize to range[-1, 1]
        x_normalized = xy[:,0,:] / ((width-1)/2) - 1
        y_normalized = xy[:,1,:] / ((height-1)/2) - 1

        # [BV, N, 2]
        xy_normalized = torch.stack((x_normalized, y_normalized), dim=-1)
        xy_normalized = xy_normalized.contiguous()
        features = []
        for level, im_feat in enumerate(self.im_feat_list):
            
            # feat [BV, C, H', W']; index [BV, 1, N, 2] => [BV, C, 1, N] => [BV, C, N] 
            # V feat match V index. In fact, these V index are the same.
            feature = F.grid_sample(im_feat, xy_normalized.unsqueeze(1), 
                mode="bilinear", padding_mode="zeros").squeeze(2)
            features.append(feature)
        # [BV, Levels, C, N]
        features = torch.stack(features, dim=1)

        # 2.3 compute cost
        # some shape
        Levels = features.shape[1]
        C = features.shape[2]
        features = features.contiguous()
        features = features.view(B, V, Levels, C, N)
        # variance [B, L, C, N]
        # TODO add mask(may not be needed)
        variance = torch.sum(features ** 2, dim=1) / V - (torch.sum(features, dim=1) / V) ** 2

        loss = 0
        # 3. surface classifier
        for level in range(Levels):
            # stack movement [B, C+1, N] 
            vector = torch.cat([variance[:,level,:,:], movements.unsqueeze(1)], dim=1)
            #vector = variance[:,level,:,:]
            preds = self.surface_classifier(vector)
            # squeeze 1 (channel_num = 1)
            preds = preds.squeeze(1)
            
            if self.mod == "train":
                # 4. compute error
                loss += self.loss_func(preds, labels)

        res = {
            "preds": preds           
        }

        if self.mod == "train":
            loss = loss / Levels
            res.update({
                "loss": loss
            })

        return res



