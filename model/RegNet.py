import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(Conv3dBNReLU, self).__init__()

        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class SimpleRegNet(nn.Module):
    def __init__(self):
        super(SimpleRegNet, self).__init__()
        
        self.conv10 = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self.conv11 = nn.Conv3d(2, 2, kernel_size=3, padding=1)
        
        self.conv20 = nn.Conv3d(2, 4, kernel_size=3, padding=1)        
        self.conv21 = nn.Conv3d(4, 4, kernel_size=3, padding=1)
        
        self.conv30 = nn.Conv3d(4, 1, kernel_size=3, padding=1)
        
        #self.conv = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=2)
        self.loss_func = nn.MSELoss()        
        
    def forward(self, v, depth_value, mv_map, mask):
        B = depth_value.shape[0]
        D = depth_value.shape[1]
        H = mask.shape[1]
        W = mask.shape[2]
        
        v = v * mask.reshape([B,1,1,H,W])
        # v = v * 10
        # v [B 1 D H W]
        
        v = self.conv11(self.conv10(v))
        v = self.conv21(self.conv20(v))
        v = self.conv30(v)
        
        v = v.squeeze(1)
        
        # softmax [B D H W]
        prob = F.softmax(v, dim=1)
        # softargmax
        # to pred_map
        pred = prob * depth_value.reshape([B,D,1,1])
        pred = torch.sum(pred, dim=1)
        #print(pred.shape)
        pred = pred * mask
        
        loss = self.loss_func(pred, mv_map * mask)
        return {
            "new_volume": v,
            "pred" : pred,
            "loss" : loss
        }


class UnetRegNet(nn.Module):
    def __init__(self, opt, base_channel=1, mod="train"):
        super(UnetRegNet, self).__init__()
        
        self.opt = opt
        BC = base_channel
        self.conv0 = Conv3dBNReLU(1, BC)

        self.conv1 = Conv3dBNReLU(BC, BC*2, stride=2)
        self.conv2 = Conv3dBNReLU(BC*2, BC*2)

        self.conv3 = Conv3dBNReLU(BC*2, BC*4, stride=2)
        self.conv4 = Conv3dBNReLU(BC*4, BC*4)
        """
        self.conv5 = Conv3dBNReLU(BC*4, BC*8, stride=2)
        self.conv6 = Conv3dBNReLU(BC*8, BC*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(BC*8, BC*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(BC*4),
            nn.ReLU(inplace=True))
        """
        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(BC*4, BC*2, kernel_size=3, padding=1, output_padding=(0,1,1), stride=2, bias=False),
            nn.BatchNorm3d(BC*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(BC*2, BC, kernel_size=3, padding=1, output_padding=(0,1,1), stride=2, bias=False),
            nn.BatchNorm3d(BC),
            nn.ReLU(inplace=True))
        
        self.last_conv = nn.Conv3d(BC, 1, kernel_size=3, stride=1, padding=1)
        
        self.loss_func = nn.MSELoss()

        self.mod = mod
        if self.mod == "train":
            self.batch_size = self.opt.if_training_batch_size
        elif self.mod == "eval":
            self.batch_size = self.opt.if_evaling_batch_size
        else:
            raise Exception("Mod name {} mismatches 'train' or 'eval'".format(mod))     

    def forward(self, v, depth_value, mv_map, mask):
        B = depth_value.shape[0]
        D = depth_value.shape[1]
        H = mask.shape[1]
        W = mask.shape[2]

        v = v * mask.reshape([B,1,1,H,W])
        
        # v [B 1 D H W]
        # volume reg
        conv0 = self.conv0(v)
        #print("conv0", conv0.shape)
        conv2 = self.conv2(self.conv1(conv0))
        #print("conv2", conv2.shape)
        v = self.conv4(self.conv3(conv2))
        #print("conv4", v.shape)
        #x = self.conv6(self.conv5(conv4))
        #x = conv4 + self.conv7(x)
        #print("conv9", self.conv9(v).shape)
        v = conv2 + self.conv9(v)
        v = conv0 + self.conv11(v)
        v = self.last_conv(v)
        
        v = v.squeeze(1)

        # softmax [B D H W]
        prob = F.softmax(v, dim=1)
        # softargmax
        # to pred_map
        pred = prob * depth_value.reshape([B,D,1,1])
        pred = torch.sum(pred, dim=1)
        #print(pred.shape)
        pred = pred * mask 

        res = {
            "pred" : pred
        }

        if self.mod == "train":
            loss = self.loss_func(pred, mv_map * mask)
            res.update({
                "loss" : loss
            })

        return  res
