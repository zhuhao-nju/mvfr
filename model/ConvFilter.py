import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from .net_utils import *

class FeatureNet(nn.Module):
    """
    Simple net with some conv + norm + relu
    """
    def __init__(self, group_channels=32, norm="batch"):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        
        self.conv2 = ConvNReLU(32, 32, 3, 1, 1, group_channels=group_channels, norm=norm)
        self.conv3 = ConvNReLU(32, 32, 3, 1, 1, group_channels=group_channels, norm=norm)

        self.conv4 = ConvNReLU(32, 64, 3, 1, 1, group_channels=group_channels, norm=norm)
        self.conv5 = ConvNReLU(64, 64, 3, 1, 1, group_channels=group_channels, norm=norm)

        self.conv6 = ConvNReLU(64, 64, 3, 1, 1, group_channels=group_channels, norm=norm)
        self.conv7 = ConvNReLU(64, 64, 3, 1, 1, group_channels=group_channels, norm=norm)
        """
        self.conv8 = ConvNReLU(64, 128, 3, 1, 1, group_channels=group_channels, norm=norm)
        self.conv9 = ConvNReLU(128, 128, 3, 1, 1, group_channels=group_channels, norm=norm)
        
        self.conv10 = ConvNReLU(128, 256, 3, 1, 1, group_channels=group_channels, norm=norm)
        self.conv11 = ConvNReLU(256, 256, 3, 1, 1, group_channels=group_channels, norm=norm)
        """
        """
        self.feature = nn.Conv2d(256, 256, 3, 1, 1)
        """
        self.feature = nn.Conv2d(64, 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv3(self.conv2(x))
        x = self.conv5(self.conv4(x))
        x = self.conv7(self.conv6(x))
        #x = self.conv9(self.conv8(x))
        #x = self.conv11(self.conv10(x))
        x = self.feature(x)
        return {"outputs":[x]}
