import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP Module
class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.filter_channels = filter_channels
        self.last_op = last_op

        for l in range(0, len(filter_channels) - 1):
            self.filters.append(nn.Conv1d(
                filter_channels[l],
                filter_channels[l + 1],
                1))
            self.add_module("conv%d" % l, self.filters[l])


    def forward(self, feature):
        '''
        :param feature [B, C_in, N]
        :return [B, C_out, N]
        '''
        y = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](y)
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
        
        if self.last_op:
            y = self.last_op(y)

        return y