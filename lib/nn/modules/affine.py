import torch
import torch.nn as nn
#from core.config import cfg

class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()
        #if cfg.BN_SETTING.SOFTMAX:
        #    self.softmax = nn.Softmax(0).cuda()


    def forward(self, x):
        #if cfg.BN_SETTING.SOFTMAX:
        #    weight = self.softmax(self.weight)
        #    return x * weight.view(1, self.num_features, 1, 1) + \
        #        self.bias.view(1, self.num_features, 1, 1)
        #else:
        return x * self.weight.view(1, self.num_features, 1, 1) + \
                   self.bias.view(1, self.num_features, 1, 1)
