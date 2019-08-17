import torch
import torch.nn as nn
#from core.config import cfg

class Pool_wwn(nn.Module):
    """ A ... """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.weight = nn.Parameter(torch.Tensor(2,num_features))
        # self.bias = nn.Parameter(torch.Tensor(num_features))
        # self.weight.data.uniform_()
        nn.init.constant_(self.weight, 1.0)
        self.softmax_GL_TYPE = nn.Softmax(0)
        #self.bias.data.zero_()

        self.GL_TYPE_ATTE_w = None



    def forward(self, x1, x2):
        #if cfg.BN_SETTING.SOFTMAX:
        #    weight = self.softmax(self.weight)
        #    return x * weight.view(1, self.num_features, 1, 1) + \
        #        self.bias.view(1, self.num_features, 1, 1)
        self.GL_TYPE_ATTE_w = self.softmax_GL_TYPE(self.weight)
        # print(GL_TYPE_ATTE_w)
        # self.GL_TYPE_ATTE_w.cuda()

        x1 = self.GL_TYPE_ATTE_w[0, :].unsqueeze(0) * x1
        x2 = self.GL_TYPE_ATTE_w[1, :].unsqueeze(0) * x2
        return x1,x2,self.GL_TYPE_ATTE_w
