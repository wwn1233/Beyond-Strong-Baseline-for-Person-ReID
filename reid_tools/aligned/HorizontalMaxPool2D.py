import torch.nn as nn

class HorizontalMaxPool2d(nn.Module):
    def __init__(self, part = 8):
        super(HorizontalMaxPool2d, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((part, 1))


    def forward(self, x):
        return self.maxpool(x)
        # inp_size = x.size()

        # return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))

class HorizontalAvgPool2d(nn.Module):
    def __init__(self, part = 8):
        super(HorizontalAvgPool2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((part, 1))


    def forward(self, x):
        return self.avgpool(x)