##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Weinong  Wang
## Based on the structure of pytorch-detectron
## Tencent, Youtu X-Lab
## Email: weinong.wang@hotmail.com
## Copyright (c) 2019
## Reference:
##     Weinong Wang et al. Orthogonal Center Learning with Subspace Masking for Person Re-Identification.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  torch.autograd.function import InplaceFunction

from core.config import cfg
import numpy as np
import pandas as pd
import nn as mynn
import copy
import utils.resnet_weights_helper as resnet_utils
# from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d, HorizontalAvgPool2d

logger = logging.getLogger(__name__)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Dropout_WWN(InplaceFunction):
    def __init__(self, p=0.5, train=False, inplace=False, noise =None):
        super(Dropout_WWN, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.train = train
        self.inplace = inplace
        self.noise = noise

    def _make_noise(self, input):
    # generate random signal
        return input.new().resize_as_(input)

    def forward(self, input):
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if self.p > 0 and self.train:
            if self.noise is None:
                self.noise = self._make_noise(input)
                # multiply mask to input
                self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
                if self.p == 1:
                    self.noise.fill_(0)
                self.noise = self.noise.expand_as(input)
                output.mul_(self.noise)
            else:
                self.noise = self.noise * 2
                output.mul_(self.noise)

        return output,self.noise/2

    def backward(self, grad_output, noise):
        if self.p > 0 and self.train:
            return grad_output.mul(self.noise)
        else:
            return grad_output


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


class Generalized_FPN(nn.Module):
    def __init__(self, num_classes,
                 loss={'softmax'},
                 aligned=False,
                 strong_baseline=True,
                 **kwargs):
        super().__init__()
        ## setting
        self.loss = loss
        self.strong_baseline = strong_baseline
        # self.FPN = False
        # self.FPN_type = FPN_type
        self.aligned = aligned
        # if cfg.REID.FPN:
        #     self.feat_dim = cfg.FPN.DIM
        # else:
        self.feat_dim = 2048

        self.feat_hidden_dim = 2048
        self.aligned_dim = 2048

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        # Backbone for feature extraction ##
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

        self._init_modules()
        ## whether to add strong baseline
        if self.strong_baseline:
            # self.bottleneck_sub1 = nn.Linear(self.feat_dim, self.feat_hidden_dim)
            self.bottleneck_sub2 = nn.Sequential(
                # nn.Linear(self.feat_dim, self.feat_hidden_dim),
                nn.BatchNorm1d(self.feat_hidden_dim),
                # nn.ReLU(),
                nn.LeakyReLU(0.1),
                # nn.Dropout(p=0.5)
            )
            self.bottleneck_sub2.apply(weights_init_kaiming)
            if cfg.REID.CLASSIFIER_NOBIAS:
                self.classifier = nn.Linear(self.feat_hidden_dim, num_classes, bias = False)
            else:
                self.classifier = nn.Linear(self.feat_hidden_dim, num_classes)
            self.classifier.apply(weights_init_classifier)

            if cfg.REID.CAMIDCLASS:
                # self.bottleneck_cam = nn.Sequential(
                #     nn.Linear(self.feat_dim, 128),
                #     nn.BatchNorm1d(128),
                #     nn.ReLU(),
                #     # nn.LeakyReLU(0.1),
                #     # nn.Dropout(p=0.5)
                # )
                # self.bottleneck_cam.apply(weights_init_kaiming)
                self.classifier_cam = nn.Linear(self.feat_dim, 8)
                self.classifier_cam.apply(weights_init_classifier)


        else:
            KeyError("Unsupported self.strong_baseline: {}".format(self.strong_baseline))

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
        if not cfg.REID.FPN:
            print('cfg.REID.FPN:{}'.format(cfg.REID.FPN))
            for p in self.Conv_Body.conv_top.parameters():
                p.requires_grad = False
            for p in self.Conv_Body.posthoc_modules.parameters():
                p.requires_grad = False
            for p in self.Conv_Body.topdown_lateral_modules.parameters():
                p.requires_grad = False
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.conv_body.parameters():
                p.requires_grad = False

        if cfg.REID.REGULARIZED_POOLING:
            # print('Not implemented!')
            self.res5_1 = copy.deepcopy(self.Conv_Body.conv_body.res5)

    def forward(self, data, label = None):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, label)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, label)

    def _forward(self, data, label = None):
        im_data = data

        # _, x = self.Conv_Body(im_data)
        # x = self.conv_adjust(x)
        if cfg.REID.REGULARIZED_POOLING:
            print('Not implemented!')
        else:
            x = self.Conv_Body(im_data)
            x = F.max_pool2d(x, x.size()[2:])

            f = x.view(x.size(0), -1)
            lf = None

            if not self.training:
                # f = x.view(x2.size(0), -1)
                return f, f ## the second f is useless and only for compatibility with the original test code,
            if self.strong_baseline:

                feat = self.bottleneck_sub2(f)
                noise = None

                y = self.classifier(feat)

                if cfg.REID.CAMIDCLASS:
                    # y_cam = self.bottleneck_cam(feat)
                    y_cam = self.classifier_cam(feat)
                else:
                    y_cam = None
            else:
                KeyError("Unsupported self.strong_baseline: {}".format(self.strong_baseline))

            if self.loss == 'softmax+metric':
                if self.aligned: raise KeyError("Unimplemented aligned")
                return y, feat, f, self.classifier.weight, noise,y_cam  #, lam, index
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron
