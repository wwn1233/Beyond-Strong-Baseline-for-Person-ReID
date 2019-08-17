import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from utils.resnet_weights_helper import convert_state_dict

from dropblock.dropblock import DropBlock2D
from dropblock.scheduler import LinearScheduler

from utils.sparse_switchable_norm import SSN2d

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #
def ResNet18_conv5_body_prune(cfg_prune):   # added by weinong wang, *******************************************3
    return ResNet_conv18_body_prune((2, 2, 2, 2),cfg_prune=cfg_prune)

def ResNet18_conv5_body():
    return ResNet_conv18_body((2, 2, 2, 2))

def ResNet34_conv5_body():
    return ResNet_conv18_body((3, 4, 6, 3))
    
def ResNet50_conv4_body():
    return ResNet_convX_body((3, 4, 6))


def ResNet50_conv5_body():
    return ResNet_convX_body((3, 4, 6, 3))


def ResNet101_conv4_body():
    return ResNet_convX_body((3, 4, 23))


def ResNet101_conv5_body():
    return ResNet_convX_body((3, 4, 23, 3))


def ResNet152_conv5_body():
    return ResNet_convX_body((3, 8, 36, 3))


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


class ResNet_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2)
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            # self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
            #                               cfg.RESNETS.RES5_DILATION, stride_init)
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                          1, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

        if cfg.RESNETS.DROPBLOCK:
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=cfg.RESNETS.DB_DROPPROB, block_size=cfg.RESNETS.DB_BLOCKSIZE),
                start_value=0.,
                stop_value=cfg.RESNETS.DB_DROPPROB,
                nr_steps=cfg.RESNETS.DB_LINEARSTEP
            )

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        if cfg.RESNETS.FREEZE_AT == 0:
            Warning('*******Not freeze Bn layer, cfg.RESNETS.FREEZE_AT: {}'.format(cfg.RESNETS.FREEZE_AT))
        else:
            self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x =  getattr(self, 'res%d' % (i + 1))(x)
        return x

## For resnet18
class ResNet_conv18_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 64, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 128, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 256, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2)
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 512, dim_bottleneck * 8, block_counts[3],
                                          cfg.RESNETS.RES5_DILATION, stride_init)
            # self.res5, dim_in = add_stage(dim_in, 512, dim_bottleneck * 8, block_counts[3],
            #                               1, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        if cfg.RESNETS.FREEZE_AT == 0:
            Warning('*******Not freeze Bn layer, cfg.RESNETS.FREEZE_AT: {}'.format(cfg.RESNETS.FREEZE_AT))
        else:
            self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x =  getattr(self, 'res%d' % (i + 1))(x)
        return x

## For resnet18 slimming
class ResNet_conv18_body_prune(nn.Module):
    def __init__(self, block_counts, cfg_prune = None): # added by weinong wang, *******************************************3
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck_ori = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        if cfg_prune is None: # added by Weinong Wang  ***********************************************************1
           ValueError('cfg_prune should not be None!!')
        else:
            dim_bottleneck = 0
        self.res2, dim_in = add_stage_prune(dim_in, 64, dim_bottleneck_ori,cfg_prune[dim_bottleneck:dim_bottleneck+2], block_counts[0],
                                      dilation=1, stride_init=1)
        dim_bottleneck += 2
        self.res3, dim_in = add_stage_prune(dim_in, 128, dim_bottleneck_ori*2, cfg_prune[dim_bottleneck:dim_bottleneck+2], block_counts[1],
                                      dilation=1, stride_init=2)
        dim_bottleneck += 2
        self.res4, dim_in = add_stage_prune(dim_in, 256, dim_bottleneck_ori*4, cfg_prune[dim_bottleneck:dim_bottleneck+2], block_counts[2],
                                      dilation=1, stride_init=2)
        dim_bottleneck += 2
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage_prune(dim_in, 512, dim_bottleneck_ori*8, cfg_prune[dim_bottleneck:dim_bottleneck+2], block_counts[3],
                                          cfg.RESNETS.RES5_DILATION, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = int(cfg_prune[-1])

        self._init_modules()

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        if cfg.RESNETS.FREEZE_AT == 0:
            Warning('*******Not freeze Bn layer, cfg.RESNETS.FREEZE_AT: {}'.format(cfg.RESNETS.FREEZE_AT))
        else:
            self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x =  getattr(self, 'res%d' % (i + 1))(x)
        return x



class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
                                            dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
          residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes

def add_stage_prune(inplanes, outplanes, innerplanes, cfg_prune, nblocks, dilation=1, stride_init=2):# added by weinong wang, *******************************************3
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for i in range(nblocks):
        res_blocks.append(add_residual_block_prune(
            inplanes, outplanes, innerplanes, dilation, stride,cfg_prune[i]
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride,
        dilation=dilation, group=cfg.RESNETS.NUM_GROUPS,
        downsample=downsample)

    return res_block

def add_residual_block_prune(inplanes, outplanes, innerplanes, dilation, stride,cfg_prune): #added by weinongwang ***************************3
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride,
        dilation=dilation, group=cfg.RESNETS.NUM_GROUPS,
        downsample=downsample, cfg_prune= cfg_prune)

    return res_block

# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    if cfg.RESNETS.SSN:
        return nn.Sequential(
            nn.Conv2d(inplanes,
                      outplanes,
                      kernel_size=1,
                      stride=stride,
                      bias=False),
            SSN2d(outplanes, using_moving_average=True), )
    else:
        if cfg.REID.BN_COMPLETE:
            return nn.Sequential(
                nn.Conv2d(inplanes,
                          outplanes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(outplanes),)
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes,
                          outplanes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                mynn.AffineChannel2d(outplanes),
        )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                     eps=cfg.GROUP_NORM.EPSILON)
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem():
    if cfg.RESNETS.SSN:
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
            ('bn1', SSN2d(64, using_moving_average=True)),
            ('relu', nn.ReLU(inplace=True)),
            # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
    else:
        if cfg.REID.BN_COMPLETE:
            return nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu', nn.ReLU(inplace=True)),
                # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        else:
            return nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
                ('bn1', mynn.AffineChannel2d(64)),
                ('relu', nn.ReLU(inplace=True)),
                # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(64), 64,
                             eps=cfg.GROUP_NORM.EPSILON)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        if cfg.RESNETS.SSN:
            self.bn1 = SSN2d(innerplanes, using_moving_average=True)
        else:
            if cfg.REID.BN_COMPLETE:
                self.bn1 = nn.BatchNorm2d(innerplanes)
            else:
                self.bn1 = mynn.AffineChannel2d(innerplanes)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        if cfg.RESNETS.SSN:
            self.bn2 = SSN2d(innerplanes, using_moving_average=True)
        else:
            if cfg.REID.BN_COMPLETE:
                self.bn2 = nn.BatchNorm2d(innerplanes)
            else:
                self.bn2 = mynn.AffineChannel2d(innerplanes)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        if cfg.RESNETS.SSN:
            self.bn3 = SSN2d(outplanes, using_moving_average=True, last_gamma=True)
        else:
            if cfg.REID.BN_COMPLETE:
                self.bn3 = nn.BatchNorm2d(outplanes)
            else:
                self.bn3 = mynn.AffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if cfg.REID.ZERO_GAMMA:
            self._init_weights()

    def _init_weights(self):
        init.constant_(self.bn3.weight, 0.0)
        init.constant_(self.bn3.bias, 0.0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class basicblock_transformation(nn.Module):
    """ Basicblock Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes,  stride=1, dilation=1, group=1,
                 downsample=None, cfg_prune=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride
        if cfg_prune is None:
            self.conv1 = nn.Conv2d(
                inplanes, innerplanes, kernel_size=3, stride=str1x1, padding=1,bias=False)
            if cfg.REID.BN_COMPLETE:
                self.bn1 = nn.BatchNorm2d(innerplanes)
            else:
                self.bn1 = mynn.AffineChannel2d(innerplanes)

            self.conv2 = nn.Conv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
                padding=1, dilation=dilation, groups=group)
            if cfg.REID.BN_COMPLETE:
                self.bn2 = nn.BatchNorm2d(innerplanes)
            else:
                self.bn2 = mynn.AffineChannel2d(innerplanes)
        else:
            self.conv1 = nn.Conv2d(
                inplanes, cfg_prune, kernel_size=3, stride=str1x1, padding=1, bias=False)
            self.bn1 = mynn.AffineChannel2d(cfg_prune)

            self.conv2 = nn.Conv2d(
                cfg_prune, innerplanes, kernel_size=3, stride=str3x3, bias=False,
                padding=1, dilation=dilation, groups=group)
            self.bn2 = mynn.AffineChannel2d(innerplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.RESNETS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        if cfg.RESNETS.TRANS_FUNC == 'bottleneck_transformation':
            for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
                dtt_bp = detectron_prefix + '_branch2' + c
                mapping_to_detectron[my_prefix
                                     + '.conv%d.weight' % i] = dtt_bp + '_w'
                orphan_in_detectron.append(dtt_bp + '_b')
                mapping_to_detectron[my_prefix
                                     + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
                mapping_to_detectron[my_prefix
                                     + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
        elif cfg.RESNETS.TRANS_FUNC == 'basicblock_transformation':
            for i, c in zip([1, 2], ['a', 'b']):
                dtt_bp = detectron_prefix + '_branch2' + c
                mapping_to_detectron[my_prefix
                                     + '.conv%d.weight' % i] = dtt_bp + '_w'
                orphan_in_detectron.append(dtt_bp + '_b')
                mapping_to_detectron[my_prefix
                                     + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
                mapping_to_detectron[my_prefix
                                     + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
        else:
            ValueError('Not proper cfg.RESNETS.TRANS_FUNC!')


    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
