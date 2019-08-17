import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

import cv2
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, input, target, weight):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = self.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if weight is not None:
            loss = loss * weight

        return loss.mean()

class dice_focal_loss(nn.Module):
    def __init__(self, batch=True,b_weight =1.0,c_weight =10.0,c_weight_gamma=2.0):
        super(dice_focal_loss, self).__init__()
        self.batch = batch
        # self.a_weight = a_weight
        self.b_weight = b_weight
        self.c_weight = c_weight
        #self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(c_weight_gamma)

    def soft_dice_coeff(self, y_true, y_pred,weight):
        y_pred = F.sigmoid(y_pred)
        if weight is not None:
            y_true = y_true * weight
            y_pred = y_pred * weight
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score

    def soft_dice_loss(self, y_true, y_pred,weight):
        loss = -torch.log(self.soft_dice_coeff(y_true, y_pred,weight))
        return loss

    def __call__(self, y_true, y_pred, weight):
        #a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred,weight)
        c = self.focal_loss(y_pred,y_true,weight)
        loss_all = self.b_weight*b + self.c_weight*c
        return  loss_all