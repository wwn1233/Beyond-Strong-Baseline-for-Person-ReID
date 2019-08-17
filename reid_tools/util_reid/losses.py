##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## modified by: Weinong  Wang
## Tencent, Youtu X-Lab
## Email: weinong.wang@hotmail.com
## Copyright (c) 2019
## Reference:
##     Weinong Wang et al. Orthogonal Center Learning with Subspace Masking for Person Re-Identification.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from __future__ import absolute_import
from aligned.local_dist import *
import torch
from torch import nn
# from core.config import cfg
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
-  Orthogonal Center Learning with Subspace Masking: OCSM
"""
__all__ = ['DeepSupervision', 'CrossEntropyLoss','CrossEntropyLabelSmooth', \
           'TripletLoss', 'CenterLoss', 'OCSM', \
           'RingLoss','FocalLoss',\
           'ContrastiveLoss_wwn']

def global_orthogonal_regularization(anchor, negative):
    neg_dis = torch.sum(torch.mul(anchor, negative), 1)
    dim = anchor.size(1)
    gor = (torch.pow(torch.mean(neg_dis), 2) + torch.clamp(torch.mean(torch.pow(neg_dis, 2)) - 1.0 / dim, min=0.0))/(anchor.size(0) * dim)

    return gor

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

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss

class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    """
    def __init__(self, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        self.use_gpu = use_gpu
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu: targets = targets.cuda()
        loss = self.crossentropy_loss(inputs, targets)
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss,0

class TripletLossAlignedReID(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False, aligned = False):
        super(TripletLossAlignedReID, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag
        self.aligned = aligned

    def forward(self, inputs, targets, local_features, epoch=None):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.aligned:
            local_features = local_features.permute(0,2,1)
            p_local_features = local_features[p_inds]
            n_local_features = local_features[n_inds]
            local_dist_ap = batch_local_dist(local_features, p_local_features)
            local_dist_an = batch_local_dist(local_features, n_local_features)
            local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap, y)
        else:
            local_loss = 0

        # gor_loss = global_orthogonal_regularization(inputs, inputs[n_inds])
        # global_loss += 0.0001 * gor_loss

        if self.mutual:
            return global_loss+local_loss,dist
        return global_loss,local_loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, noise =None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        distmat = distmat / self.feat_dim

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class OCSM(nn.Module):
    """

    Reference:
       Weinong Wang et al. Orthogonal Center Learning with Subspace Masking for Person Re-Identification.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):  # , margin = 0.3):
        super(OCSM, self).__init__()


    def forward(self, x, labels, center_weight, noise):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
            center_weight: the shared weight with softmax loss
        """
        print("Not implemented!")

        loss_all = 0
        return loss_all

class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return l * self.weight_ring

class KLMutualLoss(nn.Module):
    def __init__(self):
        super(KLMutualLoss,self).__init__()
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.log_softmax = nn.functional.log_softmax
        self.softmax = nn.functional.softmax
    def forward(self, pred1, pred2):
        pred1 = self.log_softmax(pred1, dim=1)
        pred2 = self.softmax(pred2, dim=1)
        #loss = self.kl_loss(pred1, torch.autograd.Variable(pred2.data))
        loss = self.kl_loss(pred1, pred2.detach())
        # from IPython import embed
        # embed()
        #print(loss)
        return loss

class MetricMutualLoss(nn.Module):
    def __init__(self):
        super(MetricMutualLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, dist1, dist2,pids):
        loss = self.l2_loss(dist1, dist2)
        # from IPython import embed
        # embed()
        print(loss)
        return loss

class FocalLoss(nn.Module):
    """
    FocalLoss.
    Reference:
       Focal Loss for Dense Object Detection, ICCV 2017
    """
    def __init__(self, gamma=2, alpha=0.25, size_average=True,labelsmooth = False, num_classes = None, epsilon = 0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # if isinstance(alpha, Variable):
        self.alpha = Variable(alpha * torch.ones(num_classes, 1))
        # else:
        #     self.alpha = Variable(alpha)
        self.size_average = size_average
        self.class_num = num_classes
        self.labelsmooth = labelsmooth
        if self.labelsmooth:
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, target):
        if self.labelsmooth:
            log_probs = self.logsoftmax(inputs)
            pt = Variable(log_probs.data.exp())
            target = torch.zeros(log_probs.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
            target = target.cuda()
            target = (1 - self.epsilon) * target + self.epsilon / self.class_num
            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            loss = (- self.alpha.squeeze().unsqueeze(0) * target * (1 - pt) ** self.gamma * log_probs).mean(0).sum()
            return loss

        else:
            N = inputs.size(0)
            C = inputs.size(1)
            P = F.softmax(inputs)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = target.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return loss

# Custom Contrastive Loss
class ContrastiveLoss_wwn(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=5):
        super(ContrastiveLoss_wwn, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):

        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss_contrastive

if __name__ == '__main__':
    pass
