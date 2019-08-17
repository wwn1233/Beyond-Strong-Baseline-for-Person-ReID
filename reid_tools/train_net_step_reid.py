##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Beyond strong basleine for Person Re-ID. 2019
## Created by: Weinong  Wang
## Based on the structure of pytorch-detectron and Alignedreid
## Tencent, Youtu X-Lab
## Email: weinong.wang@hotmail.com
## Copyright (c) 2019
## Reference:
##     Weinong Wang et al. Orthogonal Center Learning with Subspace Masking for Person Re-Identification.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import random
from torchcontrib.optim.swa import SWA

##added by WEINONG
# from tensorboardX import SummaryWriter

#import models_reid
from util_reid.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID, TripletLoss, FocalLoss,ContrastiveLoss_wwn, CenterLoss,OCSM
from util_reid import data_manager
# from util import transforms as T
from util_reid.dataset_loader import ImageDataset
from util_reid.utils import Logger
from util_reid.utils import AverageMeter, Logger, save_checkpoint, save_checkpoint_best
from util_reid.eval_metrics import evaluate
from util_reid.optimizers import init_optim
from util_reid.samplers import RandomIdentitySampler
from util_reid.transforms import TrainTransform, TestTransform
from IPython import embed

import _init_paths  # pylint: disable=unused-import
import utils.net as net_utils
from modeling.model_builder_reid import Generalized_FPN
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
import nn as mynn

from core_ssn import *
from utils.sparse_switchable_norm import SSN2d

def parse_args():

    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train ReID with cross entropy loss and triplet hard loss')
    # Datasets
    parser.add_argument('-j', '--workers', default=10, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--root', type=str, default='data', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=data_manager.get_names())
    parser.add_argument('--split-id', type=int, default=0, help="split index")
    # CUHK03-specific setting
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help="whether to use labeled images, if false, detected images are used (default: False)")
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help="whether to use classic split by Li et al. CVPR'14 (default: False)")
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help="whether to use cuhk03-metric (default: False)")
    # Optimization options

    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")

    parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
    parser.add_argument('--seed', type=int, default=100, help="manual seed")

    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    ## Architecture
    parser.add_argument('--eval-step', type=int, default=40,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
    parser.add_argument('--save-dir', type=str, default='log')

    parser.add_argument('--num_gpu', default=1, type=int, help='')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='')
    ## detectron
    parser.add_argument(
        '--cfg', dest='cfg_file', required=False, default='configs/reid/e2e_mask_rcnn_R-50-FPN_1x_4.yaml',
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')
    parser.add_argument(
        '--load_ckpt')#, help='checkpoint path to load',default ='../models/16-2/model_step174999.pth')
    return parser.parse_args()


##by Weinong Wang
# additional subgradient descent on the sparsity-induced penalty term
def updatePOOL(model):
    # flag = 0
    for i, m in enumerate(model.modules()):
        if isinstance(m, mynn.Pool_wwn):
            alpha = m.GL_TYPE_ATTE_w
            lamda = m.weight
            # torch.pow(torch.exp(lamda[0, :]),2)
            grad_add_1 = cfg.REID.SR * (1 - 2*alpha[0,:])* \
                       (torch.exp(lamda[0,:]) * (torch.exp(lamda[0,:]) + torch.exp(lamda[1,:])) - torch.exp(2 * lamda[0,:]))\
                        /torch.pow(torch.exp(lamda[0,:]) + torch.exp(lamda[1,:]),2)

            grad_add_2 = cfg.REID.SR * (1 - 2*alpha[1,:])* \
                       (torch.exp(lamda[1,:]) * (torch.exp(lamda[0,:]) + torch.exp(lamda[1,:])) - torch.exp(2 * lamda[1,:]))\
                        /torch.pow(torch.exp(lamda[0,:]) + torch.exp(lamda[1,:]),2)
            grad_add = torch.cat([grad_add_1.unsqueeze(0),grad_add_2.unsqueeze(0)],0)
            # print(m.weight.grad)
            # print(m.weight)
            m.weight.grad.data.add_(grad_add)  # L1

def main():
    args = parse_args()
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False
    #args.labelsmooth = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        # In Pytorch, support for deterministic execution is still a bit clunky.
        # Use a well-known seed, for repeatability of experiments
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    # if cfg.RESNETS.SSN:
    #     rank, world_size = init_dist(backend='nccl', port=29500)
    #     print("The world_size is : {}".format(world_size))
    # summary_writer = SummaryWriter(osp.join(args.save_dir, 'tensorboard_log'))

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
        WEIGHT_TEST = cfg.REID.WEIGHT_TEST,
    )


    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=TrainTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH,cfg.REID.PRE_PRO_TYPE)),
        sampler=RandomIdentitySampler(dataset.train, num_instances=cfg.REID.TRI_NUM_INSTANCES),
        batch_size=cfg.REID.TRAIN_BATCH, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    if cfg.REID.WEIGHT_TEST:
        queryloader_1 = DataLoader(
            ImageDataset(dataset.query_1,
                         transform=TestTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH, cfg.REID.PRE_PRO_TYPE)),
            batch_size=cfg.REID.TEST_BATCH, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        galleryloader_1 = DataLoader(
            ImageDataset(dataset.gallery_1,
                         transform=TestTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH, cfg.REID.PRE_PRO_TYPE)),
            batch_size=cfg.REID.TEST_BATCH, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        queryloader_2 = DataLoader(
            ImageDataset(dataset.query_2, transform=TestTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH, cfg.REID.PRE_PRO_TYPE)),
            batch_size=cfg.REID.TEST_BATCH, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        galleryloader_2 = DataLoader(
            ImageDataset(dataset.gallery_2, transform=TestTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH, cfg.REID.PRE_PRO_TYPE)),
            batch_size=cfg.REID.TEST_BATCH, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    else:
        queryloader = DataLoader(
            ImageDataset(dataset.query, transform= TestTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH, cfg.REID.PRE_PRO_TYPE)),
            batch_size=cfg.REID.TEST_BATCH, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        galleryloader = DataLoader(
            ImageDataset(dataset.gallery, transform=TestTransform(cfg.REID.HEIGHT, cfg.REID.WIDTH, cfg.REID.PRE_PRO_TYPE)),
            batch_size=cfg.REID.TEST_BATCH, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    print("Initializing model: {}".format(cfg.MODEL.CONV_BODY.split('.')[-1].split('_')[1]))
    model = Generalized_FPN(num_classes=dataset.num_train_pids,
                              loss=cfg.REID.LOSS,
                              aligned =cfg.REID.ALIGNED,
                              strong_baseline= cfg.REID.STRONG_BASELINE)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    if cfg.REID.FOCALLOSS:
        criterion_class = FocalLoss(gamma=2, alpha=0.25, \
                                     labelsmooth=cfg.REID.LABElSMOOTH,
                                     num_classes=dataset.num_train_pids,
                                     epsilon=0.1)
    elif cfg.REID.LABLESMOOTH:
        criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion_class = CrossEntropyLoss(use_gpu=use_gpu)

    criterion_class_oimwarmup = None

    criterion_metric = TripletLossAlignedReID(margin=cfg.REID.TRI_MARGIN, aligned = cfg.REID.ALIGNED)


    criterion_center = CenterLoss(num_classes=dataset.num_train_pids, feat_dim=2048)

    if cfg.REID.CAMIDCLASS:
        criterion_class_cam = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion_class_cam = None

    criterion_class_mixup = None #CrossEntropyLoss(use_gpu=use_gpu)


    def load_ckpt(model, ckpt):
        """Load checkpoint"""
        mapping, _ = model.detectron_weight_mapping
        model_state_dict = model.state_dict()
        state_dict = {}
        for name in ckpt:
            try:
                if mapping[name] and name.split('.')[1] != 'posthoc_modules'\
                        and  name.split('.')[1] !='topdown_lateral_modules' \
                        and name.split('.')[1] !='conv_top':
                    state_dict[name] = ckpt[name]

            except:
                if name.split('.')[0] != 'bottleneck_sub2' and name.split('.')[0] != 'classifier' \
                and name.split('.')[0].split('_')[0] != 'classifier':
                    state_dict[name] = ckpt[name]
                    print('parameters: {}'.format(name))

        model.load_state_dict(state_dict, strict=False)

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        load_ckpt(model, checkpoint['state_dict'])

    start_epoch = args.start_epoch
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # print(checkpoint['state_dict'].keys())
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    ### Optimizer ###
    if cfg.REID.DETECTRON_OPTIMIZER:
        gn_params = []
        bias_params = []
        bias_param_names = []
        nonbias_params = []
        nonbias_param_names = []
        ssn_params = []
        ssn_param_names = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'gn' in key:
                    gn_params.append(value)
                elif 'bias' in key:
                    bias_params.append(value)
                    bias_param_names.append(key)
                elif key.endswith('_weight') and cfg.RESNETS.SSN:
                    # print('********{}'.format(key))
                    ssn_params.append(value)
                    ssn_param_names.append(key)
                else:
                    nonbias_params.append(value)
                    nonbias_param_names.append(key)
            else:
                # print('NO!')
                print('FREEZE para: {}'.format(key))
        # Learning rate of 0 is a dummy value to be set properly at the start of training
        params = [
            {'params': nonbias_params,
             'lr': cfg.SOLVER.BASE_LR, #0,
             'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
            {'params': bias_params,
             'lr': cfg.SOLVER.BASE_LR, #0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
             'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
            {'params': gn_params,
             'lr': cfg.SOLVER.BASE_LR, #0,
             'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN},
            {'params': ssn_params,
             'lr': cfg.SOLVER.BASE_LR/10 ,  # 0,  / 10
             'weight_decay': 0} #cfg.SOLVER.WEIGHT_DECAY
        ]
        # names of paramerters for each paramter
        param_names = [nonbias_param_names, bias_param_names, ssn_param_names]

        if cfg.SOLVER.TYPE == "SGD":
            optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = torch.optim.Adam(params,  weight_decay= cfg.SOLVER.WEIGHT_DECAY)
        elif cfg.SOLVER.TYPE == 'Rmsprop':
            optimizer = torch.optim.RMSprop(params,  momentum=cfg.SOLVER.MOMENTUM, weight_decay= cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise KeyError("Unsupported optim: {}".format(cfg.SOLVER.TYPE))

        lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.
        if cfg.SOLVER.LR_POLICY == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(400 - 18), eta_min=0.000001)

        if cfg.REID.SWA:
            # SWA: initialize SWA optimizer wrapper
            print('SWA training')
            # steps_per_epoch = len(loaders['train'].dataset) / args.batch_size
            # steps_per_epoch = int(steps_per_epoch)
            # print("Steps per epoch:", steps_per_epoch)
            optimizer = SWA(optimizer, swa_start=cfg.REID.SWA_START,
                            swa_freq=cfg.REID.SWA_FREQ, swa_lr=cfg.REID.SWA_LR)


        # Set index for decay steps
        decay_steps_ind = None
        for i in range(1, len(cfg.SOLVER.STEPS)):
            if cfg.SOLVER.STEPS[i] >= args.start_epoch:
                decay_steps_ind = i
                break
        if decay_steps_ind is None:
            decay_steps_ind = len(cfg.SOLVER.STEPS)

    else:
        raise KeyError("Unimplemented cfg.REID.DETECTRON_OPTIMIZER: {}".format(cfg.REID.DETECTRON_OPTIMIZER))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        if cfg.REID.WEIGHT_TEST:
            rank1 = wwn_test_2(model, queryloader_1, galleryloader_1, queryloader_2, galleryloader_2, use_gpu, args)
        else:
            wwn_test(model, queryloader, galleryloader, use_gpu, args)
        return 0

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    max_epoch = cfg.SOLVER.MAX_ITER
    for epoch in range(start_epoch, max_epoch):
        if cfg.REID.DETECTRON_OPTIMIZER:
            # Warm up
            if epoch < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    # alpha = (epoch+1) / cfg.SOLVER.WARM_UP_ITERS
                    # warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                    alpha = (cfg.SOLVER.BASE_LR - cfg.SOLVER.WARM_UP_FACTOR)/cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = epoch*alpha + cfg.SOLVER.WARM_UP_FACTOR
                    # print(warmup_factor)
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new =  warmup_factor
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif epoch == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR

            # Learning rate decay
            if cfg.SOLVER.LR_POLICY == 'steps_with_decay':
                if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                        epoch == cfg.SOLVER.STEPS[decay_steps_ind]:
                    print('Decay the learning on step %d', epoch)
                    lr_new = lr * cfg.SOLVER.GAMMA
                    net_utils.update_learning_rate(optimizer, lr, lr_new)
                    lr = optimizer.param_groups[0]['lr']
                    assert lr == lr_new
                    decay_steps_ind += 1
            elif cfg.SOLVER.LR_POLICY == 'cosine_annealing' and epoch >= cfg.SOLVER.WARM_UP_ITERS:
                scheduler.step()
                lr_new = scheduler.get_lr()[0]
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new

        else:
            if args.strong_baseline:
                adjust_lr(optimizer, epoch + 1)
            else:
                if args.stepsize > 0:
                    scheduler.step()

        start_train_time = time.time()
        train(epoch, model, criterion_class, criterion_metric, criterion_center,optimizer, trainloader, use_gpu, \
              criterion_class_oimwarmup,criterion_class_cam, \
              criterion_class_mixup, args)
        train_time += round(time.time() - start_train_time)

        # if cfg.REID.SWA and (epoch + 1) >= cfg.REID.SWA_START:
        #     # utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        #     # swa_n += 1
        #     if epoch == 0 or epoch % args.eval_step == args.eval_step - 1 or epoch == max_epoch - 1:
        #         # Batchnorm update
        #         optimizer.swap_swa_sgd()
        #         # print('WWN')
        #         optimizer.bn_update(trainloader, model, device='cuda')
        #         # swa_res = utils.eval(loaders['test'], model, criterion)

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == max_epoch:
            print("==> Test")

            if cfg.REID.SWA and (epoch + 1) >= cfg.REID.SWA_START:
                optimizer.swap_swa_sgd()
                # print('WWN')
                optimizer.bn_update(trainloader, model, device='cuda')
                if cfg.REID.WEIGHT_TEST:
                    rank1 = wwn_test_2(model, queryloader_1, galleryloader_1, queryloader_2, galleryloader_2, use_gpu,
                                       args)
                else:
                    rank1 = wwn_test(model, queryloader, galleryloader, use_gpu, args)

                optimizer.swap_swa_sgd()
            else:
                if cfg.REID.WEIGHT_TEST:
                    rank1 = wwn_test_2(model, queryloader_1, galleryloader_1, queryloader_2, galleryloader_2, use_gpu, args)
                else:
                    rank1 = wwn_test(model, queryloader, galleryloader, use_gpu, args)


            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint_best({
               'state_dict': state_dict,
               'rank1': rank1,
               'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_class, criterion_metric,criterion_center, optimizer, trainloader, use_gpu,criterion_class_oimwarmup,criterion_class_cam, criterion_class_mixup,args):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    ce_losses = AverageMeter()

    end = time.time()
    if cfg.RESNETS.SSN:
        trainset_length = len(trainloader)
    for batch_idx, (imgs, pids, camid, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        if cfg.RESNETS.SSN:
            rad = (epoch * trainset_length + batch_idx) / (cfg.SOLVER.MAX_ITER * trainset_length)
            assert (rad >= 0.0 and rad <= 1.0)
            for module in model.modules():
                if isinstance(module, SSN2d):
                    module.set_rad(rad)

        if cfg.REID.REGULARIZED_POOLING:
            print('Not implemented!')

        else:
            outputs, features, f, center_weight, noise, outputs_cam = model(imgs, pids)

            xent_loss = criterion_class(outputs, pids)

            local_features = None
            global_loss, local_loss = criterion_metric(f, pids, local_features, epoch)

            if cfg.REID.CENTER > 0.0 and not cfg.REID.OIM_LABLESMOOTH:
                center_loss = cfg.REID.CENTER * criterion_center(features, pids, center_weight, noise)
            else:
                center_loss = 0

            if cfg.REID.CAMIDCLASS:
                camid = torch.zeros(outputs_cam.size()).scatter_(1, camid.unsqueeze(1).data.cpu(), 1).cuda()
                # print(camid.size())
                cam_xent_loss = 1.0 * criterion_class_cam(outputs_cam, camid)
            else:
                cam_xent_loss = 0



            loss = 1.0 * xent_loss + 1.0 * global_loss + center_loss + cam_xent_loss
            optimizer.zero_grad()
            loss.backward()

            # if cfg.REID.SR > 0.0:
            #     updatePOOL(model)
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        if hasattr(xent_loss, 'item'):
            xent_losses.update(xent_loss.item(), pids.size(0))
        else:
            xent_losses.update(0, pids.size(0))
        if hasattr(global_loss, 'item'):
            global_losses.update(global_loss.item(), pids.size(0))
        else:
            global_losses.update(0, pids.size(0))
        if hasattr(center_loss, 'item'):
            ce_losses.update(center_loss.item(), pids.size(0))
        else:
            ce_losses.update(0, pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'GLoss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'
                  'CELoss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                  'lr {lr:.6f}\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time, data_time=data_time,
                loss=losses, xent_loss=xent_losses, global_loss=global_losses, ce_loss=ce_losses,
                lr=optimizer.param_groups[0]['lr']))

def wwn_test(model, queryloader, galleryloader, use_gpu, args , ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()
    if cfg.RESNETS.SSN:
        sync_bn_stat(model, 1)

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        q_img_path = []
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()
            end = time.time()
            features, local_features = model(imgs)
            # features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_img_path.extend(img_paths)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_img_path = np.asarray(q_img_path)
        # np.savez("1-query.npz", qf = qf.numpy(),q_pids =q_pids, q_camids=q_camids, q_img_path=q_img_path)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        g_img_path = []
        end = time.time()
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            # features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_img_path.extend(img_paths)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_img_path = np.asarray(g_img_path)
        # np.savez("1-gallery.npz", gf =gf.numpy(), g_pids = g_pids, g_camids =g_camids, g_img_path=g_img_path)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, cfg.REID.TEST_BATCH))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if not cfg.REID.TEST_DISTANCE == 'global':
        print("Not only using global branch")
        from util_reid.distance import low_memory_local_dist
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if cfg.REID.TEST_DISTANCE == 'local':
            print("Only using local branch")
            distmat = local_distmat
        if cfg.REID.TEST_DISTANCE == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if cfg.REID.RE_RANKING:
        from util_reid.re_ranking import re_ranking
        if cfg.REID.TEST_DISTANCE == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
            elif args.test_distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
    return cmc[0]


def wwn_test_2(model, queryloader_1, galleryloader_1, queryloader_2, galleryloader_2, use_gpu, args , ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        q_img_path = []
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(queryloader_1):
            if use_gpu: imgs = imgs.cuda()
            end = time.time()
            features, local_features = model(imgs)
            # features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_img_path.extend(img_paths)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_img_path = np.asarray(q_img_path)
        # np.savez("1-query.npz", qf = qf.numpy(),q_pids =q_pids, q_camids=q_camids, q_img_path=q_img_path)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        g_img_path = []
        end = time.time()
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(galleryloader_1):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            # features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_img_path.extend(img_paths)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_img_path = np.asarray(g_img_path)
        # np.savez("1-gallery.npz", gf =gf.numpy(), g_pids = g_pids, g_camids =g_camids, g_img_path=g_img_path)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, cfg.REID.TEST_BATCH))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if not cfg.REID.TEST_DISTANCE == 'global':
        print("Not only using global branch")
        from util_reid.distance import low_memory_local_dist
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if cfg.REID.TEST_DISTANCE == 'local':
            print("Only using local branch")
            distmat = local_distmat
        if cfg.REID.TEST_DISTANCE == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc_1, mAP_1 = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    ## for 2
    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        q_img_path = []
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(queryloader_2):
            if use_gpu: imgs = imgs.cuda()
            end = time.time()
            features, local_features = model(imgs)
            # features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_img_path.extend(img_paths)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_img_path = np.asarray(q_img_path)
        # np.savez("1-query.npz", qf = qf.numpy(),q_pids =q_pids, q_camids=q_camids, q_img_path=q_img_path)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, lgf = [], [], [], []
        g_img_path = []
        end = time.time()
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(galleryloader_2):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            # features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_img_path.extend(img_paths)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_img_path = np.asarray(g_img_path)
        # np.savez("1-gallery.npz", gf =gf.numpy(), g_pids = g_pids, g_camids =g_camids, g_img_path=g_img_path)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))


    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, cfg.REID.TEST_BATCH))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if not cfg.REID.TEST_DISTANCE == 'global':
        print("Not only using global branch")
        from util_reid.distance import low_memory_local_dist

        lqf = lqf.permute(0, 2, 1)
        lgf = lgf.permute(0, 2, 1)
        local_distmat = low_memory_local_dist(lqf.numpy(), lgf.numpy(), aligned=not args.unaligned)
        if cfg.REID.TEST_DISTANCE == 'local':
            print("Only using local branch")
            distmat = local_distmat
        if cfg.REID.TEST_DISTANCE == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat + distmat
    print("Computing CMC and mAP")
    cmc_2, mAP_2 = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(0.25 * mAP_1 + 0.75 * mAP_2))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, 0.25*cmc_1[r - 1]+0.75*cmc_2[r - 1]))
    print("------------------")

    # if cfg.REID.RE_RANKING:
    #     from util_reid.re_ranking import re_ranking
    #     if cfg.TEST_DISTANCE == 'global':
    #         print("Only using global branch for reranking")
    #         distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
    #     else:
    #         local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= not args.unaligned)
    #         local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= not args.unaligned)
    #         local_dist = np.concatenate(
    #             [np.concatenate([local_qq_distmat, local_distmat], axis=1),
    #              np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
    #             axis=0)
    #         if args.test_distance == 'local':
    #             print("Only using local branch for reranking")
    #             distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
    #         elif args.test_distance == 'global_local':
    #             print("Using global and local branches for reranking")
    #             distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
    #     print("Computing CMC and mAP for re_ranking")
    #     cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
    #
    #     print("Results ----------")
    #     print("mAP(RK): {:.1%}".format(mAP))
    #     print("CMC curve(RK)")
    #     for r in ranks:
    #         print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    #     print("------------------")
    return 0.25 * cmc_1[0] + 0.75 * cmc_2[0]
if __name__ == '__main__':
    main()
