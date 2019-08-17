##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Beyond strong basleine for Person Re-ID. 2019
## Dukemtmcreid: Rank-1/mAP = 86.6/77.1
## Created by: Weinong  Wang
## Based on the structure of pytorch-detectron and Alignedreid
## Tencent, Youtu X-Lab
## Email: weinong.wang@hotmail.com
## Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Requirements

Trained and Tested on Python3.6
 1. pytorch >= 0.4.0
	torchvision>=0.2.0
	matplotlib
	numpy
	scipy
	opencv
	pyyaml
	packaging
	PIL
	tqdm
	time
## Characteristics
 1. basic data augmentation: horizontal flip, random erasing
 2.  multi backbones: Resnet50, Resnet18
 3. Focal loss; Label smooth; Warm up; Center Loss; Re-ranking; Triplet loss; Contrastive Loss; Softmax Loss; improved OIM loss
 4. In folder "dukedata", add the support to Duke4reid dataset and combining the dukereid with duke4reid

			
## Data Preparation
The data structure is following the Dukemtmcreid
 -  reid_tools/data/dukemtmc-reid
     - DukeMTMC-reID
## Train-baseline

--dataset the name of the datatset benchmark
--root the path to your datatset
--eval-step specify how many training epochs to test once
--cfg specify the configure yaml file
--save-dir specify the save path of the pth and log files
eg. 
 CUDA_VISIBLE_DEVICES=0 python reid_tools/train_net_step_reid.py --dataset dukemtmcreid --root reid_tools/data --eval-step  40 --cfg configs/reid/R-50-Base.yaml  --save-dir RD_XDIST_RESULTS/duke/base-50

## New baseline ( OIM-labelsmooth + joint dukeid-duke4reid training)
CUDA_VISIBLE_DEVICES=0 python3 reid_tools/train_net_step_reid.py --save-dir RD_XDIST_RESULTS/duke/Base-center-re-oimlabelsmoothfcnobia  --eval-step 40 --start-eval 40 --cfg configs/reid/R-50-Base-oim.yaml --root reid_tools/data/  --dataset  dukecombine

## Test
 CUDA_VISIBLE_DEVICES=0 python reid_tools/train_net_reid.py --dataset dukemtmcreid --root reid_tools/data --cfg configs/reid/R-50-Base.yaml --load_ckpt ***.pth --evaluate
## New baseline test
# test on dukereid
CUDA_VISIBLE_DEVICES=0 python3 reid_tools/train_net_step_reid.py --save-dir RD_XDIST_RESULTS/duke/test --load_ckpt RD_XDIST_RESULTS/duke/Base-center-re-oimlabelsmoothfcnobia/best_model.pth.tar  --evaluate --cfg configs/reid/R-50-Base-oim.yaml --root reid_tools/data/  --dataset  dukemtmcreid

# test on duke4reid
CUDA_VISIBLE_DEVICES=0 python3 reid_tools/train_net_step_reid.py --save-dir RD_XDIST_RESULTS/duke/test --load_ckpt RD_XDIST_RESULTS/duke/Base-center-re-oimlabelsmoothfcnobia/best_model.pth.tar  --evaluate --cfg configs/reid/R-50-Base-oim.yaml --root reid_tools/data/  --dataset  dukemtmc4reid

# test on joint dukereid-duke4reid, set  WEIGHT_TEST in R-50-Base-oim.yaml to False
 CUDA_VISIBLE_DEVICES=0 python3 reid_tools/train_net_step_reid.py --save-dir RD_XDIST_RESULTS/duke/test --load_ckpt RD_XDIST_RESULTS/duke/Base-center-re-oimlabelsmoothfcnobia/best_model.pth.tar  --evaluate --cfg configs/reid/R-50-Base-oim.yaml --root reid_tools/data/  --dataset  dukecombine

# test on dukereid and duke4reid seperately, 0.25*dukereid + 0.75*duke4reid, set  WEIGHT_TEST in R-50-Base-oim.yaml to True
 CUDA_VISIBLE_DEVICES=0 python3 reid_tools/train_net_step_reid.py --save-dir RD_XDIST_RESULTS/duke/test --load_ckpt RD_XDIST_RESULTS/duke/Base-center-re-oimlabelsmoothfcnobia/best_model.pth.tar  --evaluate --cfg configs/reid/R-50-Base-oim.yaml --root reid_tools/data/  --dataset  dukecombine

