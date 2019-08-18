# Beyond strong basleine for Person Re-ID. 2019

The codes follow the structure of [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) and [AlignedReID](https://github.com/michuanhaohao/AlignedReID). It builds a very strong baseline for Re-Identification task by exploring bag of tricks. Without center loss and label smoothing, our project can achieve comparable(or better) results than [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). Also, the project serves as the baseline of our new method，denoted as **Orthogonal Center Learning with Subspace Masking**. ***We are grateful for your contribution on this project and hope the project can help your research or work.***

## Authors
[Weinong Wang](https://github.com/paozhuanyinyuba)

## Supported tricks
- Warm up learning rate (wu)
- Last stride = 1 (lsr)
- BNNeck (bnn)
- BN of zero gamma (bnzg)
- classifier have no bias (cnb)
- Random erasing augmentation (re)
- Label smoothing (lsm)
- Center loss (cl)
- Re-ranking (rr)
- backbone: ResNet-50, ResNet-18

## Supported dataset
- Market1501
- DukeMTMC-reID
- CUHK03
- MSMT17

## Experements results (rank1/mAP)
Note: Although the tricks mentioned above are all supported, our baseline does not have all of them.

| Model | Market1501 | DukeMTMC-reID | CUHK03 | MSMT17 |
| --- | -- | -- | -- | -- |
|[reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) (ResNet-50+wu+lsr+bnn+re+lsm+cl) | 94.5 (85.9) |  86.4 (76.4) | - | -
|ours(ResNet-50+wu+lsr+bnn+re) | 93.9 (84.6) |  86.6 (77.1) | 73.1 (70.6) |  80.6(65.1)
|ours(ResNet-50+wu+lsr+bnn+re+cnb) | 93.8 (84.8) |  87.1 (77.3) | 76.2 (52.9) |  75.9(52.6)


# Requirements

Training and Testing on Python3.5

	pytorch = 0.4.0
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

## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/paozhuanyinyuba/Beyond-Strong-Baseline-for-Person-ReID.git`

3. Prepare dataset

    Create a directory to store reid datasets under this repo or outside this repo. The path to the root of the dataset is set by the command line argument, namely “ - - root **” .

    For instance, you can create a directory to store reid datasets under this repo via

    ```bash
    cd Beyond-Strong-Baseline-for-Person-ReID
    mkdir data
    ```

    （1）Market1501

    * Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    * Extract dataset and rename to `market1501`. The data structure would like:

    ```bash
    data
        market1501 # this folder contains 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```

   （2）DukeMTMC-reID

    * Download dataset to `data/` from https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset
    * Extract dataset and rename to `dukemtmc-reid`. The data structure would like:

    ```bash
    data
        dukemtmc-reid
        	DukeMTMC-reID # this folder contains 8 files.
            	bounding_box_test/
            	bounding_box_train/
            	......
    ```
	
   （3）CUHK03

    * Download dataset to `data/` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
    * Extract dataset and rename to `cuhk03`. The data structure would like:

    ```bash
    data
        cuhk03
        	cuhk03_new_protocol_config_labeled.mat #
        	cuhk03_new_protocol_config_detected.mat #
        	cuhk03_release #
            	cuhk-03.mat
            	README.md
    ```
	
   （4）MSMT17

    * Download dataset to `data/` from https://www.pkuvmc.com/publications/msmt17.html
    * Extract dataset and rename to `msmt17`. The data structure would like:
	
    ```bash
    data
        msmt17
        	MSMT17_V1 # this folder contains 6 files.
            	train/
            	test/
            	......
    ```

5. Prepare pretrained model

	Create a directory to store pretrained model under this repo or outside this repo. The path to the pretrained model is set in every single training config file in `configs/reid/*.yaml`. Take  `configs/reid/R-50-Base.yaml` for example, line 6 shows:
   	"RESNETS: 
   		IMAGENET_PRETRAINED_WEIGHTS: 'pretrained_model/resnet50-19c8e357.pth'"

    For instance, you can create a directory to store the pretrained model under this repo via

    ```bash
    cd Beyond-Strong-Baseline-for-Person-ReID
    mkdir pretrained_model
    ```

    （1）ResNet
      * put [ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) or [ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth) in the folder `pretrained_model/`

   （2）More backbone coming soon ......

6. If you want to know the detailed configurations and their meaning, please refer to `lib/core/config.py`. 

## Train

- ResNet-50+wu+lsr+bnn+re

```bash
python3 reid_tools/train_net_step_reid.py --dataset "('dataset name, eg. cuhk03, market1501, msmt17 and dukemtmcreid')"  --root "('Your path to the root of the dataset')" --eval-step  "('How many epochs to test once')" --cfg configs/reid/R-50-Base.yaml  --save-dir "('your path to save checkpoints and logs')"
```

- ResNet-18+wu+lsr+bnn+re

```bash
python3 reid_tools/train_net_step_reid.py --dataset "('dataset name, eg. cuhk03, market1501, msmt17 and dukemtmcreid')"  --root "('Your path to the root of the dataset')" --eval-step  "('How many epochs to test once')" --cfg configs/reid/R-18-Base.yaml  --save-dir "('your path to save checkpoints and logs')"
```

## Test

- ResNet-50+wu+lsr+bnn+re. Test with Euclidean distance using feature before BN without re-ranking.

```bash
python3 reid_tools/train_net_step_reid.py --save-dir "('your path to save checkpoints and logs')"  --cfg configs/reid/R-50-Base.yaml --root "('Your path to the root of the dataset')"  --dataset "('dataset name, eg. cuhk03, market1501, msmt17 and dukemtmcreid')"  --evaluate --load_ckpt "('your path to trained checkpoints')"
```

- ResNet-18+wu+lsr+bnn+re. Test with Euclidean distance using feature before BN without re-ranking.

```bash
python3 reid_tools/train_net_step_reid.py --save-dir "('your path to save checkpoints and logs')"  --cfg configs/reid/R-18-Base.yaml --root "('Your path to the root of the dataset')"  --dataset "('dataset name, eg. cuhk03, market1501, msmt17 and dukemtmcreid')"  --evaluate --load_ckpt "('your path to trained checkpoints')"
```
