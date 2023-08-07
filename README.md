# L^2 compression

## Introduction

Deep neural networks have delivered remarkable performance and have been widely used in various visual tasks. However, their huge size causes significant inconvenience for transmission and storage. This work proposes a unified post-training model size compression method that combines lossy and lossless compression. 

## Requirements

* torch

* torchvision

* constriction

* ninja

* matplotlib

* timm

````shell
pip install -r requirements.txt
````

## Usage

### train

* change the model checkpoints paths in train/train_final

  ```` python
  state_path = {
      "resnet18": "",
      "resnet50": "",
      "mobilenetv2": "",
      "mnasnet": "",
      "regnetx_600m": "",
      "regnetx_3200m": ""
  }
  ````

* change dataset in  train/train_final

  ````python
  train_dataset = ImageNetDataset(ROOTDIR + '/train/', 'train.txt', train_transform)
  test_dataset = ImageNetDataset(ROOTDIR + 'val/', METADIR + 'val.txt', val_transform)
  ````

  * the calibration dataset we provide is train.txt in the repo

* examples are in run.sh

  ````shell
  # example ResNet18
  python -u -m train.train_final \
      --model_name resnet18 \
      --lambda_r 1e-6 \
      --lambda_kd 1.0 \
      --weight_transform edgescale \
      --bias_transform scale \
      --transform_iter 300 \
      --transform_lr 0.0001 \
      --reconstruct_iter 1000 \
      --reconstruct_lr 5e-06 \
      --resolution 64 \
      --diffkernel cos \
      --log_path ./log \
      --target_CR 10.0  \
      --run_name resnet_test
  ````

## encode and decode

refer to cal_bit.py