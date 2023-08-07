!/bin/bash

# example ViT
# need to install timm
# high timm version may conflict
python -u -m train.train_final \
    --model_name ViT \
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
    --run_name vit_test \

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