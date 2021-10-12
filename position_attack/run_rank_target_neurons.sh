#!/usr/bin/env bash


python main_rank_neurons.py \
       --model deeplabv3_resnet50 --gpu_id 0  --lr 0.1 \
       --val_batch_size 1 --output_stride 16 \
       --data_root /mnt/zeta_share_1/amirul/datasets/CITYSCAPES \
       --dataset cityscapes --test_only \
       --ckpt checkpoints/zero_padding_best_deeplabv3_resnet50_cityscapes_os16.pth

python main_position_attack.py \
       --model deeplabv3_resnet50 --gpu_id 0  --lr 0.1 \
       --val_batch_size 4  --output_stride 16 \
       --data_root /mnt/zeta_share_1/amirul/datasets/CITYSCAPES \
       --dataset cityscapes --test_only \
       --region overall \
       --topN 100 \
       --ckpt checkpoints/zero_padding_best_deeplabv3_resnet50_cityscapes_os16.pth