#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='freenet.freenet_1_0_salinas'

model_dir='./log/salinas/freenet/1.0_poly'


python train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.save_ckpt_interval_epoch 9999