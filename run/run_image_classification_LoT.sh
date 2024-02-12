#!/bin/bash

models_num=2
depth_list='20_20'
detach=1
epochs=180
T=1.5
alpha=0.5
student_steps_ratio=2
# loss choice: kl_ce kl symmetric_kl_ce symmetric_kl  
loss='kl_ce'
seed=0
gpu=0
dataset='cifar100'
prefix='LoT_ResNet_CIFAR'
experiment_name=depth_list${depth_list}_aplha${alpha}_N${student_steps_ratio}_epochs${epochs}_loss${student_loss}_T${T}_seed${seed}_gpu${gpu}
log_folder_name=logs/${prefix}_${dataset}
if [ ! -d ${log_folder_name} ]; then
    mkdir -p ${log_folder_name}
fi
ckpt_folder_name=ckpt/${prefix}_${dataset}
if [ ! -d ${ckpt_folder_name} ]; then
    mkdir -p ${ckpt_folder_name}
fi
save=${ckpt_folder_name}/${experiment_name}

log_filename=${log_folder_name}/${experiment_name}.log
nohup python -u trainer/image_classification.py \
    --exp_name=${experiment_name} \
    --epochs=${epochs} \
    --lr=1 \
    --weight_decay=0.0001 \
    --batch_size=256 \
    --seed=${seed} \
    --gpu=${gpu} \
    --alpha=${alpha} \
    --models_num=${models_num} \
    --depth_list=${depth_list} \
    --T=${T} \
    --student_steps_ratio=${student_steps_ratio} \
    --loss=${loss} \
    --save=${save} \
> ${log_filename} 2>&1 &