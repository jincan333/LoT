#!/bin/bash

models_num=2
detach=1
epochs=3
alpha=0.1
lr=40
student_steps_ratio=4
dropout=0.1
clip=0.2
batch_size=30
bptt=100
seed=0
gpu=2
data='wt103'
prefix='LoT_LSTM_WikiText103'
experiment_name=aplha${alpha}_N${student_steps_ratio}_epochs${epochs}_lr${lr}_droupout${dropout}_clip${clip}_seed${seed}_gpu${gpu}
save=ckpt/${prefix}/${experiment_name}.pt

log_folder_name=logs/${prefix}
if [ ! -d ${log_folder_name} ]; then
    mkdir -p ${log_folder_name}
fi
ckpt_folder_name=ckpt/${prefix}
if [ ! -d ${ckpt_folder_name} ]; then
    mkdir -p ${ckpt_folder_name}
fi

log_filename=${log_folder_name}/${experiment_name}.log
nohup python -u trainer/lstm_wt103.py \
    --exp_name ${prefix}_${experiment_name} \
    --data ${data} \
    --models_num ${models_num} \
    --detach ${detach} \
    --alpha ${alpha} \
    --gpu ${gpu} \
    --epochs ${epochs} \
    --save ${save} \
    --seed ${seed} \
    --lr ${lr} \
    --dropout ${dropout} \
    --clip ${clip} \
    --batch_size ${batch_size} \
    --bptt ${bptt} \
    --student_steps_ratio ${student_steps_ratio} \
> ${log_filename} 2>&1 &