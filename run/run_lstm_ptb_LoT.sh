#!/bin/bash

models_num=2
detach=1
epochs=60
alpha=1
lr=30
student_steps_ratio=5
dropout=0.45
clip=0.2
batch_size=20
bptt=35
seed=0
gpu=0
data='ptb'
prefix='LoT_LSTM_PTB'
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
nohup python -u trainer/lstm_ptb.py \
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