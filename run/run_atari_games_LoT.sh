#!/bin/bash

total_timesteps=20000000
T=1
alpha=0.5
student_steps_ratio=5
obs_num=10
lr=0.00025
seed=0
gpu=1
# 'BreakoutNoFrameskip-v4', 'BeamRiderNoFrameskip-v4', 'UpNDownNoFrameskip-v4', 'GravitarNoFrameskip-v4'
env_id='BeamRiderNoFrameskip-v4'
prefix='LoT_RL'
experiment_name=${prefix}_${env_id}_total_steps${total_timesteps}_aplha${alpha}_student_steps_ratio${student_ratio}_lr${lr}_obs_num${obs_num}_seed${seed}
folder_name=logs/${prefix}_${env_id}
save=ckpt/${prefix}_${env_id}/${experiment_name}.pt
if [ ! -d ${folder_name} ]; then
    mkdir -p ${folder_name}
fi
ckpt_folder_name=ckpt/${prefix}_${env_id}
if [ ! -d ${ckpt_folder_name} ]; then
    mkdir -p ${ckpt_folder_name}
fi
log_filename=${folder_name}/${experiment_name}.log
nohup python -u atari_games.py \
    --env-id ${env_id} \
    --exp-name ${experiment_name} \
    --total-timesteps ${total_timesteps} \
    --T ${T} \
    --alpha ${alpha} \
    --student-steps-ratio ${student_steps_ratio} \
    --learning-rate ${lr} \
    --obs-num ${obs_num} \
    --gpu ${gpu} \
    --seed ${seed} \
    --save ${save} \
> ${log_filename} 2>&1 &