#!/bin/bash

depth=20
models_num=2
depth_list='20_20'
detach=1
epochs_list=(180 180 180 180)
T_list=(1.5 1.5 1.5 1.5)
alpha_list=(0 0 0 0.5 0.5)
student_ratio=(0 0 0 1 1 1 1 1)
student_weight_decay=0.0001
# loss choice: kl_ce kl symmetric_kl_ce symmetric_kl  
student_loss='kl_ce'
seed_list=(0 1 2 0 0 0 0 0)
gpu_list=(1 0 1)
prefix='2.ensemble_ckpt_cifar10'
for i in ${!gpu_list[@]};do
    alpha=${alpha_list[i]}
    epochs=${epochs_list[i]}
    student_alpha=${alpha}
    student_ratio=${student_ratio[i]}
    T=${T_list[i]}
    gpu=${gpu_list[i]}
    seed=${seed_list[i]}
    experiment_name=${prefix}_depth_list${depth_list}_aplha${alpha}_epochs${epochs}_student_ratio${student_ratio}_student_weight_decay${student_weight_decay}_student_loss${student_loss}_T${T}_seed${seed}_gpu${gpu}
    folder_name=logs/unequal_steps_cifar10
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u scripts/image_classification_unequal_steps.py \
    exp_name=${experiment_name} \
    teacher.use_ckpts=False \
    classifier.depth=${depth} \
    trainer.num_epochs=${epochs} \
    trainer.optimizer.lr=1 \
    trainer.optimizer.weight_decay=0.0001 \
    trainer.lr_scheduler.eta_min=0. \
    trainer.distill_teacher=False \
    dataloader.batch_size=256 \
    trial_id=${seed} \
    gpu=${gpu} \
    loss.alpha=${alpha} \
    models_num=${models_num} \
    detach=${detach} \
    depth_list=${depth_list} \
    T=${T} \
    student_alpha=${student_alpha} \
    student_ratio=${student_ratio} \
    student_weight_decay=${student_weight_decay} \
    student_loss=${student_loss} \
    > ${log_filename} 2>&1 &
done