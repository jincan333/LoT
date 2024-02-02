#!/bin/bash
max_epoch=60
len=70
lr=0.001
clip=0.35
dropout=0.25
batch_size=60
alpha=0.5
student_steps_ratio=5
T=1.5
gpu=0
prefix='Transformer_PTB_LoT'
experiment_name=${prefix}_max_epoch${max_epoch}_alpha${alpha}_N${student_steps_ratio}_T${T}_lr${lr}_gpu${gpu}
echo 'Run training...'
log_folder_name=logs
if [ ! -d ${log_folder_name} ]; then
    mkdir -p ${log_folder_name}
fi
log_filename=${log_folder_name}/${experiment_name}.log
nohup python -u transformer_xl_lm.py \
    --cuda \
    --gpu ${gpu} \
    --data data/ptb/ \
    --dataset wt103 \
    --n_layer 4 \
    --d_model 200 \
    --n_head 8 \
    --d_head 20 \
    --d_inner 1000 \
    --dropout ${dropout} \
    --dropatt 0.0 \
    --optim adam \
    --lr ${lr} \
    --clip ${clip} \
    --dropout ${dropout} \
    --warmup_step 0 \
    --tgt_len ${len} \
    --mem_len ${len} \
    --eval_tgt_len ${len} \
    --batch_size ${batch_size} \
    --eval-interval 1000 \
    --alpha ${alpha} \
    --student_steps_ratio ${student_steps_ratio} \
    --T ${T} \
    --exp_name ${experiment_name} \
    --max_epoch ${max_epoch} \
    --seed 1 \
    > ${log_filename} 2>&1 &
    