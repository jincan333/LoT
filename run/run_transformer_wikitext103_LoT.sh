#!/bin/bash
max_epoch=4
len=100
lr=0.001
clip=0.25
dropout=0.1
batch_size=30
alpha=0.1
student_steps_ratio=4
T=1.5
seed=0
gpu=0
prefix='LoT_Transformer_WikiText103'
experiment_name=max_epoch${max_epoch}_alpha${alpha}_N${student_steps_ratio}_T${T}_lr${lr}_seed${seed}_gpu${gpu}
work_dir=ckpt/${prefix}
echo 'Run training...'
log_folder_name=logs/${prefix}
if [ ! -d ${log_folder_name} ]; then
    mkdir -p ${log_folder_name}
fi
log_filename=${log_folder_name}/${experiment_name}.log
nohup python -u trainer/transformer_xl_lm.py \
    --cuda \
    --gpu ${gpu} \
    --data data/wikitext-103/ \
    --dataset wt103 \
    --n_layer 4 \
    --d_model 410 \
    --n_head 12 \
    --d_head 41 \
    --d_inner 2100 \
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
    --exp_name ${prefix}_${experiment_name} \
    --max_epoch ${max_epoch} \
    --seed ${seed} \
    --work_dir ${work_dir} \
    > ${log_filename} 2>&1 &
    