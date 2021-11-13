# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run_finetune.sh
#   Last Modified : 2021-11-13 22:08
#   Describe      : 
#
# ====================================================
# CUDA_LAUNCH_BLOCKING=1
pretrained_model=$1
CUDA_VISIBLE_DEVICES=$2
export CUDA_VISIBLE_DEVICES

for bert_lr in 5e-6 1e-5 3e-5 5e-5
do
    for lr in 1e-4 3e-4 5e-4
    do
            python train.py \
            --gpus 1 \
            --max_epochs 100 \
            --lr ${lr} \
            --bert_lr ${bert_lr} \
            --train_batchsize 1 \
            --valid_batchsize 1 \
            --num_workers 8 \
            --data_dir "data/" \
            --model_name 'BertModel' \
            --pretrained_model $1 \
            --warmup 0.1 \
            --pooler_type 'cls' \
            --finetune \
            --accelerator 'ddp' \
            --gradient_clip_val 1 \
            --precision 32 \
            --val_check_interval 1.0 \
            --adv \
            --mlp_dropout 0. 
    done
done
            # --checkpoint_path 'epoch=02-valid_f1=0.6172.ckpt'
            # --eval
            # --track_grad_norm 2 \
            # --child_tuning \
            # --recreate_dataset \
            # --plugins "fsdp" \
