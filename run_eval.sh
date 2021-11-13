# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run_eval.sh
#   Last Modified : 2021-11-13 22:08
#   Describe      : 
#
# ====================================================

pretrained_model=$1
CUDA_VISIBLE_DEVICES=$2
export CUDA_VISIBLE_DEVICES
    python train.py \
    --gpus 1 \
    --train_batchsize 1 \
    --valid_batchsize 1 \
    --num_workers 8 \
    --data_dir "data/" \
    --model_name 'BertModel' \
    --pooler_type 'cls' \
    --accelerator 'ddp' \
    --checkpoint_path "epoch=02-valid_f1=0.6152.ckpt" \
    --eval \
    # --recreate_dataset
