#!/usr/bin/env bash
# 
# Train MVSTER (Multi-GPU training):
# -Train with middle size (512x640): bash ./scripts/train_dtu.sh  experiment_name
# 
# Note: 
#   bash ./scripts/train_BDS6.sh  BDS6_512x640_rt10pct_Nviews5
#   bash ./scripts/train_BDS6.sh  BDS6_512x640_rt10pct_Nviews2
#   bash ./scripts/train_BDS6.sh  BDS6_512x640_rt10pct_Nviews5_10pctL1
#   bash ./scripts/train_BDS6.sh  BDS6_512x640_rt20pct25srcs_Nviews7_0.5pctL1
#   bash ./scripts/train_BDS6.sh  BDS6_512x640_rt20pct25srcs_Nviews9_0.5pctL1
#   bash ./scripts/train_BDS6.sh  newBDS6_512x640_Nviews5_rt20pct10srcs


TRAIN_DATASET="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS6_mvs_training_1024x1280"
TRAINLIST="lists/BDS6/train.txt"
TESTLIST="lists/BDS6/test.txt"

exp=$1
PY_ARGS=${@:2}

LOG_DIR="./outputs/"$exp 
LOG_FILE="log_"$exp".txt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

echo "====== Check log in file: tail -f ${LOG_DIR}/${LOG_FILE}"

python train_mvs4.py \
--logdir $LOG_DIR \
--trainpath $TRAIN_DATASET \
--trainlist $TRAINLIST \
--testlist $TESTLIST  \
--dataset=blender4 \
--train_nviews 5 \
--interval_scale=1.04 \
--epochs=24 \
--lr=0.001 \
--wd=0.0001 \
--l1ce_lw="0.003,1" \
--lrepochs="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24:1.2" \
--batch_size=6 \
--summary_freq 100 \
--mono \
--group_cor \
--inverse_depth \
--rt \
--attn_temp 2 \
--resume \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR"/"$LOG_FILE 


# --loadckpt $CHKPT \
# --dcn \
# --mono \
# --use_raw_train \
# --resume \