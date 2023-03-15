#!/usr/bin/env bash
# 
# Train MVSTER (Multi-GPU training):
#
# Note: 
#   bash ./scripts/train_BDS5.sh  BDS5_512x640_rt20pct_25srcs_smoothL1_LR0.005
#   bash ./scripts/train_BDS5.sh  BDS5_512x640_rt20pct_10srcs_L1_LR0.001

TRAIN_DATASET="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS5_mvs_training_512x640"
TRAINLIST="lists/BDS5/train.txt"
TESTLIST="lists/BDS5/test.txt"
PAIRFILE="pair_70x10.txt"

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
--pair_fname $PAIRFILE \
--dataset=blender4 \
--interval_scale=1.5 \
--train_nviews 5 \
--ndepths="8,8,4,4" \
--depth_inter_r="0.5,0.5,0.5,1" \
--epochs=16 \
--lr=0.001 \
--wd=0.0001 \
--l1ce_lw="0.003,1" \
--lrepochs="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15:1.2" \
--batch_size=6 \
--summary_freq 100 \
--group_cor \
--group_cor_dim="8,8,4,4" \
--rt \
--mono \
--inverse_depth \
--attn_temp 2 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &


# --pos_enc 2 \
# --ot_continous \
# --mono \
# --use_raw_train \
# --resume \