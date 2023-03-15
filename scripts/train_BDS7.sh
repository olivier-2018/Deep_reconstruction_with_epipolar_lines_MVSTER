#!/usr/bin/env bash
# 
# Train MVSTER (Multi-GPU training):
# -Train with middle size (512x640): bash ./scripts/train_dtu.sh  experiment_name
# 
# Note: 
#   bash ./scripts/train_BDS7.sh  BDS7_512x640_rt20pct_10srcs_N5_L10.005
#   bash ./scripts/train_BDS7.sh  newBDS7_512x640_Nviews5_rt20pct10srcs


TRAIN_DATASET="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS7_mvs_training_512x640"
TRAINLIST="lists/BDS7/train.txt"
TESTLIST="lists/BDS7/test.txt"
PAIRFILE="pair_49x15.txt"

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
--train_nviews 5 \
--interval_scale=1.55 \
--ndepths="8,8,4,4" \
--depth_inter_r="0.5,0.5,0.5,1" \
--epochs=24 \
--lr=0.001 \
--wd=0.0001 \
--l1ce_lw="0.003,1" \
--lrepochs="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24:1.2" \
--batch_size=6 \
--summary_freq 100 \
--group_cor \
--group_cor_dim="8,8,4,4" \
--rt \
--mono \
--inverse_depth \
--attn_temp 2 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR"/"$LOG_FILE 


# --pos_enc 2 \
# --ot_continous \
# --mono \
# --use_raw_train \
# --resume \
