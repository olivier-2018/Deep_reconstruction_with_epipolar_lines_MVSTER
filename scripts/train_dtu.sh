#!/usr/bin/env bash
# 
# bash ./scripts/train_dtu.sh DTU_512x640_Nviews5
# bash ./scripts/train_dtu.sh DTU_512x640_Nviews5_v2

TRAIN_DATASET="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/DTU/mvs_training_512x640"
TRAINLIST="lists/dtu/train.txt"
TESTLIST="lists/dtu/test.txt"

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
--dataset=dtu_yao4 \
--train_nviews 5 \
--interval_scale=1.0625 \
--epochs=22 \
--batch_size=6 \
--l1ce_lw="0.003,1" \
--wd=0.0001 \
--lr=0.001 \
--lrepochs="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21:1.2" \
--summary_freq 100 \
--group_cor \
--group_cor_dim="8,8,4,4" \
--ndepths="8,8,4,4" \
--depth_inter_r="0.5,0.5,0.5,1" \
--rt \
--mono \
--mono_stg_itrpl="nearest" \
--inverse_depth \
--attn_temp 2 \
--seed 0 \
--pos_enc 2 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR"/"$LOG_FILE 

# 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16:1.2"
# --pos_enc 2 \
# --ot_continous \
# --mono \
# --use_raw_train \
# --resume \


# --depth_inter_r="0.5,0.5,0.5,1" \
# --ndepths="8,8,4,4" \
# --group_cor_dim="8,8,4,4" \