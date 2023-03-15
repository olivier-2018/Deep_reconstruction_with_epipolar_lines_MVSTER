#!/usr/bin/env bash
# 
# Train MVSTER (Multi-GPU training):
# -Train with middle size (512x640): bash ./scripts/train_dtu.sh  experiment_name
# 
# Note: 

#   bash ./scripts/train_BDS2.sh  BDS2_512x640_rt20pct_10srcs
#   bash ./scripts/train_BDS2.sh  newBDS2_512x640_rt20pct_10srcs


TRAIN_DATASET="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS2_mvs_training_512x640"
TRAINLIST="lists/BDS2/train.txt"
TESTLIST="lists/BDS2/test.txt"
PAIRFILE="pair_48x10.txt"
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
--lightings=-3 \
--pair_fname=$PAIRFILE \
--train_nviews 5 \
--interval_scale=1.45 \
--ndepths="8,8,4,4" \
--depth_inter_r="0.5,0.5,0.5,1" \
--epochs=24 \
--lr=0.002 \
--wd=0.0001 \
--l1ce_lw="0.003,1" \
--lrepochs="4,6,8,10,12,14,16,18,20,22:2" \
--batch_size=6 \
--summary_freq 100 \
--group_cor \
--group_cor_dim="8,8,4,4" \
--rt \
--mono \
--inverse_depth \
--attn_temp 2 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

# 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16:1.2"
# --pos_enc 2 \
# --ot_continous \
# --mono \
# --use_raw_train \
# --resume \


# --depth_inter_r="0.5,0.5,0.5,1" \
# --ndepths="8,8,4,4" \
# --group_cor_dim="8,8,4,4" \