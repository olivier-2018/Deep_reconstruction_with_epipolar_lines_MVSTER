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
#   bash ./scripts/train_BDS6.sh  newBDS6_512x640_N5_itvl1.05_rt20pct10srcs_posenc2


TRAIN_DATASET="data/Blender/BDS6_mvs_train_1024x1280"
TRAINLIST="lists/BDS6/train.txt"
TESTLIST="lists/BDS6/test.txt"
PAIRFILE="pair_25x10.txt"

CHKPT="outputs/newBDS6_512x640_Nviews5_rt20pct10srcs_posenc2/model_23.ckpt"

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
--loadckpt $CHKPT \
--dataset=blender4 \
--train_nviews 5 \
--test_nviews 5 \
--Nlights="4:7" \
--interval_scale=1.05 \
--dataloader_workers=4 \
--pin_m \
--ndepths="8,8,4,4" \
--depth_inter_r="0.5,0.5,0.5,1" \
--epochs=32 \
--lr=0.001 \
--wd=0.0001 \
--l1ce_lw="0.003,1" \
--lrepochs="2,3,4,5,6,7,8,9,10,12,14,16,20,24,28,32:1.2" \
--batch_size=6 \
--summary_freq 100 \
--group_cor \
--group_cor_dim="8,8,4,4" \
--rt \
--mono \
--inverse_depth \
--attn_temp 2 \
--pos_enc 2 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR"/"$LOG_FILE 


# --loadckpt $CHKPT \
# --dcn \
# --mono \
# --use_raw_train \
# --resume \