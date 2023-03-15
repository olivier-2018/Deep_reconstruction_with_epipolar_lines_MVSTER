#!/usr/bin/env bash
# 
# Train MVSTER (Multi-GPU training):
# -Train with middle size (512x640): bash ./scripts/train_dtu.sh  experiment_name
# 
# Note: 
#   bash ./scripts/train_BDS4.sh  BDS4_512x640_rt20pct_10srcs_v2
#   bash ./scripts/train_BDS4.sh  BDS4_512x640_rt20pct_10srcs_mse
#   bash ./scripts/train_BDS4.sh  BDS4_512x640_rt20pct_10srcs_L1ce1
#   bash ./scripts/train_BDS4.sh  BDS4_512x640_rt20pct_10srcs_depths-32-16-8-8
#   bash ./scripts/train_BDS4.sh  BDS4_512x640_rt20pct_10srcs_depths-8-8-4-8
#   bash ./scripts/train_BDS4.sh  BDS4_512x640_rt20pct_10srcs_v2
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_trainMono
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_V2_triLR # not working
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_posenc2
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_posenc2_V2
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_posenc2_monoBiLin
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_posenc2_V3_LR0.0001
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_posenc2_V4_LR0.01
#   bash ./scripts/train_BDS4.sh  newBDS4_512x640_rt20pct_10srcs_posenc2_OTi20
#   bash ./scripts/train_BDS4.sh  test


# "/data/3Dreconstruction/MVS_datasets/Blender/BDS4_mvs_training_1024x1280"
TRAIN_DATASET="../datasets/Blender/BDS4_mvs_training_1024x1280"
TRAINLIST="lists/BDS4/train.txt"
TESTLIST="lists/BDS4/test.txt"

# CHKPT="outputs/newBDS4_512x640_rt20pct_10srcs_V2_triLR/model_15.ckpt"

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
--interval_scale=1.45 \
--epochs=24 \
--batch_size=6 \
--wd=0.0001 \
--l1ce_lw="0.003,1" \
--lr_scheduler="MS" \
--lr=0.001 \
--lrepochs="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23:1.2" \
--summary_freq 100 \
--group_cor \
--group_cor_dim="8,8,4,4" \
--ndepths="8,8,4,4" \
--depth_inter_r="0.5,0.5,0.5,1" \
--rt \
--mono \
--inverse_depth \
--attn_temp 2 \
--seed 0 \
--pos_enc 2 \
--ot_eps 0.1 \
# $PY_ARGS &> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR"/"$LOG_FILE 

# --ot_continous \
# 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16:1.2"
# --pos_enc 2 \
# --ot_continous \
# --mono \
# --use_raw_train \
# --resume \
# --loadckpt $CHKPT \CHKPT="outputs/newBDS4_512x640_rt20pct_10srcs_V2_triLR/model_15.ckpt"
# --mono_stg_itrpl="bilinear" \
# --ot_iter 10 \

# --depth_inter_r="0.5,0.5,0.5,1" \
# --ndepths="8,8,4,4" \
# --group_cor_dim="8,8,4,4" \