#!/usr/bin/env bash
#
# -Test with middle size: bash ./scripts/test_dtu.sh mid exp_name
# -Test with raw size: bash ./scripts/test_dtu.sh raw exp_name
# 
# Ex: 
#   bash ./scripts/test_dtu.sh raw dtu_rerun_raw_1200x1600

TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/DTU/mvs_testing_1200x1600"
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/DTU/dtu_testing"
# TESTLIST="lists/dtu/test.txt"
TESTLIST="lists/dtu/test_only1.txt"

# CKPT_FILE="./outputs_train/dtu_pretrained/pretrained_finalmodel.ckpt"
CKPT_FILE="./outputs_train/dtu_mid_512x640/model_17.ckpt"



size=$1
exp=$2
PY_ARGS=${@:3}

LOG_DIR="./outputs_eval/"$exp
if [ ! -d "$LOG_DIR" ]; then
    echo "=== Creating log dir: "$LOG_DIR
    mkdir -p $LOG_DIR
fi
LOG_FILE="log_"$exp".txt"
echo "=== Check log in file: tail -f  ${LOG_DIR}/${LOG_FILE}"

if [ $size = "raw" ] ; then
python test_mvs4.py \
--dataset=general_eval4 \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE \
--interval_scale 1.0625 \
--outdir $LOG_DIR \
--use_raw_train \
--thres_view 4 \
--mono \
--conf 0.5 \
--group_cor \
--attn_temp 2 \
--inverse_depth \
$PY_ARGS \
&> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR/log_test.txt
else
python test_mvs4.py \
--dataset=general_eval4 \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE \
--interval_scale 1.0625 \
--outdir $LOG_DIR \
--thres_view 4 \
--mono \
--conf 0.5 \
--group_cor \
--attn_temp 2 \
--inverse_depth \
$PY_ARGS \
&> $LOG_DIR"/"$LOG_FILE &
# $PY_ARGS | tee -a $LOG_DIR/log_test.txt
fi



# --vis_ETA \