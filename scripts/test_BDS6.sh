#!/usr/bin/env bash
#
# -Test with middle size: bash ./scripts/test_dtu.sh mid exp_name
# -Test with raw size: bash ./scripts/test_dtu.sh raw exp_name
# 
# Ex: 
#   bash ./scripts/train.sh <weights_folder> <test_img_folder>
#
#   bash ./scripts/test_BDS6.sh   BDS6_1024x1280_Nviews5    BDS6
#   bash ./scripts/test_BDS6.sh   BDS6_1024x1280_Nviews5_10pctL1    BDS6
#   bash ./scripts/test_BDS6.sh   BDS6_1024x1280_Nviews2    BDS6


run_folder=$1
experiment=$2
PY_ARGS=${@:3}

TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS6_mvs_training_1024x1280"

TESTLIST="lists/BDS6/test_only2.txt"


# CKPT_FILE="./outputs/BDS6_512x640_rt10pct_Nviews2/model_23.ckpt"
# CKPT_FILE="./outputs/BDS6_512x640_Nviews5_rt10pct/model_15.ckpt"
CKPT_FILE="./outputs/BDS6_512x640_rt10pct_Nviews5_10pctL1/model_37.ckpt"

PAIRFILE="pair.txt"
# PAIRFILE="pair_0-2-4.txt"
# PAIRFILE="pair_0-4-20-24.txt"


NVIEWS=5
PHOTO_MASK=0.75
GEO_MASK=3
GEO_PIX=1
GEO_DEPTH=0.01
LIGHT_IDX=-3

run_experiment=$experiment"_NViews"$NVIEWS"_PhotoMask"$PHOTO_MASK"_GeoMask"$GEO_MASK"_GeoPix"$GEO_PIX"_GeoDepth"$GEO_DEPTH"_LightIdx"$LIGHT_IDX"_"$PAIRFILE

OUTDIR="./outputs/"$run_folder"/"$run_experiment

if [ ! -d "$OUTDIR" ]; then
    echo "=== Creating log dir: "$OUTDIR
    mkdir -p $OUTDIR
fi
LOG_FILE="log_"$experiment".txt"
echo "=== Check log in file: tail -f  ${OUTDIR}/${LOG_FILE}"


python test_mvs4.py \
--dataset=blender4_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE \
--interval_scale 1.44 \
--num_view $NVIEWS \
--outdir $OUTDIR \
--pair_fname=$PAIRFILE \
--thres_view $GEO_MASK \
--conf $PHOTO_MASK \
--mono \
--group_cor \
--attn_temp 2 \
--inverse_depth \
--use_raw_train \
| tee -a $OUTDIR"/"$LOG_FILE 


# --vis_ETA \
# --save_jpg \
# --mono \ 


# --interval_scale 1.44 \ # BDS4
# --interval_scale 1.06 \ # DTU
