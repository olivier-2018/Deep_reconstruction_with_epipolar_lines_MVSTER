#!/usr/bin/env bash
#
# -Test with middle size: bash ./scripts/test_dtu.sh mid exp_name
# -Test with raw size: bash ./scripts/test_dtu.sh raw exp_name
# 
# Ex: 
#   bash ./scripts/test_dtu.sh raw dtu_rerun_raw_1200x1600
#   bash ./scripts/test_BDS4.sh raw BDS4_raw_1024x1280
#   bash ./scripts/test_BDS4.sh mid BDS4_raw_512x640
#   bash ./scripts/test_BDS4.sh BDS4_raw_1024x1280_photo0.8_geo5_consist1pix_40cams_10views
#   bash ./scripts/test_BDS4.sh BDS4_raw_1024x1280_photo0.5_geo1_consist1pix_4cams_4views
#   bash ./scripts/test_BDS4.sh BDS4_raw_1024x1280_photo0.5_geo2_consist1pix_4cams_4views
#   bash ./scripts/test_BDS4.sh BDS4_raw_1024x1280_photo0.5_geo3_consist1pix_4cams_4views+self_DTUweights
#   bash ./scripts/test_BDS4.sh BDS4_raw_1024x1280_photo0.5_geo2_consist1pix_4cams_4views+self

exp=$1
PY_ARGS=${@:2}

TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS4_mvs_training_1024x1280"
TESTLIST="lists/blender/test_only1.txt"


# CKPT_FILE="./outputs/dtu_pretrained/pretrained_finalmodel.ckpt"
CKPT_FILE="./outputs/BDS4_512x640_rt20pct_10srcs/model_10.ckpt"
# CKPT_FILE="./outputs/BDS4_512x640_rt10pct_20srcs/model_14.ckpt"
# CKPT_FILE="./outputs/BDS4_512x640_noMono_itvl1.25_rt10pct/model_15.ckpt"

OUTPUT_DIR="./outputs/BDS4_512x640_rt20pct_10srcs/"$exp


# PAIR_FNAME="pair_eval_40x10.txt"
PAIR_FNAME="pair_eval_4x4_15-19-23-27.txt" # corrected
# PAIR_FNAME="pair_eval_4x4_15-19-23-27_self.txt"
# PAIR_FNAME="pair_eval_5x5_00-03-11-14-32.txt"

PHOTO_MASK=0.5
GEO_MASK=2
NUMVIEWS=4

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "=== Creating log dir: "$OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
fi
LOG_FILE="log_"$exp".txt"
echo "=== Check log in file: tail -f  ${OUTPUT_DIR}/${LOG_FILE}"


python test_mvs4.py \
--dataset=blender4_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE \
--interval_scale 1.44 \
--num_view $NUMVIEWS \
--outdir $OUTPUT_DIR \
--pair_fname=$PAIR_FNAME \
--thres_view $GEO_MASK \
--conf $PHOTO_MASK \
--mono \
--group_cor \
--attn_temp 2 \
--inverse_depth \
--use_raw_train 


# --vis_ETA \
# --save_jpg \
# --mono \


# --interval_scale 1.44 \ # BDS4
# --interval_scale 1.06 \ # DTU
