#!/usr/bin/env bash
# MVSTER
# -Test with middle size: bash ./scripts/test_dtu.sh mid exp_name
# -Test with raw size: bash ./scripts/test_dtu.sh raw exp_name
# 
# Ex:  bash ./scripts/train.sh <weights_folder> <test_img_folder>
#
#   bash ./scripts/test_merlin.sh dtu_pretrained Merlin_1024x1280_BlenderSet
#   bash ./scripts/test_merlin.sh dtu_pretrained Merlin_1024x1280_Full2Empty
#   bash ./scripts/test_merlin.sh dtu_pretrained Merlin_1024x1280_Mario_GT
#   bash ./scripts/test_merlin.sh dtu_pretrained overhead01_1024x1280_mario_w_GT
#
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 Merlin_1024x1280_BlenderSet
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 Merlin_1024x1280_Full2Empty
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 Merlin_1024x1280_Mario_GT
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 overhead01_1024x1280_mario_w_GT
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 overhead02_1024x1280_mario_w_GT
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 overhead02_1024x1280_Luigi
#   bash ./scripts/test_merlin.sh DTU_512x640_Nviews5 overhead03_1024x1280_mario_w_GT
#
#   bash ./scripts/test_merlin.sh BDS4_512x640_rt20pct_10srcs Merlin_1024x1280_BlenderSet
#   bash ./scripts/test_merlin.sh BDS4_512x640_rt20pct_10srcs Merlin_1024x1280_Full2Empty
#   bash ./scripts/test_merlin.sh BDS4_512x640_rt20pct_10srcs Merlin_1024x1280_Mario_GT
#
#   bash ./scripts/test_merlin.sh newBDS4_512x640_rt20pct_10srcs overhead01_1024x1280_mario_w_GT
#   bash ./scripts/test_merlin.sh newBDS4_512x640_rt20pct_10srcs overhead02_1024x1280_mario_w_GT
# 
# 
#   bash ./scripts/test_merlin.sh newBDS4_512x640_rt20pct_10srcs_v2 overhead01_1024x1280_mario_w_GT
#   bash ./scripts/test_merlin.sh newBDS4_512x640_rt20pct_10srcs_V2 overhead02_1024x1280_mario_w_GT
#



run_folder=$1
experiment=$2
PY_ARGS=${@:3}

# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2022-07-15_setup_Merlin_Mario_Set_Full_to_Empty"
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2022-12-05_setup_Merlin_mario_blender_set"
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2022-09-30_setup_Merlin_Mario_Set_with_GT"
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2023-02-01_setup_overhead01_mario_w_GT"   # scan1-11
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2023-02-01_setup_overhead01_luigi_w_GT" # scan1-12
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2023-02-06_setup_overhead02_mario_w_GT" # scan1-10
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2023-02-06_setup_overhead02_objects1_w_GT" # scan1-7
# TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2023-02-06_setup_overhead02_objects2_w_GT" # scan1-6
TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Merlin/2023-02-15_setup_overhead03_mario_w_GT"  # scan1-5

TESTLIST="lists/Merlin/test_scan1.txt"
# TESTLIST="lists/Merlin/test_scan8.txt"
# TESTLIST="lists/Merlin/test_scan5-7-10.txt"
# TESTLIST="lists/Merlin/test_scan1-3-6.txt"
# TESTLIST="lists/Merlin/test_scan1-6.txt"
# TESTLIST="lists/Merlin/test_scan1-7.txt"
# TESTLIST="lists/Merlin/test_scan1-10.txt"
# TESTLIST="lists/Merlin/test_scan1-11.txt"
# TESTLIST="lists/Merlin/test_scan1-12.txt"

# CKPT_FILE="./outputs/dtu_pretrained/pretrained_finalmodel.ckpt"    # --interval_scale 1.06 \ # DTU
CKPT_FILE="./outputs/DTU_512x640_Nviews5/model_17.ckpt"              # --interval_scale 1.06 \ # DTU
# CKPT_FILE="./outputs/BDS4_512x640_rt20pct_10srcs/model_10.ckpt"      # --interval_scale 1.44   # BDS4
# CKPT_FILE="./outputs/BDS4_512x640_rt10pct_20srcs/model_14.ckpt"
# CKPT_FILE="./outputs/BDS4_512x640_rt20pct_10srcs/model_10.ckpt"      
# CKPT_FILE="./outputs/newBDS4_512x640_rt20pct_10srcs/model_21.ckpt"     # --interval_scale 1.44   # newBDS4
# CKPT_FILE="./outputs/newBDS4_512x640_rt20pct_10srcs_V2/model_15.ckpt"  

# the number
# PAIRFILE="../pair_merlin_4x4.txt"
# PAIRFILE="../pair_merlin_4x3.txt"
# PAIRFILE="../pair_merlin_4x2.txt"
# PAIRFILE="../pair_overhead_4x4.txt"
PAIRFILE="../pair_overhead_4x2.txt"
# PAIRFILE="../pair_overhead_4x3.txt"

NVIEWS=2    # DEPTH GENERATION: nviews will be used from pairfile (this includes ref view)
PHOTO_MASK=0.75
GEO_MASK=2
GEO_PIX=1
GEO_DEPTH=0.01
LIGHT_IDX=-3

run_experiment=$experiment"_NViews"$NVIEWS"_PhotoMask"$PHOTO_MASK"_GeoMask"$GEO_MASK"_GeoPix"$GEO_PIX"_GeoDepth"$GEO_DEPTH"_"${PAIRFILE#"../"}

OUTDIR="./outputs/"$run_folder"/"$run_experiment

if [ ! -d "$OUTDIR" ]; then
    echo "=== Creating log dir: "$OUTDIR
    mkdir -p $OUTDIR
fi
LOG_FILE="log_"$experiment".txt"
echo "=== Check log in file: tail -f  ${OUTDIR}/${LOG_FILE}"


python test_mvs4.py \
--dataset=blender4_eval \
--dataset_name=merlin \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE \
--outdir $OUTDIR \
--pair_fname=$PAIRFILE \
--batch_size=1 \
--interval_scale 1.0625 \
--num_view $NVIEWS \
--thres_view $GEO_MASK \
--conf $PHOTO_MASK \
--group_cor \
--attn_temp 2 \
--inverse_depth \
--use_raw_train \
--debug 1 \
--mono \
| tee -a $OUTDIR"/"$LOG_FILE 


# --vis_ETA \
# --save_jpg \
# --mono \ 
# --debug 1
# --use_raw_train \

# --save_jpg \
# --interval_scale 1.44 \ # BDS4
# --interval_scale 1.06 \ # DTU
