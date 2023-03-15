#!/usr/bin/env bash
# MVSTER
#   bash ./scripts/test_BDS4.sh dtu_pretrained BDS4_1024x1280
#   bash ./scripts/test_BDS4.sh DTU_512x640_Nviews5 BDS4_1024x1280/
#   bash ./scripts/test_BDS4.sh BDS4_512x640_rt20pct_10srcs BDS4_1024x1280

#   bash ./scripts/test_BDS4.sh BDS4_512x640_rt20pct_10srcs BDS4_TEST2_1024x1280


run_folder=$1
experiment=$2
PY_ARGS=${@:3}

TESTPATH="/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/Blender/BDS4_mvs_training_1024x1280"

TESTLIST="lists/BDS4/test_only1.txt"


# CKPT_FILE="./outputs/dtu_pretrained/pretrained_finalmodel.ckpt" # trained without mono
# CKPT_FILE="./outputs/BDS4_512x640_rt20pct_10srcs/model_10.ckpt"
# CKPT_FILE="./outputs/BDS4_512x640_rt10pct_20srcs/model_14.ckpt"
# CKPT_FILE="./outputs/BDS4_512x640_noMono_itvl1.25_rt10pct/model_15.ckpt"
CKPT_FILE="./outputs/newBDS4_512x640_rt20pct_10srcs/model_21.ckpt"


# PAIRFILE="pair.txt"
# PAIRFILE="pair_eval_40x10.txt"
# PAIRFILE="pair_eval_15-19-23-27.txt"
# PAIRFILE="pair_eval_15-19-23-27_self.txt"
# PAIRFILE="pair_eval_00-02-28-30_self.txt"
# PAIRFILE="pair_eval_15-16-25-26_self.txt"
# PAIRFILE="pair_eval_0-4-16-20_self.txt"
# PAIRFILE="pair_eval_0-2-9-11_self.txt"
# PAIRFILE="pair_eval_2-12-28-35_self.txt"
# PAIRFILE="pair_eval_00-02_self.txt"
# PAIRFILE="pair_eval_28-30-34-36_self.txt"
# PAIRFILE="pair_eval_28-30-37-39_self.txt"

# INFO: depth creation will only use NVIEW views (dataloader caping)
# INFO: filter algo (line379) does not cap views to NVIEW so what ever in pair file will be used
NVIEWS=2
PHOTO_MASK=0.75
GEO_MASK=2
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
--dataset_name=blender \
--batch_size=1 \
--testpath=$TESTPATH \
--testlist=$TESTLIST \
--loadckpt $CKPT_FILE \
--interval_scale 1.44 \
--num_view $NVIEWS \
--outdir $OUTDIR \
--pair_fname=$PAIRFILE \
--thres_view $GEO_MASK \
--conf $PHOTO_MASK \
--group_cor \
--mono \
--save_jpg \
--attn_temp 2 \
--inverse_depth \
--use_raw_train \
--debug 1 \
| tee -a $OUTDIR"/"$LOG_FILE 


# --vis_ETA \
# --save_jpg \
# --mono \ 


# --interval_scale 1.44 \ # BDS4
# --interval_scale 1.06 \ # DTU
