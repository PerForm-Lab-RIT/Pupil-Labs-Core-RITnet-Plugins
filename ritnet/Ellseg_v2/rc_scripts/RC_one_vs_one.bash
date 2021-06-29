#!/bin/bash -l

path_data='/home/rsk3900/Datasets'
epochs=80
workers=8
lr=0.001

declare -a CUR_OBJ_LIST=("OpenEDS" "NVGaze" "Fuhl" "Santini" "LPW" "UnityEyes" "Swirski" "riteyes-s-general" "riteyes-s-natural")
declare -a PRETRAINED_LIST=(0 0 0 0 0 0 0 0 0)
MODE='one_vs_one'
MIXED_PREC=0
BATCH_SIZE=24

IN_NORM=1
GRAD_REV=0
AUG_FLAG=0
ADA_IN_NORM=0
GROWTH_RATE=1.2
MAKE_UNCERTAIN=0

for i in "${!CUR_OBJ_LIST[@]}";
do
    PRETRAINED="${PRETRAINED_LIST[$i]}"
    CUR_OBJ="${CUR_OBJ_LIST[$i]}"
    EXP_NAME="GR-${GROWTH_RATE}_one_vs_one_${CUR_OBJ}_AUG-${AUG_FLAG}_IN_NORM-${IN_NORM}"
    runCMD="bash ../launch_RC.bash --BATCH_SIZE ${BATCH_SIZE} --MODE ${MODE} --CUR_OBJ ${CUR_OBJ} --ADA_IN_NORM ${ADA_IN_NORM} "
    runCMD+="--PRETRAINED ${PRETRAINED} --EXP_NAME ${EXP_NAME} --GPU_TYPE v100:1 --EPOCHS ${epochs} --GROWTH_RATE ${GROWTH_RATE} "
    runCMD+="--MAKE_UNCERTAIN ${MAKE_UNCERTAIN} --MIXED_PREC ${MIXED_PREC} --AUG_FLAG ${AUG_FLAG} --IN_NORM ${IN_NORM}"
    echo $runCMD
    eval $runCMD
done
