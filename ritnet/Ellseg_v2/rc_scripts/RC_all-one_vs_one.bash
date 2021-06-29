#!/bin/bash -l

epochs=80
workers=8
lr=0.001

declare -a CUR_OBJ_LIST=("OpenEDS" "NVGaze" "Fuhl" "Santini" "LPW" "UnityEyes" "Swirski" "riteyes-s-general" "riteyes-s-natural")
declare -a PRETRAINED_LIST=(0 0 0 0 0 0 0 0 0)
MODE='all-one_vs_one'
MIXED_PREC=0
BATCH_SIZE=24

ADV_DG=0
IN_NORM=1
FRN_TLU=0
GRAD_REV=0
AUG_FLAG=0
GROWTH_RATE=1.2
ADA_IN_NORM=0
PSEUDO_LABELS=0
MAKE_UNCERTAIN=0

for i in "${!CUR_OBJ_LIST[@]}";
do
    PRETRAINED="${PRETRAINED_LIST[$i]}"
    CUR_OBJ="${CUR_OBJ_LIST[$i]}"
    EXP_NAME="GR-${GROWTH_RATE}_all-one_vs_one_${CUR_OBJ}_AUG-${AUG_FLAG}_IN_NORM-${IN_NORM}_ADV_DG-${ADV_DG}_PSEUDO_LABELS-${PSEUDO_LABELS}"
    runCMD="bash ../launch_RC.bash --BATCH_SIZE ${BATCH_SIZE} --MODE ${MODE} --CUR_OBJ ${CUR_OBJ} "
    runCMD+="--PRETRAINED ${PRETRAINED} --EXP_NAME ${EXP_NAME} --GPU_TYPE a100:1 --ADA_IN_NORM ${ADA_IN_NORM} "
    runCMD+="--EPOCHS ${epochs} --MIXED_PREC ${MIXED_PREC} --AUG_FLAG ${AUG_FLAG} --GROWTH_RATE ${GROWTH_RATE} "
    runCMD+="--GRAD_REV ${GRAD_REV} --MAKE_UNCERTAIN ${MAKE_UNCERTAIN} --FRN_TLU ${FRN_TLU} --IN_NORM ${IN_NORM} "
    runCMD+="--ADV_DG ${ADV_DG} --PSEUDO_LABELS ${PSEUDO_LABELS}"
    echo $runCMD
    eval $runCMD
done
