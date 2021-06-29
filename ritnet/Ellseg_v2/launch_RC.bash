#!/bin/bash

# Load the environment
spack env activate riteyes_distributed-21031501

# RC parameters:
USERNAME=${USERNAME:-"rsk3900"} # Local username

SEED=${SEED:-0}
EPOCHS=${EPOCHS:-2}
AUG_FLAG=${AUG_FLAG:-0}
LR_DECAY=${LR_DECAY:-0}
MIXED_PREC=${MIXED_PREC:-0}
EARLY_STOP=${EARLY_STOP:-15}
ITR_LIMIT_PER_EPOCH=${ITR_LIMIT_PER_EPOCH:-2000}
NUM_VALID_PER_EPOCH=${NUM_VALID_PER_EPOCH:-400}
WORKERS=${WORKERS:-6}

DO_DISTRIBUTED=${DO_DISTRIBUTED:-0}

############################

GPU_TYPE=${GPU_TYPE:-"v100:1"}
EXP_NAME=${EXP_NAME:-"dev"}

############################

LEARNING_RATE=${LEARNING_RATE:-5e-4} #
BATCH_SIZE=${BATCH_SIZE:-40} #
PRETRAINED=${PRETRAINED:-0}
CLIP_NORM=${CLIP_NORM:-0}
EPOCHS=${EPOCHS:-50}

ADA_IN_NORM=${ADA_IN_NORM:-0}
GROWTH_RATE=${GROWTH_RATE:-1.2}
PSEUDO_LABELS=${PSEUDO_LABELS:-0}
GRAD_REV=${GRAD_REV:-0}
USE_SCSE=${USE_SCSE:-0}
FRN_TLU=${FRN_TLU:-0}
IN_NORM=${IN_NORM:-0}
AMSGRAD=${AMSGRAD:-1}
DROPOUT=${DROPOUT:-0}
ADV_DG=${ADV_DG:-0}

MAKE_UNCERTAIN=${MAKE_UNCERTAIN:-0}
MAKE_ALEATORIC=${MAKE_ALEATORIC:-0}
REGRESS_FROM_LATENT=${REGRESS_FROM_LATENT:-1}
MAXPOOL_IN_REGRESS_MOD=${MAXPOOL_IN_REGRESS_MOD:-0}

############################

MODE=${MODE:-"one_vs_one"}
CUR_OBJ=${CUR_OBJ:-"OpenEDS"}

############################

# PARSE ARGUMENTS - OVERWRITE DEFAULT VALUES
while [ $# -gt 0 ]; do # If input arguments provided, enter the loop

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}" # Splice out the flag name
        declare $param=$2 # Assign the value to the flag name
   fi

  shift # Read the next argument
done

############################

# NOTE: please make sure to update the two paths below as necessary.
LOCAL_SRC_DIR="/home/rsk3900/multiset_gaze/src" # local sync point of git repo
OUTPUT_DIR_STEM="/home/rsk3900/Results"         # directory where results are stored
PATH_DATA="/scratch/multiset/All"               # directory where data is stored

############################

# run training:
# Prelim commands
# Add this command before python call to track RAM consumption for
# analysis with behavior on RC.
# <mprof run --include-children --multiprocess --python python run.py ...>
# RUN_CMD+="python -m torch.distributed.launch --nproc_per_node=2 \

RUN_CMD+="python \
         ${LOCAL_SRC_DIR}/run.py \
         --lr=${LEARNING_RATE} \
         --seed=${SEED} \
         --batch_size=${BATCH_SIZE} \
         --lr_decay=${LR_DECAY} \
         --dropout=${DROPOUT} \
         --grad_rev=${GRAD_REV} \
         --path_data=${PATH_DATA} \
         --path_exp_tree=${OUTPUT_DIR_STEM} \
         --exp_name=${EXP_NAME} \
         --mode=${MODE} \
         --epochs=${EPOCHS} \
         --growth_rate=${GROWTH_RATE} \
         --cur_obj=${CUR_OBJ} \
         --aug_flag=${AUG_FLAG} \
         --use_scSE=${USE_SCSE} \
         --use_frn_tlu=${FRN_TLU} \
         --equi_var=0 \
         --adv_DG=${ADV_DG} \
         --early_stop=${EARLY_STOP} \
         --mixed_precision=${MIXED_PREC} \
         --use_instance_norm=${IN_NORM} \
         --use_ada_instance_norm=${ADA_IN_NORM} \
         --batches_per_ep=${ITR_LIMIT_PER_EPOCH} \
         --make_aleatoric=${MAKE_ALEATORIC} \
         --make_uncertain=${MAKE_UNCERTAIN} \
         --pseudo_labels=${PSEUDO_LABELS} \
         --regression_from_latent=${REGRESS_FROM_LATENT} \
         --pretrained=${PRETRAINED} \
         --workers=${WORKERS} \
         --do_distributed=${DO_DISTRIBUTED} \
         --grad_clip_norm=${CLIP_NORM} \
         --repo_root=${LOCAL_SRC_DIR} \
         --reduce_valid_samples=${NUM_VALID_PER_EPOCH} \
         --maxpool_in_regress_mod=${MAXPOOL_IN_REGRESS_MOD} \
         "
RUN_CMD+="; "

RUN_CMD=$(echo "$RUN_CMD" | tr -s ' ' | tr  -d '\n' | tr -d '\t')

############################

# Generate run command

RUN_CMD="#!/bin/bash\n$RUN_CMD"
echo -e $RUN_CMD > command.lock
cat command.lock
RC_CMD="sbatch -J ${EXP_NAME} -o ${LOCAL_SRC_DIR}/rc_logs/${EXP_NAME}_out.o -e ${LOCAL_SRC_DIR}/rc_logs/${EXP_NAME}_error.e "
RC_CMD+="--mem=40G --cpus-per-task=8 -p tier3 --mail-user=$USERNAME@rit.edu --mail-type=FAIL "
RC_CMD+="-A riteyes --gres=gpu:${GPU_TYPE} -t 3-0:0:0 command.lock"
echo $RC_CMD
eval $RC_CMD
