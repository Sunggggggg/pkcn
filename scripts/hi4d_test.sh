#!/bin/bash

echo "TEST"
TRAIN_CONFIGS='configs/hi4d_test_hrnet.yml'
# TRAIN_CONFIGS='configs/hi4d_test_dino.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m pkcn.hi4d_demo --configs_yml=${TRAIN_CONFIGS}