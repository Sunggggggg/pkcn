#!/bin/bash

echo "TRAIN"
TRAIN_CONFIGS='configs/hi4d_train_hrnet.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=1 python -u -m pkcn.fine_tune --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS}