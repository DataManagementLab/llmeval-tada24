#!/bin/bash

set -e

limit_instances=null

api_name="openai"

limit_instances=500

use_inst_all_column_types=false
num_inst_all_column_types=0

datasets=("sportstables" "gittablesCTA")
models=("gpt-3.5-turbo-1106" "gpt-4-0613")

######################################################################################################################
# run main experiments
######################################################################################################################

for dataset in "${datasets[@]}"; do

  for model in "${models[@]}"; do

    exp_name="$model-with-headers"
    params="exp_name=$exp_name dataset=$dataset limit_instances=$limit_instances api_name=$api_name model=$model use_inst_all_column_types=$use_inst_all_column_types num_inst_all_column_types=$num_inst_all_column_types"
    python scripts/column_type_inference/$dataset/preprocess.py $params
    python scripts/column_type_inference/prepare_requests.py $params
    python scripts/execute_requests.py -cp "../config/column_type_inference" $params
    python scripts/column_type_inference/evaluate.py exp_name=$exp_name $params
    python scripts/column_type_inference/plot.py exp_name=$exp_name $params

    exp_name="$model-without-headers"
    params="exp_name=$exp_name dataset=$dataset limit_instances=$limit_instances api_name=$api_name model=$model use_inst_all_column_types=$use_inst_all_column_types num_inst_all_column_types=$num_inst_all_column_types"
    python scripts/column_type_inference/$dataset/preprocess.py $params
    python scripts/column_type_inference/prepare_requests.py $params 'linearize_table.template="{{table}}"' linearize_table.csv_params.header=false
    python scripts/execute_requests.py -cp "../config/column_type_inference" $params
    python scripts/column_type_inference/evaluate.py $params
    python scripts/column_type_inference/plot.py $params

  done
done