#!/bin/bash

set -e

datasets=("gittablesCTA" "sportstables" "sotab" "wikitables-turl")
for dataset in "${datasets[@]}"; do
  if [ "$dataset" == "wikitables-turl" ]; then
    if [ ! -d "data/column_type_inference/wikitables-turl/download" ]; then
        echo "You must manually obtain \`data/column_type_inference/wikitables-turl/download\` (see README.md)!"
        exit
    fi
  else
    python scripts/column_type_inference/$dataset/download.py dataset=$dataset
  fi
done