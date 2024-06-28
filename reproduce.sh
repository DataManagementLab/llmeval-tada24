#!/bin/bash

set -e

if ! command -v conda >/dev/null; then
    echo "You must have Conda installed (see README.md)!"
fi

if [ ! -d "data/openai_cache" ]; then
    echo "You must manually obtain \`data/openai_cache\` (see README.md)!"
    exit
fi

if [ ! -d "data/column_type_inference/sportstables/download" ]; then
    echo "You must manually obtain \`data/column_type_inference/sportstables/download\` (see README.md)!"
    exit
fi

if [ ! -d "data/column_type_inference/gittablesCTA/download" ]; then
    echo "You must manually obtain \`data/column_type_inference/gittablesCTA/download\` (see README.md)!"
    exit
fi

if [ ! -d "data/column_type_inference/sotab/download" ]; then
   echo "You must manually obtain \`data/column_type_inference/sotab/download\` (see README.md)!"
   exit
fi

if [ ! -d "data/column_type_inference/wikitables-turl/download" ]; then TODO
    echo "You must manually obtain \`data/column_type_inference/wikitables-turl/download\` (see README.md)!"
    exit
fi

if [ ! -d "data/column_type_inference/sapdata" ]; then
    echo "You must manually obtain \`data/column_type_inference/sapdata\` (see README.md)!"
    exit
fi

# commented out since we want to use the prepared versions (artifacts) of the datasets
# bash scripts/column_type_inference/download_tuda.sh
# bash scripts/column_type_inference/download_sap.sh  # requires redacted code

bash scripts/analyze_datasets/analyze_tuda.sh
bash scripts/analyze_datasets/analyze_sap.sh

bash scripts/column_type_inference/experiments_tuda.sh
# bash scripts/column_type_inference/experiments_sap.sh  # requires redacted code

bash scripts/column_type_inference/gather_all.sh
bash scripts/column_type_inference/gather_sap.sh
