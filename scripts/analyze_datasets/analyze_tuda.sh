#!/bin/bash

set -e

python scripts/analyze_datasets/analyze_sportstables.py
python scripts/analyze_datasets/analyze_gittables.py
python scripts/analyze_datasets/analyze_sotab.py
python scripts/analyze_datasets/analyze_wikitables-turl.py