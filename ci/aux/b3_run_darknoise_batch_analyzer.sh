#!/usr/bin/env bash

# Catch the path to the YAML parameters files
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_darknoise_YAML>"
  exit 1
fi

# Execute the notebook 
papermill \
  apps/darknoise_batch_analyzer.ipynb \
  apps/darknoise_batch_analyzer_executed.ipynb \
  -p input_parameters_filepath "$1"