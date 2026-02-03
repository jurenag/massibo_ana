#!/usr/bin/env bash

# Define papermill parameters
PARAM="apps/parameters_examples/darknoise_input_parameters.yml"

# Execute the notebook 
papermill \
  apps/darknoise_batch_analyzer.ipynb \
  apps/darknoise_batch_analyzer_executed.ipynb \
  -p input_parameters_filepath "$PARAM"