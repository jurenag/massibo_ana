#!/usr/bin/env bash

# Define papermill parameters
PARAM="apps/parameters_examples/gain_input_parameters.yml"

# Execute the notebook 
papermill \
  apps/gain_batch_analyzer.ipynb \
  apps/gain_batch_analyzer_executed.ipynb \
  -p input_parameters_filepath "$PARAM"