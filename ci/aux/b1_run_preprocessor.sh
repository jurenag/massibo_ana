#!/usr/bin/env bash

# Define papermill parameters
PARAM="apps/parameters_examples/preprocessor_parameters.yml"

# Execute the notebook
papermill \
  apps/preprocessor.ipynb \
  apps/preprocessor_executed.ipynb \
  -p input_parameters_filepath "$PARAM"