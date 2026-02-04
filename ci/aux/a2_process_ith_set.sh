#!/usr/bin/env bash

# Catch the number of the set to process and the
# paths to the YAML parameters files for the dark
# noise and gain analyses
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <integer> <path_to_darknoise_YAML> <path_to_gain_YAML>"
  exit 1
fi

SET_TO_PROCESS="$1"
DARKNOISE_YAML_PATH="$2"
GAIN_YAML_PATH="$3"

# Validate that the first argument is a positive integer (digits only)
if ! [[ "$SET_TO_PROCESS" =~ ^[0-9]+$ ]]; then
  echo "Error: first argument must be a natural number"
  exit 1
fi

# File existence should have been already checked in the main script

./ci/aux/b0_download_data.sh "$SET_TO_PROCESS"
./ci/aux/b1_run_preprocessor.sh
./ci/aux/b2_configure_YAMLs.sh "$SET_TO_PROCESS" "$DARKNOISE_YAML_PATH" "$GAIN_YAML_PATH"
./ci/aux/b3_run_darknoise_batch_analyzer.sh "$DARKNOISE_YAML_PATH"
./ci/aux/b4_run_gain_batch_analyzer.sh "$GAIN_YAML_PATH"