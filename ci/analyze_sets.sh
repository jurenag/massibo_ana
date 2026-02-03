#!/usr/bin/env bash

set -e

# Catch the number of the sets to process and the
# paths to the YAML parameters files for the dark
# noise and gain analyses
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <comma-separated-numbers-without-whitespaces> <path_to_darknoise_YAML> <path_to_gain_YAML>"
  exit 1
fi

SETS_TO_PROCESS="$1"
DARKNOISE_YAML_PATH="$2"
GAIN_YAML_PATH="$3"

# Ensure well-formed input by extracting unique numbers
SETS_TO_PROCESS=$(python3 ci/aux/unique_numbers.py "$SETS_TO_PROCESS")

# Check that file paths exist and are files
if [ ! -f "$DARKNOISE_YAML_PATH" ]; then
  echo "Error: file does not exist or is not a regular file: $DARKNOISE_YAML_PATH"
  exit 1
fi

if [ ! -f "$GAIN_YAML_PATH" ]; then
  echo "Error: file does not exist or is not a regular file: $GAIN_YAML_PATH"
  exit 1
fi

IFS=',' read -ra NUM_ARRAY <<< "$SETS_TO_PROCESS"

# Check current working directory and create workspace
./ci/aux/a0_check_cwd.sh
./ci/aux/a1_create_workspace.sh

# Loop over the sets and process them one by one
for i in "${NUM_ARRAY[@]}"; do
    ./ci/aux/a2_process_ith_set.sh "$i" "$DARKNOISE_YAML_PATH" "$GAIN_YAML_PATH"
done