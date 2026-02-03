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

# Write the set number into the YAML configuration files
# (look for the line which starts with 'set_to_analyze', 
# and replace it with 'set_to_analyze: <SET_TO_PROCESS>')
sed -i.bak "s/^set_to_analyze.*/set_to_analyze: $SET_TO_PROCESS/" "$DARKNOISE_YAML_PATH"
sed -i.bak "s/^set_to_analyze.*/set_to_analyze: $SET_TO_PROCESS/" "$GAIN_YAML_PATH"

# Do the same with the output files so that they are not
# overwritten by subsequent runs
sed -i.bak "s/^report_output_filename.*/report_output_filename: 'darknoise_report_set_$SET_TO_PROCESS.pdf'/" "$DARKNOISE_YAML_PATH"
sed -i.bak "s/^report_output_filename.*/report_output_filename: 'gain_report_set_$SET_TO_PROCESS.pdf'/" "$GAIN_YAML_PATH"

# Yes, no extension here
sed -i.bak "s/^output_dataframe_filename.*/output_dataframe_filename: 'darknoise_output_dataframe_set_$SET_TO_PROCESS'/" "$DARKNOISE_YAML_PATH"
sed -i.bak "s/^output_dataframe_filename.*/output_dataframe_filename: 'gain_output_dataframe_set_$SET_TO_PROCESS'/" "$GAIN_YAML_PATH"