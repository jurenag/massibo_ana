#!/usr/bin/env bash

# Catch the number of the set to download
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <integer>"
  exit 1
fi

SET_TO_DOWNLOAD="$1"

# Navigate to the input_data folder
cd ci_workspace/input_data

# Download the data
# wget -r --reject="index.html*" -e robots=off https://massibo.web.cern.ch/clean_set_${SET_TO_DOWNLOAD}/
# For now, just copy from a local path
cp -r /Users/jurenag/cernbox/www/massibo/clean_set_${SET_TO_DOWNLOAD}/ .

# Navigate back to the repository root folder
cd ../..