#!/usr/bin/env bash

# Catch the number of the set to download
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <integer>"
  exit 1
fi

SET_TO_DOWNLOAD="$1"

# Download the data
# - Recursive download
# - Avoid recursive search in parent directories
# - Do not create the host directory (massibo.web.cern.ch/...)
# - Remove the first directory level (clean_set_X/)
# - Download to ci_workspace/input_data
# - Do not download index.html files
wget \
  -r \
  -np \
  -nH \
  --cut-dirs=1 \
  -P ci_workspace/input_data \
  --reject="index.html*" \
  -e robots=off \
  https://massibo.web.cern.ch/clean_set_${SET_TO_DOWNLOAD}/