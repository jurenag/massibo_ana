#!/usr/bin/env bash

# Check that the current working directory is the repository root folder
if [[ "$(basename "$PWD")" != "massibo_ana" ]]; then
  echo "Error: the current working directory is not 'massibo_ana'."
  exit 1
fi