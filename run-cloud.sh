#!/bin/bash

# project name: reco-sys-capstone

# run command:
# ./run-cloud.sh --work-dir gs://ahsu-movielens

set -e

# Parse command line arguments
unset WORK_DIR
PROJECT=$(gcloud config get-value project || echo $PROJECT)
REGION=us-central1

while [[ $# -gt 0 ]]; do
  case $1 in
    --work-dir)
      WORK_DIR=$2
      shift
      ;;
    --project)
      PROJECT=$2
      shift
      ;;
    --region)
      REGION=$2
      shift
      ;;
    *)
      echo "error: unrecognized argument $1"
      exit 1
      ;;
  esac
  shift
done

if [[ -z $WORK_DIR ]]; then
  echo "error: argument --work-dir is required"
  exit 1
fi

if [[ $WORK_DIR != gs://* ]]; then
  echo "error: --work-dir must be a Google Cloud Storage path"
  echo "       example: gs://your-bucket"
  exit 1
fi

if [[ -z $PROJECT ]]; then
  echo 'error: --project is required to run in Google Cloud Platform.'
  exit 1
fi

# Wrapper function to print the command being run
function run {
  echo "$ $@"
  "$@"
}

# Extract the data files
echo '>> Extracting data'
run python data-extractor.py \
  --work-dir $WORK_DIR
echo ''

