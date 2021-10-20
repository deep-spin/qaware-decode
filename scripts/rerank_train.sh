#!/bin/bash
set -e

if [[ -z $TRAVATAR_DIR ]]; then
  echo "\$TRAVATAR_DIR enviromental variable needs to be set"
  exit 1
fi

weight_in=$1
valid_feat=$2
valid_target=$3

# tune rerank weights on validation set
$TRAVATAR_DIR/src/bin/batch-tune \
  -nbest $valid_feat \
  -rand_seed 9 \
  -algorithm mert -weight_in $weight_in \
  -eval bleu \
  -restarts 100 \
  -threads 4 \
  -debug 2 \
  $valid_target 