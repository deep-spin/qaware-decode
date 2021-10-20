#!/bin/bash
set -e

if [[ -z $TRAVATAR_DIR ]]; then
  echo "\$TRAVATAR_DIR enviromental variable needs to be set"
  exit 1
fi

valid_feat=$1
valid_target=$2
test_feat=$3
test_hyp=$4
# seed=$1

mkdir -p rerank_models

echo "logprob=1 cometsrc=0.1" > weight_in.txt

# tune rerank weights on validation set
save_model_name=rerank.weights
echo process $valid_feat
$TRAVATAR_DIR/src/bin/batch-tune \
  -nbest $valid_feat \
  -rand_seed 9 \
  -algorithm mert -weight_in weight_in.txt \
  -eval bleu \
  -restarts 100 \
  -threads 4 \
  -debug 2 \
  $valid_target > rerank_models/${save_model_name}

# use tuned weights to decode target
$TRAVATAR_DIR/src/bin/rescorer \
  -nbest $test_feat \
  -weight_in rerank_models/${save_model_name} \
  -nbest_out test.nbest \
  > $test_hyp