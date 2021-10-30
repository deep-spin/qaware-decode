#!/bin/bash
set -e

if [[ -z $TRAVATAR_DIR ]]; then
  echo "\$TRAVATAR_DIR enviromental variable needs to be set"
  exit 1
fi

weight_in=$1
test_feat=$2
reranked_feat=$3

# use tuned weights to decode target
$TRAVATAR_DIR/src/bin/rescorer \
  -nbest $test_feat \
  -weight_in $weight_in \
  -nbest_out $reranked_feat > /dev/null