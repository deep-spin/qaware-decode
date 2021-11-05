#!/bin/bash
set -e
set -o pipefail
set -u

NUM_SAMPLES=300

WORKFLOW_OUTPUTS=$1
DECODE_METHOD=$2
LANG=$3
OUTPUT_DIR=$4

REFERENCE_DIR=$WORKFLOW_OUTPUTS/GetDevTestData/TargetLang.$LANG
BASELINE=$WORKFLOW_OUTPUTS/PickBestRanked/DecodeType.beam_search+NBest.5+Rerank.false+TargetLang.$LANG+UsePretrained.true
RERANK_FIXED=$WORKFLOW_OUTPUTS/PickBestRanked/DecodeType.$DECODE_METHOD+NBest.200+Rerank.fixed+RerankMetric.cometsrc+TargetLang.$LANG+UsePretrained.true
RERANK_TRAIN=$WORKFLOW_OUTPUTS/PickBestRanked/DecodeType.$DECODE_METHOD+NBest.200+Rerank.train+RerankEval.comet+TargetLang.$LANG+UsePretrained.true
MBR=$WORKFLOW_OUTPUTS/MBRDecoding/DecodeType.$DECODE_METHOD+MBRMetric.comet+MBRSamples.50+NBest.50+Rerank.false+TargetLang.$LANG+UsePretrained.true
RERANK_MBR=$WORKFLOW_OUTPUTS/MBRDecoding/DecodeType.$DECODE_METHOD+MBRMetric.comet+MBRSamples.50+NBest.200+Rerank.train+RerankEval.comet+TargetLang.$LANG+UsePretrained.true

mkdir -p $OUTPUT_DIR

cp $REFERENCE_DIR/test_src    $OUTPUT_DIR/test.src
cp $REFERENCE_DIR/test_tgt    $OUTPUT_DIR/test.tgt.ref
cp $BASELINE/predictions      $OUTPUT_DIR/test.tgt.baseline
cp $RERANK_FIXED/predictions  $OUTPUT_DIR/test.tgt.rr_fixed
cp $RERANK_TRAIN/predictions  $OUTPUT_DIR/test.tgt.rr_train
cp $MBR/predictions           $OUTPUT_DIR/test.tgt.mbr
cp $RERANK_MBR/predictions    $OUTPUT_DIR/test.tgt.rr_mbr

perl syseval-combine.pl                     \
            -src $OUTPUT_DIR/test.src       \
            -ids $OUTPUT_DIR/output.ids     \
            -min 1                          \
            -max 30                         \
            -num $NUM_SAMPLES               \
            $OUTPUT_DIR/test.tgt.ref $OUTPUT_DIR/test.tgt.baseline $OUTPUT_DIR/test.tgt.rr_fixed $OUTPUT_DIR/test.tgt.rr_train $OUTPUT_DIR/test.tgt.mbr $OUTPUT_DIR/test.tgt.rr_mbr \
                > $OUTPUT_DIR/output.csv







