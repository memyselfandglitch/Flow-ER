#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export DATA_NAME=flower_new_dataset EXP_NAME=best_large_hyperparam
export EMB_DIM=256 RBF_HIGH=12 RBF_GAP=0.1 SIGMA=0.15
export MODEL_NAME=model.2940000_97.pt
export SCALE=1 SAMPLE_SIZE="${SAMPLE_SIZE:-64}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
export MAX_SAMPLE_MATRIX_ELEMENTS="${MAX_SAMPLE_MATRIX_ELEMENTS:-4000000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MODEL_PATH="$PWD/checkpoints/$DATA_NAME/$EXP_NAME/"
export RESULT_PATH="${RESULT_PATH:-$PWD/results/$DATA_NAME/$EXP_NAME/}"
export TEST_FILE="${TEST_FILE:-$PWD/data/$DATA_NAME/test.txt}"
export TRAIN_FILE="$PWD/data/$DATA_NAME/train.txt"
export VAL_FILE="$PWD/data/$DATA_NAME/val.txt"
test -s "$MODEL_PATH/$MODEL_NAME"
test -s "$TEST_FILE"
mkdir -p "$RESULT_PATH"
exec torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS_PER_NODE" eval_multiGPU.py
