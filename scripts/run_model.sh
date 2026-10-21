#!/bin/bash
# usage: run_model.sh <gpu> <hf model id>
# set HF_HOME (model cache) and PY (python with transformers) for your machine
set -e
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=$1
PY=${PY:-python}
M=$2; TAG=$(basename $M)-q4
$PY scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions direct oracle oracle_only wrong shuffled
$PY scripts/recon.py data/v1 --backend hf --model $M --quant 4
$PY scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions recon --recon $TAG
$PY scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions cot
echo ALL_DONE
