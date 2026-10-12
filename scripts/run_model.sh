#!/bin/bash
# usage: run_model.sh <gpu> <hf model id>
set -e
export HF_HOME=/shared/scratch/0/home/v_laxya_kumar/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1
PY=/shared/scratch/0/home/v_laxya_kumar/miniconda3/envs/vlmjudge/bin/python
M=$2; TAG=$(basename $M)-q4
$PY scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions direct oracle oracle_only wrong shuffled
$PY scripts/recon.py data/v1 --backend hf --model $M --quant 4
$PY scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions recon --recon $TAG
$PY scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions cot
echo ALL_DONE
