#!/usr/bin/env bash
# Full training on the six VGGT-processed adapters (configs/train_vggt_sixmix.yaml).
# Checkpoints/logs go to /jfs (local /home is nearly full); schedule comes from the config.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export D4RT_CV2_WORKER_THREADS="${D4RT_CV2_WORKER_THREADS:-0}"
export D4RT_TORCH_WORKER_THREADS="${D4RT_TORCH_WORKER_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#_gpus[@]}}"
OUT_DIR="${OUT_DIR:-/jfs/jing.feng/outputs/d4rt_vggt_sixmix_clip48}"
INIT_CKPT="${INIT_CKPT:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt}"

mkdir -p "$OUT_DIR"

.venv/bin/torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29721}" \
  train.py \
  --tb_log \
  --model-config configs/model_effective.yaml \
  --train-config configs/train_vggt_sixmix.yaml \
  --init-model "$INIT_CKPT" \
  --override "experiment.output_dir=${OUT_DIR}" \
  --override "checkpoint.keep_last_k=${KEEP_LAST_K:-5}" \
  --override "runtime.val_num_workers=${VAL_NUM_WORKERS:-4}" \
  "$@"
