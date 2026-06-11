#!/usr/bin/env bash
# Smoke run: 300 steps on the six VGGT-processed adapters (configs/train_vggt_sixmix.yaml).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
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
TOTAL_STEPS="${TOTAL_STEPS:-300}"
OUT_DIR="${OUT_DIR:-output/smoke_vggt_sixmix}"
INIT_CKPT="${INIT_CKPT:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt}"

.venv/bin/torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29719}" \
  train.py \
  --tb_log \
  --model-config configs/model_effective.yaml \
  --train-config configs/train_vggt_sixmix.yaml \
  --init-model "$INIT_CKPT" \
  --override "experiment.name=smoke_vggt_sixmix" \
  --override "experiment.output_dir=${OUT_DIR}" \
  --override "schedule.total_steps=${TOTAL_STEPS}" \
  --override "optimizer.learning_rate.warmup_steps=30" \
  --override "logging.log_every_steps=10" \
  --override "checkpoint.save_every_steps=${TOTAL_STEPS}" \
  --override "checkpoint.step_save_every_steps=${TOTAL_STEPS}" \
  "$@"
