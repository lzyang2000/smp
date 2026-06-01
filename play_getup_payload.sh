#!/usr/bin/env bash
# Play the payload SMP getup teacher (Smp-Getup-G1) in the native MuJoCo viewer.
#
# Auto-resolves the latest local run + highest checkpoint under
# logs/rsl_rl/smp_getup_g1/ when no checkpoint is given. SMP_ATTACH_PAYLOADS=1
# is forced so the played robot (Jetson + welded Dex1-1 grippers, base_lin_vel
# dropped from the actor) matches what training saw — otherwise the obs/robot
# won't match the checkpoint.
#
# Usage:
#   bash play_getup_payload.sh [GPU_ID] [CHECKPOINT.pt] [extra play flags...]
#
# Examples:
#   bash play_getup_payload.sh 0
#   bash play_getup_payload.sh 0 logs/rsl_rl/smp_getup_g1/<run>/model_20000.pt
#   bash play_getup_payload.sh 0 --video True            # headless: record instead of viewer
#   bash play_getup_payload.sh 0 --wandb-run-path org/smp/<run>   # pull from W&B

set -euo pipefail
cd "$(dirname "$0")"

GPU_ID="${1:-0}"
shift || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export SMP_ATTACH_PAYLOADS="${SMP_ATTACH_PAYLOADS:-1}"

TASK="Smp-Getup-G1"
EXP_DIR="logs/rsl_rl/smp_getup_g1"

# If the next arg is an explicit .pt, use it; else auto-resolve latest local.
CKPT_ARGS=()
if [[ "${1:-}" == *.pt ]]; then
  CKPT_ARGS=(--checkpoint-file "$1")
  echo "[play] checkpoint (explicit): $1"
  shift
elif [[ "${1:-}" != --wandb-run-path && -d "$EXP_DIR" ]]; then
  RUN_DIR="$(ls -dt "$EXP_DIR"/*/ 2>/dev/null | head -1 || true)"
  if [[ -n "$RUN_DIR" ]]; then
    NUM="$(ls -1 "${RUN_DIR}"model_*.pt 2>/dev/null \
            | sed -E 's@.*/model_([0-9-]+)\.pt@\1@' | sort -n | tail -1 || true)"
    if [[ -n "$NUM" ]]; then
      CKPT_ARGS=(--checkpoint-file "${RUN_DIR}model_${NUM}.pt")
      echo "[play] checkpoint (latest local): ${RUN_DIR}model_${NUM}.pt"
    fi
  fi
fi

if [[ ${#CKPT_ARGS[@]} -eq 0 && "${1:-}" != --wandb-run-path ]]; then
  echo "[play] WARNING: no local checkpoint found under ${EXP_DIR};"
  echo "       pass a model_*.pt path or --wandb-run-path org/smp/<run>."
fi

# Native viewer by default; small env count. Pass --video True for headless.
uv run scripts/play.py "${TASK}" \
  --num-envs 1 \
  --video False \
  "${CKPT_ARGS[@]}" \
  "$@"
