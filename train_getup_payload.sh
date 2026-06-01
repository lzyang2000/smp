#!/usr/bin/env bash
# Train the SMP getup teacher WITH the sim-to-real payload attached
# (NVIDIA Jetson on the back + Dex1-1 grippers on both wrists) and a
# deployment-safe actor (base_lin_vel moved to the critic only).
#
# This is the candidate "recovery teacher" to swap in for the wbc_mjlab AMP
# teacher's recovery role. The frozen getup diffusion prior
# (datasets/pretrain_ckpt/pretrained_getup_f2s2.pt) is already wired into the
# task and supplies the SMP guidance reward — no pretraining needed.
#
# Usage:
#   bash train_getup_payload.sh [GPU_ID] [extra --env.* / --agent.* overrides...]
#
# Examples:
#   bash train_getup_payload.sh 0
#   bash train_getup_payload.sh 1 --env.scene.num-envs=8192 --agent.max-iterations=40000
#
# To reproduce the ORIGINAL payload-free getup task, prepend SMP_ATTACH_PAYLOADS=0.

set -euo pipefail
cd "$(dirname "$0")"

GPU_ID="${1:-0}"
shift || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
# Attach Jetson + Dex1-1 grippers (default on; export =0 to disable).
export SMP_ATTACH_PAYLOADS="${SMP_ATTACH_PAYLOADS:-1}"

TASK="Smp-Getup-G1"

# Defaults live BEFORE "$@" so any flag you pass on the command line overrides
# them. Video off by default (headless training server); pass --video True to
# enable if a display / EGL is available.
uv run scripts/train.py "${TASK}" \
  --env.scene.num-envs=4096 \
  --agent.run-name=smp_getup_g1_payload \
  --agent.wandb-project=smp \
  --video False \
  "$@"
