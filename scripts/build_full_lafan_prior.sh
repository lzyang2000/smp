#!/usr/bin/env bash
# Build the "full_lafan_prior" — a balanced, left/right-mirrored G1 SMP
# locomotion prior trained on the full LAFAN1 G1 set — and wire it into
# Smp-Dodgeball-G1.
#
# Why: the shipped `pretrained_lafan_run.pt` is forward-run-only, so the SMP
# guidance reward pays the G1 to run forward → it runs away instead of dodging
# in place. MimicKit's dodgeball prior is balanced multi-directional locomotion
# (forward/left/right run + walk). This reproduces that recipe for the G1:
# broad LAFAN locomotion + a sagittal mirror for exact left/right symmetry, so
# in-place stepping and sidestep-dodging (either direction) are on-manifold.
#
# Pipeline:  gather CSVs -> csv_to_npz (--mirror) -> [reuse norm_stats] ->
#            pretrain (EMA) -> install prior -> train dodgeball policy.
#
# One invocation builds the prior AND trains Smp-Dodgeball-G1. Run from anywhere;
# paths resolve relative to the repo. Override any UPPER_CASE var via the env:
#   LAFAN_G1_DIR=/data/lafan/g1 USE_WANDB=true ./scripts/build_full_lafan_prior.sh
#   RUN_RL=false ./scripts/build_full_lafan_prior.sh    # build the prior only
set -euo pipefail

# ---- Config (override via env) ------------------------------------------------
# Directory of raw LAFAN1 G1 retargeted CSVs (one clip per .csv, 30 fps, 36 cols).
LAFAN_G1_DIR="${LAFAN_G1_DIR:-$HOME/LAFAN1_Retargeting_Dataset/g1}"
# Which clips to include. Default = full LAFAN. For locomotion-only, set e.g.
#   CLIP_GLOB='walk*.csv run*.csv sprint*.csv'
CLIP_GLOB="${CLIP_GLOB:-*.csv}"
PRIOR_NAME="${PRIOR_NAME:-full_lafan_prior}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"   # diffusion pretraining epochs
BATCH_SIZE="${BATCH_SIZE:-1024}"   # pretrain batch (bump big on an H100 -- the windows live on GPU)
NUM_ENVS="${NUM_ENVS:-4096}"       # RL parallel envs
USE_WANDB="${USE_WANDB:-false}"    # log pretraining to W&B
RUN_RL="${RUN_RL:-true}"           # train the dodgeball policy at the end (RUN_RL=false = build prior only)

# ---- Resolve paths ------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_DIR="$REPO_DIR/datasets/csv/$PRIOR_NAME"
NPZ_DIR="$REPO_DIR/datasets/npz/$PRIOR_NAME"
NORM_STATS="$REPO_DIR/datasets/norm_stats.npz"
CKPT_OUT="$REPO_DIR/datasets/pretrain_ckpt/$PRIOR_NAME.pt"
cd "$REPO_DIR"

echo "=== full_lafan_prior build ==="
echo "  LAFAN_G1_DIR=$LAFAN_G1_DIR  CLIP_GLOB='$CLIP_GLOB'"
echo "  -> prior: $CKPT_OUT  (epochs=$NUM_EPOCHS, wandb=$USE_WANDB)"

# ---- 1. Gather clips ----------------------------------------------------------
[ -d "$LAFAN_G1_DIR" ] || { echo "ERROR: LAFAN_G1_DIR not found: $LAFAN_G1_DIR"; exit 1; }
rm -rf "$CSV_DIR"; mkdir -p "$CSV_DIR"
# Split CLIP_GLOB into literal patterns (glob disabled so '*.csv' isn't expanded
# against the CWD), then expand each pattern inside LAFAN_G1_DIR (nullglob).
set -f; read -ra CLIP_PATS <<< "$CLIP_GLOB"; set +f
shopt -s nullglob
n=0
for pat in "${CLIP_PATS[@]}"; do
  for f in "$LAFAN_G1_DIR"/$pat; do cp -f "$f" "$CSV_DIR/"; n=$((n + 1)); done
done
echo "[1/5] gathered $n CSV(s) into $CSV_DIR"
[ "$n" -gt 0 ] || { echo "ERROR: no CSVs matched '$CLIP_GLOB' in $LAFAN_G1_DIR"; exit 1; }

# ---- 2. CSV -> windowed NPZ, with left/right mirror augmentation --------------
# Clean NPZ_DIR first: the pretrain loads EVERY .npz in this dir, so stale files from a
# previous run with a different CLIP_GLOB would leak in (e.g. dance/fight NPZs surviving a
# switch to a locomotion-only glob, inflating the dataset back to the full set).
echo "[2/5] csv_to_npz (--mirror): doubling clips with sagittal mirror"
rm -rf "$NPZ_DIR"; mkdir -p "$NPZ_DIR"
uv run scripts/csv_to_npz.py --input-dir "$CSV_DIR" --output-dir "$NPZ_DIR" --mirror

# ---- 3. Normalization stats ---------------------------------------------------
# Reuse the shipped wide full-LAFAN q01/q99 stats (the normalizer is baked into
# the checkpoint; a wide one keeps the frozen denoiser's score reliable across
# the states RL actually visits). Recompute only if you changed the layout:
#   uv run scripts/compute_norm_stats.py --input-dir "$NPZ_DIR" --output "$NORM_STATS"
[ -f "$NORM_STATS" ] || { echo "ERROR: missing $NORM_STATS"; exit 1; }
echo "[3/5] reusing norm stats: $NORM_STATS"

# ---- 4. Pretrain the diffusion prior ------------------------------------------
# EMA on: the RL loader prefers the EMA weights (load_denoiser: model_ema or model).
WANDB_FLAG="--no-use-wandb"; [ "$USE_WANDB" = "true" ] && WANDB_FLAG="--use-wandb"
echo "[4/5] pretraining diffusion prior '$PRIOR_NAME'"
uv run scripts/pretrain.py \
  --data-dir "$NPZ_DIR" \
  --norm-stats-file "$NORM_STATS" \
  --name "$PRIOR_NAME" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --use-ema \
  $WANDB_FLAG

# ---- 5. Install the prior + wire into the dodgeball env -----------------------
FINAL="$(ls -t logs/pretrain/"$PRIOR_NAME"/*/pretrained.pt 2>/dev/null | head -1 || true)"
[ -n "$FINAL" ] || { echo "ERROR: no pretrained.pt under logs/pretrain/$PRIOR_NAME/"; exit 1; }
mkdir -p "$(dirname "$CKPT_OUT")"
cp -f "$FINAL" "$CKPT_OUT"
echo "[5/5] installed prior: $FINAL -> $CKPT_OUT"
echo
if [ "$PRIOR_NAME" = "full_lafan_prior" ]; then
  echo "Smp-Dodgeball-G1 already points at datasets/pretrain_ckpt/full_lafan_prior.pt"
else
  echo "NOTE: Smp-Dodgeball-G1 points at full_lafan_prior.pt; to use this prior,"
  echo "      set the ckpt_path in src/smp/rl/tasks/dodgeball/dodgeball_env_cfg.py"
  echo "      to datasets/pretrain_ckpt/$PRIOR_NAME.pt"
fi
echo "Optional sanity-check (needs a display):"
echo "    uv run scripts/generate_viz.py --ckpt-path $CKPT_OUT"

# ---- 6. Train the dodgeball policy --------------------------------------------
RL_CMD="uv run scripts/train.py Smp-Dodgeball-G1 --env.scene.num-envs=$NUM_ENVS"
echo
if [ "$RUN_RL" = "true" ]; then
  echo "Launching RL: $RL_CMD"
  eval "$RL_CMD"
else
  echo "To train the dodgeball policy on the new prior, run:"
  echo "    $RL_CMD"
fi
