#!/usr/bin/env bash
# Overnight ablation: 18 cells × N_TRIALS trials on the MV dataset
# (latent_dim=4). Fixed versus the original Phase-F script:
#
#   1. Real concurrency. Original used ``n_jobs>1`` inside Optuna,
#      which is in-process joblib threading and was GIL-serialized
#      (we observed ~32% GPU util with n_jobs=6). Here we launch
#      N_WORKERS independent ddssm.app processes per cell that all
#      share one Optuna SQLite store — TPE coordinates via the DB.
#
#   2. No per-trial subdir collisions. With the original layout each
#      worker's Hydra ``--multirun`` had its own local job counter
#      (0,1,2,...) and they all wrote into the same ``hydra.sweep.dir``
#      so workers overwrote each other. We override
#      ``hydra.sweep.subdir='w${oc.env:HYDRA_WORKER_ID}_${hydra.job.num}'``
#      so worker W's trials land at ``.../wW_0/``, ``.../wW_1/``...
#      All trial outputs survive.
#
#   3. Stage-2 step budget actually honored. The original passed
#      ``experiment.training.steps`` which is shadowed by
#      ``model.stages.n_stage2`` in the init-centering cells — so the
#      "1200-step" runs only actually trained ~1300 steps because
#      n_stage2's default is 1000. Most trials underfit. Here we bump
#      ``experiment.model.stages.n_stage2`` directly. Default 4000
#      (total ≈ 4000 + n_pretrain_sampled ≈ 4500 effective steps).
#
# Resource notes for a single 24 GB GPU:
#   * Single trial peaks ~3.5 GB → 4 workers × ~14 GB = comfortable.
#     6 workers ≈ 21 GB — close to OOM, not recommended.
#   * Cells run sequentially. Concurrent cells × concurrent workers
#     would split GPU and starve each other; that's why we serialize
#     across cells but parallelize within.
#
# Run::
#   nohup bash scripts/run_overnight_mv_ablation.sh > runs/overnight.log 2>&1 &
#
# Tunable env vars::
#   STAMP=<YYYYMMDD>       timestamp (default: today)
#   N_TRIALS_PER_WORKER=12 Optuna trials each worker runs
#                          (4 workers × 12 trials = 48 trials per cell)
#   N_WORKERS=4            independent processes per cell sharing the DB
#   STAGE2_STEPS=3500      n_stage2 override (governs total wallclock)
#   SWEEP_GROUP=init_ablation_moo   single- vs multi-objective sweep:
#                          'init_ablation' = single-obj TPE on stage2_elbo_surrogate;
#                          'init_ablation_moo' = NSGA-II MOO on
#                          (wallclock_to_target_seconds, stage2_elbo_surrogate).
#                          MOO is the round-1 default.
#   WALLCLOCK_TARGET=-30   ELBO threshold for the wallclock_to_target
#                          objective. Override to retune the front
#                          without editing presets.
#   BASELINE_MODES=pinned  axis filter for baseline_mode (default: pinned only
#                          for the round-1 gating; set to 'pinned learnable'
#                          or unset to enable both)
#   BASELINE_FORMS=...     axis filter for baseline_form (default: all)
#   TRACKING_MODES=...     axis filter for tracking_mode (default: all)
#   CELLS_PATTERN='*'      additional glob on rendered sbatches
#                          (e.g. CELLS_PATTERN='init_zero*' for a subset)
#
# After it finishes, aggregate + plot::
#   .venv/bin/python -m experiments.init_centering.report all \
#       --sweeps-root runs/sweeps --optuna-dir runs/optuna \
#       --study-prefix overnight_${STAMP} --dataset mv \
#       --out runs/report/overnight_${STAMP}

set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%d)}
SBATCH_DIR=runs/sbatch/overnight_${STAMP}
N_TRIALS_PER_WORKER=${N_TRIALS_PER_WORKER:-12}
N_WORKERS=${N_WORKERS:-4}
STAGE2_STEPS=${STAGE2_STEPS:-3500}
CELLS_PATTERN=${CELLS_PATTERN:-'*'}
BASELINE_MODES=${BASELINE_MODES:-pinned}
BASELINE_FORMS=${BASELINE_FORMS:-}
TRACKING_MODES=${TRACKING_MODES:-}
SWEEP_GROUP=${SWEEP_GROUP:-init_ablation_moo}
WALLCLOCK_TARGET=${WALLCLOCK_TARGET:--30}

STUDY_PREFIX=overnight_${STAMP}

mkdir -p runs/optuna runs/sweeps

# 1. Render the MV-only sbatch scripts. We pass n_jobs=1 — workers
#    are separate processes, no in-process threading.
launch_args=(
  --write-dir "$SBATCH_DIR"
  --study-prefix "$STUDY_PREFIX"
  --n-trials "$N_TRIALS_PER_WORKER"
  --n-jobs 1
  --datasets mv
  --sweep-group "$SWEEP_GROUP"
  --wallclock-target "$WALLCLOCK_TARGET"
)
[ -n "$BASELINE_MODES" ] && launch_args+=(--baseline-modes $BASELINE_MODES)
[ -n "$BASELINE_FORMS" ] && launch_args+=(--baseline-forms $BASELINE_FORMS)
[ -n "$TRACKING_MODES" ] && launch_args+=(--tracking-modes $TRACKING_MODES)
.venv/bin/python -m experiments.init_centering.launch_ablation_tiny "${launch_args[@]}"

# 2. Per cell: fan out N_WORKERS independent processes sharing one
#    Optuna SQLite DB. Each worker writes to per-worker-prefixed trial
#    subdirs (collision-free) and bumps n_stage2 for real convergence.
echo
total_cells=$(ls "$SBATCH_DIR"/${CELLS_PATTERN}__mv.sbatch 2>/dev/null | wc -l)
echo "=== Starting MV ablation ==="
echo "    cells:       $total_cells (mv only)"
echo "    sweep:       $SWEEP_GROUP  wallclock_target=$WALLCLOCK_TARGET"
echo "    filters:     baseline_modes='$BASELINE_MODES' baseline_forms='$BASELINE_FORMS' tracking_modes='$TRACKING_MODES'"
echo "    workers:     $N_WORKERS per cell (multi-process, GIL-free)"
echo "    trials:      $N_TRIALS_PER_WORKER per worker × $N_WORKERS = $((N_WORKERS*N_TRIALS_PER_WORKER)) per cell"
echo "    n_stage2:    $STAGE2_STEPS"
echo "    layout:      \$sweep_dir/w{W}_{N}/  (per-worker subdir prefix)"
echo

cell_idx=0
for sbatch_file in "$SBATCH_DIR"/${CELLS_PATTERN}__mv.sbatch; do
  [ -f "$sbatch_file" ] || continue
  cell_idx=$((cell_idx + 1))
  cell=$(basename "$sbatch_file" __mv.sbatch)
  base_cmd=$(grep '^exec python' "$sbatch_file" | sed 's/^exec //')
  db_path="runs/optuna/${STUDY_PREFIX}_${cell}__mv.db"

  # WAL on the SQLite store reduces 'database is locked' retries when
  # multiple workers commit concurrently. Pre-touch the file so
  # Optuna's SQLAlchemy opens an already-WAL store on first contact.
  .venv/bin/python - <<PY
import sqlite3, pathlib
p = pathlib.Path("$db_path"); p.parent.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(str(p)); c.execute("PRAGMA journal_mode=WAL;"); c.close()
PY

  echo "--- [$cell_idx/$total_cells] $cell  n_stage2=$STAGE2_STEPS  workers=${N_WORKERS}×${N_TRIALS_PER_WORKER}=$((N_WORKERS*N_TRIALS_PER_WORKER)) trials ---"
  echo "    started: $(date -Iseconds)"

  pids=()
  for w in $(seq 0 $((N_WORKERS - 1))); do
    log="runs/sweeps/${STUDY_PREFIX}_${cell}__mv.worker${w}.log"
    mkdir -p "$(dirname "$log")"
    # The two added overrides:
    #   * hydra.sweep.subdir: per-worker-prefixed unique subdir name.
    #     Note: single-quoted so the shell doesn't expand ${...} —
    #     OmegaConf resolves at config eval time inside Python.
    #   * experiment.model.stages.n_stage2: actual step budget knob.
    cmd="$base_cmd 'hydra.sweep.subdir=w\${oc.env:HYDRA_WORKER_ID}_\${hydra.job.num}' experiment.model.stages.n_stage2=$STAGE2_STEPS"
    echo "    + worker $w → $log"
    HYDRA_WORKER_ID=$w eval "$cmd" > "$log" 2>&1 &
    pids+=($!)
    # Tiny stagger so workers don't all race the DB schema creation.
    sleep 1
  done

  status=0
  for p in "${pids[@]}"; do
    wait "$p" || status=$?
  done
  echo "    finished: $(date -Iseconds), max worker exit=$status"
  if [ "$status" -ne 0 ]; then
    echo "    WARNING: a worker exited non-zero. Check the worker logs and the Optuna DB before trusting this cell."
  fi
done

echo
echo "=== All $cell_idx cells done. Aggregate + plot with:"
echo ".venv/bin/python -m experiments.init_centering.report all \\"
echo "    --sweeps-root runs/sweeps --optuna-dir runs/optuna \\"
echo "    --study-prefix ${STUDY_PREFIX} --dataset mv \\"
echo "    --out runs/report/${STUDY_PREFIX}"
