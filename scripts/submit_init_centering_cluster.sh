#!/usr/bin/env bash
# Submit the init-centering round-1 MOO sweep to SLURM.
#
# Cell layout (8 pinned cells from the post-global_ema-removal grid):
#   - 1 cell -> A100 80GB on gpupriority (run with N=6 workers).
#   - 7 cells -> A40 48GB on gpupriority (run each with N=6 workers).
#
# Each cell's sbatch is single-job-single-GPU but with multi-process
# concurrency inside the bash body — N parallel ``python -m ddssm.app``
# processes share one Optuna SQLite DB. Per-worker ``hydra.sweep.subdir``
# prefix prevents trial-dir collisions.
#
# At 96 trials × 5000 stage-2 steps × 6 workers, the slowest A40 cell
# is expected to take 14-16h; the SLURM time limit (16:00:00) is the
# cap. If a worker hits the cap mid-trial, the in-progress trial is
# lost but completed trials are intact in the Optuna DB — NSGA-II
# ranks against whatever completed. Padding the budget would mean
# longer queue waits for less marginal information.
#
# Cluster path PLACEHOLDERS — fill these in before submitting. The
# paths must be on shared FS visible to every compute node. The
# script refuses to submit while any value still starts with TODO.

CLUSTER_RUNS_DIR="TODO_FILL_IN/ddssm/runs"
CLUSTER_SWEEPS_DIR="TODO_FILL_IN/ddssm/sweeps"
CLUSTER_OPTUNA_DIR="TODO_FILL_IN/ddssm/optuna"

# A100 cell. The canonical (mlp, pinned, per_t) gets the fast GPU since
# it has the most parameters in round 1 and is the headline cell.
A100_CELL="init_mlp_pinned_per_t"

# Concurrency knobs by GPU type. A40 is ~0.75× of an RTX 4090 for this
# workload; to fit 96 trials × 5000 steps in 16h we need 6 A40 workers
# (4 would push the cell wallclock to ~20-25h). A100 stays at 6 — it's
# already so fast that more workers buy little.
A100_WORKERS=6
A40_WORKERS=6

# Total Optuna trials per cell. Distributed across workers via
# ``hydra.sweeper.n_trials = TOTAL_TRIALS / N_WORKERS`` — choose values
# that divide cleanly (96 = 6×16 = 8×12; 48 = 6×8 = 4×12).
TOTAL_TRIALS=${TOTAL_TRIALS:-96}

# Training and objective knobs (round-1 design):
STAGE2_STEPS=${STAGE2_STEPS:-5000}
WALLCLOCK_TARGET=${WALLCLOCK_TARGET:--30}
SWEEP_GROUP=${SWEEP_GROUP:-init_ablation_moo}

# SLURM time budget per GPU type. Capped at 16h — slow A40 cells may
# not complete all 96 trials within budget; remaining trials are simply
# left pending in the Optuna DB and the Pareto front is built from
# whatever did complete.
A100_TIME=${A100_TIME:-16:00:00}
A40_TIME=${A40_TIME:-16:00:00}

# Submission gating.
DRY_RUN=${DRY_RUN:-0}        # 1 = don't sbatch, just render the wrapped scripts
SBATCH_PARTITION=${SBATCH_PARTITION:-gpupriority}
SBATCH_MEM=${SBATCH_MEM:-64G}

# ---------------------------------------------------------------------------

set -euo pipefail

if [[ "$CLUSTER_RUNS_DIR" == TODO* || "$CLUSTER_SWEEPS_DIR" == TODO* || "$CLUSTER_OPTUNA_DIR" == TODO* ]]; then
  echo "ERROR: fill in the CLUSTER_*_DIR placeholders at the top of $0 before submitting."
  echo "  CLUSTER_RUNS_DIR=$CLUSTER_RUNS_DIR"
  echo "  CLUSTER_SWEEPS_DIR=$CLUSTER_SWEEPS_DIR"
  echo "  CLUSTER_OPTUNA_DIR=$CLUSTER_OPTUNA_DIR"
  exit 1
fi

if (( TOTAL_TRIALS % A100_WORKERS != 0 )) || (( TOTAL_TRIALS % A40_WORKERS != 0 )); then
  echo "WARNING: TOTAL_TRIALS=$TOTAL_TRIALS doesn't divide cleanly by both worker counts (${A100_WORKERS}, ${A40_WORKERS}). Some workers will run fewer trials."
fi

STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
STUDY_PREFIX=round1_${STAMP}
SBATCH_DIR=${SBATCH_DIR:-runs/sbatch/${STUDY_PREFIX}_cluster}

# Local staging placeholders. The launcher mkdir's its --storage-dir
# and --sweeps-root, which would fail on cluster paths we can't reach
# from the submitting node. Render base sbatches with these local
# placeholders, then sed-substitute the cluster paths into the wrapped
# sbatches before submitting.
STAGE_OPTUNA="__CLUSTER_OPTUNA__"
STAGE_SWEEPS="__CLUSTER_SWEEPS__"
STAGE_BASE_DIR=$(mktemp -d)/optuna_stage
STAGE_SWEEP_DIR=$(mktemp -d)/sweeps_stage
mkdir -p "$SBATCH_DIR" "$STAGE_BASE_DIR" "$STAGE_SWEEP_DIR"

# 1. Render the base sbatches via the launcher. We pass safe local
#    paths so the launcher's internal mkdir succeeds; the cluster
#    paths are substituted into the wrapped sbatches further below.
#    n_trials at this stage is the TOTAL — we rewrite to a per-worker
#    value when we wrap.
.venv/bin/python -m experiments.init_centering.launch_ablation_tiny \
    --write-dir "$SBATCH_DIR/_base" \
    --study-prefix "$STUDY_PREFIX" \
    --n-trials "$TOTAL_TRIALS" \
    --n-jobs 1 \
    --datasets mv \
    --baseline-modes pinned \
    --sweep-group "$SWEEP_GROUP" \
    --wallclock-target "$WALLCLOCK_TARGET" \
    --storage-dir "$STAGE_BASE_DIR" \
    --sweeps-root "$STAGE_SWEEP_DIR" >/dev/null

base_count=$(ls "$SBATCH_DIR/_base"/init_*__mv.sbatch 2>/dev/null | wc -l)
echo "Rendered $base_count base sbatch scripts under $SBATCH_DIR/_base/"

# 2. For each base sbatch, build a multi-worker wrapper that:
#    - picks worker count + time budget by cell (A100 vs A40)
#    - rewrites n_trials to per-worker, adds the per-worker subdir prefix
#    - adds the stage_2 step override
#    - pre-touches the Optuna SQLite store in WAL mode
#    - spawns N parallel python processes and waits
JOB_IDS=()
for base in "$SBATCH_DIR/_base"/init_*__mv.sbatch; do
  [ -f "$base" ] || continue
  cell=$(basename "$base" __mv.sbatch)

  if [[ "$cell" == "$A100_CELL" ]]; then
    n_workers=$A100_WORKERS
    time_budget=$A100_TIME
    gpu_label="a100"
  else
    n_workers=$A40_WORKERS
    time_budget=$A40_TIME
    gpu_label="a40"
  fi

  trials_per_worker=$(( TOTAL_TRIALS / n_workers ))
  # Pull the python command out of the launcher's base sbatch, then:
  #   - rewrite n_trials to per-worker,
  #   - substitute cluster paths in for the launcher's local staging
  #     placeholders (so trials write to shared FS),
  #   - strip the trailing ``"$@"`` (we don't pass extra args to sbatch).
  base_cmd=$(grep '^exec python' "$base" | sed -E 's/^exec //' \
             | sed -E "s|hydra.sweeper.n_trials=[0-9]+|hydra.sweeper.n_trials=${trials_per_worker}|" \
             | sed -E "s|${STAGE_BASE_DIR}|${CLUSTER_OPTUNA_DIR}|g" \
             | sed -E "s|${STAGE_SWEEP_DIR}|${CLUSTER_SWEEPS_DIR}|g" \
             | sed -E 's/ "\$@"$//')
  db_path="${CLUSTER_OPTUNA_DIR}/${STUDY_PREFIX}_${cell}__mv.db"
  # The ``\\\${...}`` escapes are intentional and load-bearing.
  # In this assignment (bash double quotes): ``\\`` → ``\``, ``\$`` → ``$``,
  # so ``\\\$`` lands as ``\$`` in worker_cmd. After heredoc expansion the
  # rendered sbatch contains ``\${oc.env:...}`` inside an ``eval "..."``
  # double-quoted arg; SLURM bash sees ``\$`` and treats it as a literal
  # ``$``, skipping parameter expansion. ``eval`` then re-parses, single
  # quotes preserve the OmegaConf placeholder, and Hydra/OmegaConf
  # resolves it at config time. Without the escape, bash would try to
  # expand ``${oc.env:HYDRA_WORKER_ID}`` and fail with "bad substitution"
  # (the dot in ``oc.env`` isn't a valid bash identifier).
  worker_cmd="${base_cmd} 'hydra.sweep.subdir=w\\\${oc.env:HYDRA_WORKER_ID}_\\\${hydra.job.num}' experiment.model.stages.n_stage2=${STAGE2_STEPS}"

  wrapped="$SBATCH_DIR/${cell}__mv.sbatch"
  cat > "$wrapped" <<EOF
#!/bin/bash
#SBATCH --job-name=${cell}__mv
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --time=${time_budget}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --nodes=1
#SBATCH --output=${CLUSTER_RUNS_DIR}/slurm-${STUDY_PREFIX}-${cell}__mv-%j.out

set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"

echo "[\$(date -Iseconds)] cell=${cell}  gpu=${gpu_label}  workers=${n_workers}  trials/worker=${trials_per_worker}  n_stage2=${STAGE2_STEPS}  wallclock_target=${WALLCLOCK_TARGET}"

# Pre-touch the SQLite store in WAL mode so concurrent workers don't
# block each other on schema-creation contention.
python - <<'PY'
import sqlite3, pathlib
p = pathlib.Path("${db_path}"); p.parent.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(str(p))
c.execute("PRAGMA journal_mode=WAL;")
c.close()
PY

# Spawn N concurrent workers. Each writes its trials into a unique
# subdir within the shared hydra.sweep.dir; the shared Optuna DB
# coordinates TPE/NSGA-II across them.
pids=()
for w in \$(seq 0 \$((${n_workers} - 1))); do
  log="${CLUSTER_RUNS_DIR}/${STUDY_PREFIX}_${cell}__mv.worker\${w}.log"
  mkdir -p "\$(dirname "\$log")"
  HYDRA_WORKER_ID=\$w eval "${worker_cmd}" > "\$log" 2>&1 &
  pids+=(\$!)
  sleep 1
done
status=0
for p in "\${pids[@]}"; do wait "\$p" || status=\$?; done
echo "[\$(date -Iseconds)] cell=${cell} finished, max worker exit=\$status"
exit \$status
EOF

  if (( DRY_RUN )); then
    echo "  DRY: $wrapped  (would submit)"
  else
    out=$(sbatch "$wrapped" 2>&1)
    job_id=$(echo "$out" | awk '/Submitted batch job/ {print $4}')
    echo "  submitted: $wrapped  -> job ${job_id:-UNKNOWN}"
    [ -n "$job_id" ] && JOB_IDS+=("$job_id")
  fi
done

echo
if (( DRY_RUN )); then
  echo "DRY_RUN=1: no jobs submitted. Re-run with DRY_RUN=0 (or unset) to submit."
else
  echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
  echo
  echo "Aggregate + plot once all jobs finish:"
  echo "  .venv/bin/python -m experiments.init_centering.report all \\"
  echo "      --sweeps-root ${CLUSTER_SWEEPS_DIR} --optuna-dir ${CLUSTER_OPTUNA_DIR} \\"
  echo "      --study-prefix ${STUDY_PREFIX} --dataset mv \\"
  echo "      --out runs/report/${STUDY_PREFIX}"
fi
