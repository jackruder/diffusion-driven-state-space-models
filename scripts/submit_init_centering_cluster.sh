#!/usr/bin/env bash
# Submit the init-centering round-1 MOO sweep to SLURM.
#
# Cell layout (8 pinned cells from the post-global_ema-removal grid).
# Tempest capacity: gpupriority = 1 A100 + 1 A40 (both non-preemptible),
# gpuunsafe = unlimited A40 (preemptible). Allocation:
#   - 1 cell  (A100_CELL)         -> A100 on gpupriority, non-preempt (N=6).
#   - 1 cell  (A40_PRIORITY_CELL) -> A40  on gpupriority, non-preempt (N=6).
#   - 6 cells                     -> A40  on gpuunsafe, preempt w/ --requeue (N=6).
# All 8 run concurrently (gpuunsafe is uncapped; both gpupriority slots used).
# GRES is type-qualified (gpu:a100:1 / gpu:a40:1) so cells land on the
# intended GPU. Jobs run under the priority-michaelwojnowicz account.
#
# Preemption note: gpuunsafe jobs can be killed and requeued. Per the
# handoff protocol, parametric-μ_p forms need an uninterrupted stage-1;
# these round-1 cells are baseline_mode=pinned (μ_p fixed, not learned),
# so a preempt-restart re-runs from scratch but doesn't corrupt a
# half-trained μ_p. Completed Optuna trials persist in the shared DB.
#
# Concurrency model differs by partition (see step 2 below):
#   - gpupriority cells: one job packs N ``python -m ddssm.app`` worker
#     processes on the single non-preempt GPU, sharing the cell's Optuna
#     SQLite study.
#   - gpuunsafe cells: a job ARRAY of N tasks, each one worker on its own
#     A40, all sharing the cell's study. Per-task --requeue means a
#     preempt loses only that task's in-flight trial; the others keep
#     running. (This is the preempt-resilient pattern — distribute trials
#     of one study across many independent jobs.)
# In both, per-worker ``hydra.sweep.subdir`` (keyed on HYDRA_WORKER_ID)
# prevents trial-dir collisions; the shared SQLite DB (WAL) coordinates.
#
# At 96 trials × 5000 stage-2 steps spread over 6 workers/tasks, the
# slowest A40 cell is expected to take 14-16h; the SLURM time limit
# (16:00:00) is the cap. Trials that don't finish (time cap or preempt)
# are simply absent from the Optuna DB — NSGA-II ranks against whatever
# completed. Note: each array task runs n_trials independently, so a
# requeued task can push a cell slightly past TOTAL_TRIALS; harmless
# (more Pareto samples), and undershoot is fine too.
#
# Cluster paths on Tempest (Montana State University). These live on
# /home (shared FS, visible to every compute node). Override via env
# vars if you relocate to /scratch. The script refuses to submit while
# any value still starts with TODO.

CLUSTER_BASE=${CLUSTER_BASE:-/home/z89p425/ddssm}
CLUSTER_RUNS_DIR=${CLUSTER_RUNS_DIR:-${CLUSTER_BASE}/runs}
CLUSTER_SWEEPS_DIR=${CLUSTER_SWEEPS_DIR:-${CLUSTER_BASE}/sweeps}
CLUSTER_OPTUNA_DIR=${CLUSTER_OPTUNA_DIR:-${CLUSTER_BASE}/optuna}

# Environment bootstrap inside each job. Tempest has no PyTorch >= 2.9
# module, so we load Python 3.13 + CUDA 13 (matches the cu13 wheels in
# uv.lock) + the uv module, then activate a project venv built on the
# cluster:
#   module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22
#   cd $CLUSTER_BASE && uv sync   # creates .venv with torch>=2.9.1 (cu13)
CLUSTER_VENV=${CLUSTER_VENV:-${CLUSTER_BASE}/.venv}
MODULE_LOADS=${MODULE_LOADS:-"Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22"}

# SLURM account. michaelwojnowicz is the account with GPU access on
# Tempest. Override via SBATCH_ACCOUNT=<acct>; empty string omits it.
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-priority-michaelwojnowicz}

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

# Partition + GRES per GPU type. The A100 cell runs on gpupriority
# (non-preemptible); the A40 cells run on gpuunsafe (preemptible — see
# the requeue note below). GRES is type-qualified on Tempest, so a bare
# ``gpu:1`` could land the A100 cell on an A40 — pin the type explicitly.
A100_PARTITION=${A100_PARTITION:-gpupriority}
A40_PARTITION=${A40_PARTITION:-gpuunsafe}
A100_GRES=${A100_GRES:-gpu:a100:1}
A40_GRES=${A40_GRES:-gpu:a40:1}

# gpupriority offers exactly 1 A100 + 1 A40 (both non-preemptible). The
# A100 cell takes the A100; route ONE A40 cell to the spare non-preempt
# A40 slot so it's immune to preemption (the rest go to gpuunsafe). Set
# A40_PRIORITY_CELL="" to disable and send all A40 cells to gpuunsafe.
A40_PRIORITY_CELL=${A40_PRIORITY_CELL:-init_mlp_pinned_fixed}
A40_PRIORITY_PARTITION=${A40_PRIORITY_PARTITION:-gpupriority}

# Submission gating.
DRY_RUN=${DRY_RUN:-0}        # 1 = don't sbatch, just render the wrapped scripts
SBATCH_MEM=${SBATCH_MEM:-64G}
# Cell filter (glob against the cell name, e.g. 'init_zero_pinned_fixed' or
# 'init_zero*'). Default '*' = all 8 cells. Use it to submit one cell for a
# small validation run before the full launch.
CELL_GLOB=${CELL_GLOB:-'*'}

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

# Validate the pinned-A40 cell actually exists (and isn't the A100 cell).
# A non-matching name would silently send every A40 cell to gpuunsafe,
# wasting the non-preempt slot — so fail loudly instead.
if [[ -n "$A40_PRIORITY_CELL" ]]; then
  if [[ "$A40_PRIORITY_CELL" == "$A100_CELL" ]]; then
    echo "ERROR: A40_PRIORITY_CELL ($A40_PRIORITY_CELL) is the A100 cell. Pick a different A40 cell or set A40_PRIORITY_CELL=''."
    exit 1
  fi
  if [[ ! -f "$SBATCH_DIR/_base/${A40_PRIORITY_CELL}__mv.sbatch" ]]; then
    echo "ERROR: A40_PRIORITY_CELL ($A40_PRIORITY_CELL) does not match any rendered cell. Available:"
    for b in "$SBATCH_DIR/_base"/init_*__mv.sbatch; do echo "  - $(basename "$b" __mv.sbatch)"; done
    exit 1
  fi
  echo "Pinning A40 cell '$A40_PRIORITY_CELL' to the non-preempt gpupriority A40."
fi

# Account directive, injected into each wrapped sbatch header. A comment
# (not a blank line) when unset, to keep the header tidy.
SBATCH_ACCOUNT_LINE="# (no --account configured)"
if [[ -n "$SBATCH_ACCOUNT" ]]; then
  SBATCH_ACCOUNT_LINE="#SBATCH --account=${SBATCH_ACCOUNT}"
fi

# 2. For each base sbatch, build a wrapper. Two shapes:
#    - gpupriority cells (A100 + pinned A40): one job that PACKS N worker
#      processes on its single non-preempt GPU, sharing the cell's study.
#    - gpuunsafe cells: a SLURM job ARRAY of N tasks (1 worker each, own
#      A40, --requeue) sharing the cell's study. Preempting one task loses
#      only its in-flight trial; the rest keep going. (Per the user's
#      preemptible-sweep preference.)
#    Both rewrite n_trials to per-worker, add the per-worker subdir prefix
#    + stage_2 step override. The Optuna SQLite store is pre-touched in WAL
#    mode once from the submit node (avoids an array-task creation race).
JOB_IDS=()
for base in "$SBATCH_DIR/_base"/init_*__mv.sbatch; do
  [ -f "$base" ] || continue
  cell=$(basename "$base" __mv.sbatch)

  # Cell filter (unquoted RHS so it's a glob match, not literal).
  [[ $cell == $CELL_GLOB ]] || continue

  if [[ "$cell" == "$A100_CELL" ]]; then
    n_workers=$A100_WORKERS
    time_budget=$A100_TIME
    gpu_label="a100"
    partition=$A100_PARTITION
    gres=$A100_GRES
    use_array=0
    # gpupriority is non-preemptible — no requeue needed.
    requeue_line="# (gpupriority: non-preemptible, no --requeue)"
  elif [[ -n "$A40_PRIORITY_CELL" && "$cell" == "$A40_PRIORITY_CELL" ]]; then
    # The one A40 cell pinned to the spare non-preempt gpupriority A40.
    n_workers=$A40_WORKERS
    time_budget=$A40_TIME
    gpu_label="a40-prio"
    partition=$A40_PRIORITY_PARTITION
    gres=$A40_GRES
    use_array=0
    requeue_line="# (gpupriority A40: non-preemptible, no --requeue)"
  else
    n_workers=$A40_WORKERS
    time_budget=$A40_TIME
    gpu_label="a40"
    partition=$A40_PARTITION
    gres=$A40_GRES
    use_array=1
    # gpuunsafe is preemptible — array tasks each requeue independently.
    requeue_line="#SBATCH --requeue"
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
  worker_cmd="${base_cmd} 'hydra.sweep.subdir=w\\\${oc.env:HYDRA_WORKER_ID}_\\\${hydra.job.num}' experiment.training.stages.n_stage2=${STAGE2_STEPS}"

  # Pre-touch the SQLite store in WAL mode once, here on the submit node
  # (/home is shared FS visible to compute nodes). Doing it once — not in
  # each job/array-task — avoids array tasks racing on schema creation.
  # Skipped under DRY_RUN (don't create cluster DBs while just rendering).
  if (( ! DRY_RUN )); then
    .venv/bin/python - "$db_path" <<'PY'
import sqlite3, pathlib, sys
p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(str(p)); c.execute("PRAGMA journal_mode=WAL;"); c.close()
PY
  fi

  wrapped="$SBATCH_DIR/${cell}__mv.sbatch"
  if (( use_array )); then
    # gpuunsafe: job array, one worker per task, shared study, --requeue.
    array_max=$(( n_workers - 1 ))
    cat > "$wrapped" <<EOF
#!/bin/bash
#SBATCH --job-name=${cell}__mv
#SBATCH --partition=${partition}
${SBATCH_ACCOUNT_LINE}
#SBATCH --time=${time_budget}
#SBATCH --gres=${gres}
#SBATCH --requeue
#SBATCH --array=0-${array_max}
#SBATCH --cpus-per-task=4
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --nodes=1
#SBATCH --output=${CLUSTER_RUNS_DIR}/slurm-${STUDY_PREFIX}-${cell}__mv-%A_%a.out

set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"
module purge
module load ${MODULE_LOADS}
source "${CLUSTER_VENV}/bin/activate"

echo "[\$(date -Iseconds)] cell=${cell} gpu=${gpu_label} mode=array task=\${SLURM_ARRAY_TASK_ID} trials/task=${trials_per_worker} n_stage2=${STAGE2_STEPS} wallclock_target=${WALLCLOCK_TARGET}"
echo "[\$(date -Iseconds)] python=\$(command -v python) torch=\$(python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1)"

# One worker per array task. HYDRA_WORKER_ID = array task id keeps the
# per-worker hydra.sweep.subdir collision-free; all tasks share this
# cell's Optuna study (pre-touched in WAL mode) and coordinate NSGA-II
# via the DB. Foreground (no &): the task exit status is the worker's; on
# preempt SLURM kills it and --requeue restarts just this one task.
log="${CLUSTER_RUNS_DIR}/${STUDY_PREFIX}_${cell}__mv.task\${SLURM_ARRAY_TASK_ID}.log"
mkdir -p "\$(dirname "\$log")"
HYDRA_WORKER_ID=\${SLURM_ARRAY_TASK_ID} eval "${worker_cmd}"
EOF
  else
    # gpupriority: one job packing N workers on the single non-preempt GPU.
    cat > "$wrapped" <<EOF
#!/bin/bash
#SBATCH --job-name=${cell}__mv
#SBATCH --partition=${partition}
${SBATCH_ACCOUNT_LINE}
#SBATCH --time=${time_budget}
#SBATCH --gres=${gres}
${requeue_line}
#SBATCH --cpus-per-task=4
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --nodes=1
#SBATCH --output=${CLUSTER_RUNS_DIR}/slurm-${STUDY_PREFIX}-${cell}__mv-%j.out

set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"
module purge
module load ${MODULE_LOADS}
source "${CLUSTER_VENV}/bin/activate"

echo "[\$(date -Iseconds)] cell=${cell} gpu=${gpu_label} mode=packed workers=${n_workers} trials/worker=${trials_per_worker} n_stage2=${STAGE2_STEPS} wallclock_target=${WALLCLOCK_TARGET}"
echo "[\$(date -Iseconds)] python=\$(command -v python) torch=\$(python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1)"

# Pack N workers on the one non-preempt GPU; all share this cell's study.
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
  fi

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
