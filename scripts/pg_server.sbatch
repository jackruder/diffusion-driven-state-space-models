#!/bin/bash
# Ephemeral PostgreSQL server as a SLURM job — shared Optuna storage for a sweep.
#
# Why: per-cell SQLite files can't be viewed together and don't scale to ~100+
# concurrent workers on one NFS file. This runs ONE Postgres on a non-preemptible
# CPU node so every sweep cell lands in one DB (distinct study per cell), live.
#
# Usage:
#   sbatch scripts/pg_server.sbatch
#   squeue -u $USER -n ddssm-pg -o '%i %T %N'     # wait for RUNNING, note the node
#   grep PG_URL runs/pg/pg-<jobid>.out            # the postgresql:// URL to use
#   # then launch the sweep with:  --storage-url postgresql://ddssm@<node>:5432/round1
#   # stop with:  scancel <jobid>   (a final pg_dump is written to $HOME first)
#
# The `priority` partition is CPU-only, PreemptMode=OFF, 14-day max — so this job
# is not preempted and outlives the sweep. Data lives on node-local disk (NOT NFS,
# which Postgres mishandles); a periodic pg_dump to $HOME covers a node crash.
# Auth is `trust` on the cluster fabric (ephemeral internal DB; no secret in URLs).
#SBATCH --job-name=ddssm-pg
#SBATCH --partition=priority
#SBATCH --account=priority-michaelwojnowicz
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/pg/pg-%j.out
#SBATCH --open-mode=append
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/pg

# epyc sbatch shells do not pre-define `module`; source /etc/profile for the
# Lmod function. Relax `set -u` around it (Lmod references unset vars).
set +u
source /etc/profile
module purge
module load PostgreSQL/17.5-GCCcore-14.3.0
set -u

# Node-local data dir (never NFS). Fall back to /tmp if SLURM_TMPDIR is unset.
export PGDATA="${SLURM_TMPDIR:-/tmp/$USER}/ddssm_pgdata"
export PGPORT=5432
PGLOG="${SLURM_TMPDIR:-/tmp/$USER}/ddssm_pg.log"
DUMP="$HOME/ddssm_pg_round1.sql"
rm -rf "$PGDATA"

# Superuser role = ddssm; trust auth (the DB is ephemeral + on the isolated fabric).
initdb -D "$PGDATA" -U ddssm --auth=trust >/dev/null

# Listen on all interfaces; size connections for ~176 workers.
cat >> "$PGDATA/postgresql.conf" <<CONF
listen_addresses = '*'
port = $PGPORT
max_connections = 300
shared_buffers = 2GB
CONF

# Allow any cluster node to connect (trust). Only the internal fabric can route here.
cat >> "$PGDATA/pg_hba.conf" <<HBA
host    all    all    0.0.0.0/0    trust
host    all    all    ::/0         trust
HBA

pg_ctl -D "$PGDATA" -l "$PGLOG" -w start
createdb -U ddssm -h 127.0.0.1 -p "$PGPORT" round1

NODE=$(hostname)
echo "PG_URL=postgresql://ddssm@${NODE}:${PGPORT}/round1"
echo "[$(date -Iseconds)] PostgreSQL ready on ${NODE}:${PGPORT} (db=round1 user=ddssm auth=trust)"

# Stop cleanly + final backup on preempt/cancel/walltime/exit.
trap 'pg_dump -U ddssm round1 > "$DUMP" 2>/dev/null; pg_ctl -D "$PGDATA" -m fast stop; exit 0' TERM INT EXIT

# Periodic backup to NFS home (covers a node hardware crash).
while true; do
    sleep 1800
    pg_dump -U ddssm round1 > "$DUMP" 2>/dev/null || true
done
