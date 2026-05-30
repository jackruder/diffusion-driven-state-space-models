#!/usr/bin/env bash
# Self-configuring monitor for a DDSSM Optuna sweep on a remote SLURM cluster.
#
# Flow: reachability -> (discover + suggest if no study named) -> "have I seen
# this experiment before?" profile check -> build context if new (introspect
# objectives/derived columns from the study + resolved_config.yaml) -> probe
# per-cell stats per that context -> render a context-driven table + delta ->
# pull DBs, merge, (re)launch optuna-dashboard.
#
# Nothing about the table is hardcoded: columns/labels/derived metrics come from
# the per-experiment context, so a CRPS sweep and an ELBO MOO sweep render
# differently and correctly.
#
# Env (all optional except as noted):
#   HOST          ssh target            (default z89p425@tempest-login.msu.montana.edu)
#   REMOTE_DIR    project dir on cluster (default ~/diffusion-driven-state-space-models)
#   STUDY_PREFIX  which experiment to show. If unset, discover + auto-pick the
#                 most-active and print the ranked alternatives.
#   SUFFIX        DB dataset suffix. Resolved from discovery/profile if unset.
#   TARGET        trials/cell target for the ETA projection (default 128)
#   REBUILD=1     force a context rebuild (re-introspect, overwrite the profile)
#   PORT/PULL_DIR/NO_DASH   dashboard port / local pull dir / skip dashboard
#
# Exit codes: 2 bad args/empty, 3 cluster unreachable (VPN down).

set -uo pipefail
HOST=${HOST:-z89p425@tempest-login.msu.montana.edu}
REMOTE_DIR=${REMOTE_DIR:-'~/diffusion-driven-state-space-models'}
TARGET=${TARGET:-128}
PORT=${PORT:-8080}
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-$(git -C "$SKILL_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}
PULL_DIR=${PULL_DIR:-"$REPO_ROOT/runs/optuna_pull"}
PROFILE_DIR="$SKILL_DIR/profiles"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
mkdir -p "$PROFILE_DIR" "$PULL_DIR"

jget() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2"; }

# 0. reachability ------------------------------------------------------------
if ! $SSH "$HOST" true 2>/dev/null; then
  echo "UNREACHABLE: $HOST (private campus address; is the MSU VPN up?)." >&2
  echo "Bring up the VPN, confirm with 'ping ${HOST#*@}', then re-run." >&2
  exit 3
fi
$SSH "$HOST" "mkdir -p /tmp/mcsweep" 2>/dev/null
CLEAN() { grep -vE "post-quantum|store now|openssh.com|may need to be"; }
runpy() {  # scp a skill script and run it in the cluster venv; args follow
  local script="$1"; shift
  scp -q "$SKILL_DIR/$script" "$HOST:/tmp/mcsweep/$script"
  $SSH "$HOST" "bash -lc 'cd $REMOTE_DIR && source .venv/bin/activate && \
    python /tmp/mcsweep/$script $*'" 2>&1 | CLEAN
}

# 1. discover + suggest if no study named ------------------------------------
if [ -z "${STUDY_PREFIX:-}" ]; then
  echo "No STUDY_PREFIX given — discovering running experiments…"
  DISC=$(runpy discover.py --remote-dir "$REMOTE_DIR")
  echo "$DISC" | grep -v '^__JSON__'
  DJSON=$(echo "$DISC" | grep '^__JSON__' | sed 's/^__JSON__//')
  echo "$DJSON" > "$PULL_DIR/.discover.json"
  read -r STUDY_PREFIX SUFFIX_AUTO < <(python3 - "$PULL_DIR/.discover.json" <<'PY'
import json,sys
exps=json.load(open(sys.argv[1])).get("experiments",[])
exps=[e for e in exps if "(idle)" not in e["study_prefix"]] or exps
print(exps[0]["study_prefix"], exps[0]["suffix"]) if exps else print("", "")
PY
)
  [ -n "$STUDY_PREFIX" ] && echo ">> suggesting most-active: $STUDY_PREFIX (override with STUDY_PREFIX=…)"
fi
if [ -z "${STUDY_PREFIX:-}" ]; then
  echo "ERROR: no experiment to show (set STUDY_PREFIX=…)." >&2; exit 2
fi
SUFFIX=${SUFFIX:-${SUFFIX_AUTO:-__mv}}

# 2. profile: "have I worked on this experiment before?" ---------------------
PROFILE="$PROFILE_DIR/${STUDY_PREFIX}.json"
if [ -f "$PROFILE" ] && [ "${REBUILD:-0}" != "1" ]; then
  echo "✓ Seen '$STUDY_PREFIX' before — using cached context ($PROFILE)."
else
  echo "• First time on '$STUDY_PREFIX'${REBUILD:+ (rebuild forced)} — building display context…"
  CTX=$(runpy build_context.py --remote-dir "$REMOTE_DIR" \
        --study-prefix "$STUDY_PREFIX" --suffix "$SUFFIX")
  echo "$CTX" | grep -v '^__JSON__'
  echo "$CTX" | grep '^__JSON__' | sed 's/^__JSON__//' > "$PROFILE"
  if [ ! -s "$PROFILE" ]; then echo "ERROR: context build failed." >&2; exit 2; fi
fi
SUFFIX=$(jget "$PROFILE" suffix); SUFFIX=${SUFFIX:-__mv}

# 3. probe per-cell stats per the context ------------------------------------
scp -q "$PROFILE" "$HOST:/tmp/mcsweep/ctx.json"
PROBE=$(runpy probe.py --remote-dir "$REMOTE_DIR" --context /tmp/mcsweep/ctx.json \
        --target "$TARGET")
echo "$PROBE" | grep -v '^__JSON__'
JSON=$(echo "$PROBE" | grep '^__JSON__' | sed 's/^__JSON__//')

# 4. context-driven summary table + delta vs last snapshot -------------------
SNAP="$PULL_DIR/.snapshot_${STUDY_PREFIX}.json"
if [ -n "$JSON" ]; then
  CUR_TMP="$PULL_DIR/.cur_${STUDY_PREFIX}.json"; echo "$JSON" > "$CUR_TMP"
  PREV_ARG="$SNAP"; [ -f "$SNAP" ] || PREV_ARG="NONE"
  python3 - "$PREV_ARG" "$CUR_TMP" "$SNAP" "$(date +%s)" <<'PYEOF'
import json, sys
prev_path, cur_path, snap_path, now = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prev, prev_ts = {}, None
if prev_path != "NONE":
    pj = json.load(open(prev_path))
    prev_ts = pj.get('_snapshot_ts')
    prev = {c['cell']: c for c in pj.get('cells', []) if 'cell' in c}
cur = json.load(open(cur_path))
elapsed_min = ((now - prev_ts) / 60) if prev_ts else None
ctx = cur.get('context', {})
objs = ctx.get('objectives', [])
hi = ctx.get('headline_obj_idx', 0)
hlabel = ctx.get('headline_short', 'obj%d' % hi)
hdir = objs[hi]['direction'] if hi < len(objs) else 'MINIMIZE'
derived = ctx.get('derived', [])
def fnum(x): return f"{x:.1f}" if x is not None else "—"
if not prev:
    window = "  (baseline — no deltas yet)"
elif elapsed_min is None:
    window = "  (Δ vs last check; window unknown)"
elif elapsed_min < 90:
    window = f"  (Δ over last {elapsed_min:.0f} min)"
else:
    window = f"  (Δ over last {elapsed_min/60:.1f} h)"
print("\n### Summary" + window)
# one "best <obj>" column per objective (MOO shows the time axis too); the
# headline objective is starred. ⬇ marks a new best this window (dir-aware).
def objhdr(o):
    return f"best {o['short']}" + ("*" if o['idx'] == hi else "")
cols = ["cell", "completed", "Δ"] + [objhdr(o) for o in objs] + [d['label'] for d in derived]
print("| " + " | ".join(cols) + " |")
print("|" + "|".join(["---"] * len(cols)) + "|")
total_dc = 0
for c in cur.get('cells', []):
    if 'cell' not in c or 'error' in c:
        continue
    p = prev.get(c['cell'], {})
    dcomp = c['complete'] - p.get('complete', 0) if prev else None
    if dcomp:
        total_dc += dcomp
    dc = (f"+{dcomp}" if dcomp else ("0" if prev else "—"))
    cb = (c.get('best') or []); pb = (p.get('best') or [])
    cells_row = [c['cell'], str(c['complete']), dc]
    for o in objs:
        i = o['idx']
        cur_v = cb[i] if i < len(cb) else None
        prev_v = pb[i] if i < len(pb) else None
        improved = (prev and cur_v is not None and (prev_v is None or (
            cur_v < prev_v - 1e-9 if o['direction'] == 'MINIMIZE'
            else cur_v > prev_v + 1e-9)))
        cells_row.append(fnum(cur_v) + (" ⬇" if improved else ""))
    for d in derived:
        dd = (c.get('derived') or {}).get(d['label'], {})
        cells_row.append(f"{dd['pct']:.0f}%" if dd.get('pct') is not None else "—")
    print("| " + " | ".join(cells_row) + " |")
if prev and elapsed_min and elapsed_min > 0.5:
    rate = total_dc / elapsed_min * 60
    print(f"\n_{total_dc:+d} trials over {elapsed_min:.0f} min ≈ {rate:.1f} trials/hr_")
elif prev and (elapsed_min is not None) and elapsed_min <= 0.5:
    print(f"\n_(only {elapsed_min*60:.0f}s since last check — rerun later for a meaningful rate)_")
cur['_snapshot_ts'] = now
json.dump(cur, open(snap_path, 'w'))
PYEOF
fi

# 5. pull DBs + merge + (re)launch dashboard ---------------------------------
scp -q "$HOST:$REMOTE_DIR/optuna/${STUDY_PREFIX}_*${SUFFIX}.db" "$PULL_DIR/" 2>/dev/null \
  || echo "warn: scp of DBs returned nonzero (some cells may not exist yet)" >&2
COMBINED="$PULL_DIR/${STUDY_PREFIX}_combined.db"
uv run --with optuna python "$SKILL_DIR/merge_dbs.py" "$PULL_DIR" "$COMBINED" "$STUDY_PREFIX" \
  2>/dev/null | grep -E '^merged|^combined' || true
if [ "${NO_DASH:-0}" != "1" ]; then
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/dashboard" 2>/dev/null; then
    echo "dashboard already live -> http://127.0.0.1:$PORT/dashboard (serves $COMBINED)"
  else
    nohup uv run --with optuna-dashboard --with optuna --with gunicorn \
      optuna-dashboard --port "$PORT" --host 127.0.0.1 "sqlite:///$COMBINED" \
      > "$PULL_DIR/dashboard.log" 2>&1 &
    sleep 6
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/dashboard" 2>/dev/null)
    echo "dashboard launched (HTTP $code) -> http://127.0.0.1:$PORT/dashboard"
  fi
fi
