#!/usr/bin/env python3
"""Build the display *context* for a DDSSM Optuna experiment by introspection.

Runs ON the cluster. Given a study prefix + suffix, it figures out — with NO
hardcoded per-experiment knowledge — what the table should show:

  * objectives: read ``experiment.objective.specs[]`` from a trial's
    ``resolved_config.yaml`` (metric name + penalty), paired in order with the
    Optuna study's ``directions`` (MIN/MAX). If the study set ``metric_names``
    those win. A short, human label is derived heuristically from the name.
  * headline objective: the quality/loss axis to feature as "best X" in the
    summary table — the objective whose name is NOT a time/wallclock axis
    (prefer the last MINIMIZE); falls back to objective 0.
  * derived columns: sniff a trial's ``metrics.json`` keys. A metric of the form
    ``*_to_target_seconds`` (null-on-miss) with a sibling ``*_to_target_step``
    enables a hit-rate / crossing-step column; the target value is pulled from
    ``eval.kwargs``.
  * param_names, dataset mode, the trial the context was built from.

Emits the context as one JSON line (``__JSON__`` prefixed). The local driver
caches it as the per-experiment profile.

Usage (on the cluster):
    python build_context.py --remote-dir <dir> --study-prefix <p> --suffix __mv
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse


def short_label(name):
    """Derive a short, human display label for an objective/metric name.

    Maps known families (ELBO/CRPS/MAE/RMSE/JSD/time-to-target) to fixed
    labels; otherwise falls back to the last one or two underscore tokens.
    """
    n = name.lower()
    if "elbo" in n:
        return "ELBO"
    if "crps" in n:
        return "CRPS"
    if "mae" in n:
        return "MAE"
    if "rmse" in n:
        return "RMSE"
    if "jsd" in n:
        return "JSD"
    if "seconds" in n or ("wallclock" in n and "to_target" in n):
        return "t→tgt(s)"
    # generic: last 1-2 underscore tokens, trimmed
    toks = name.split("_")
    return "_".join(toks[-2:])[:12]


def is_time_axis(name):
    """Return True if the metric name denotes a wallclock/time axis."""
    n = name.lower()
    return ("seconds" in n) or ("wallclock" in n) or n.endswith("_time")


def main():
    """Introspect the study + a resolved config and emit the display context.

    Prints a human summary to stderr and one ``__JSON__``-prefixed context line
    to stdout for the local driver to cache as the per-experiment profile.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-dir", default="~/diffusion-driven-state-space-models")
    ap.add_argument("--study-prefix", required=True)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    remote = os.path.expanduser(args.remote_dir)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # pick a representative cell DB + a trial dir with resolved_config.yaml
    # Search runs/optuna/ first (ddssm.launch default), then optuna/ (legacy).
    dbs = []
    for _subdir in ("runs/optuna", "optuna"):
        dbs = sorted(glob.glob(os.path.join(
            remote, _subdir, f"{args.study_prefix}_*{args.suffix}.db")))
        if dbs:
            break
    if not dbs:
        print(f"FATAL: no DBs match {args.study_prefix}_*{args.suffix}.db", file=sys.stderr)
        raise SystemExit(2)
    rep_study = os.path.basename(dbs[0])[:-3]
    st = optuna.load_study(study_name=rep_study, storage=f"sqlite:///{dbs[0]}")
    directions = [d.name for d in st.directions]
    metric_names = list(getattr(st, "metric_names", None) or [])
    param_names = sorted({k for t in st.get_trials(deepcopy=False)
                          for k in t.params})

    # find a resolved_config.yaml (any cell, any trial; also try *_topup)
    rc = None
    for base in [rep_study] + [os.path.basename(d)[:-3] for d in dbs]:
        for sd in (base, base + "_topup"):
            hits = glob.glob(os.path.join(remote, "sweeps", sd, "*",
                                          "resolved_config.yaml"))
            if hits:
                rc = hits[0]
                break
        if rc:
            break

    obj_names, target_value, eval_metrics, data_mode, penalties = [], None, [], None, []
    if rc:
        import yaml
        c = yaml.safe_load(open(rc))
        exp = c.get("experiment", c)
        obj = exp.get("objective") or {}
        specs = obj.get("specs") if isinstance(obj, dict) else None
        if specs:
            for s in specs:
                obj_names.append(s.get("metric"))
                penalties.append(s.get("penalty"))
        ev = exp.get("eval") or {}
        eval_metrics = ev.get("metrics") or []
        kw = ev.get("kwargs") or {}
        for v in kw.values():
            if isinstance(v, dict) and "target_value" in v:
                target_value = v["target_value"]
        data_mode = (exp.get("data") or {}).get("mode")

    # metric_names (if study set them) win over config-derived names
    names = metric_names or obj_names or [f"obj{i}" for i in range(len(directions))]
    objectives = []
    for i, d in enumerate(directions):
        nm = names[i] if i < len(names) else f"obj{i}"
        objectives.append({
            "idx": i, "name": nm, "direction": d, "short": short_label(nm or ""),
            "penalty": penalties[i] if i < len(penalties) else None,
            "time_axis": is_time_axis(nm or ""),
        })

    # headline = preferred quality axis: last MINIMIZE non-time, else obj0
    headline = 0
    for o in objectives:
        if o["direction"] == "MINIMIZE" and not o["time_axis"]:
            headline = o["idx"]
    head_short = objectives[headline]["short"] if objectives else "obj0"

    # derived columns from metrics.json keys
    derived = []
    mj = glob.glob(os.path.join(remote, "sweeps", rep_study, "*", "metrics.json"))
    keys = set()
    if mj:
        try:
            keys = set(json.load(open(mj[0])).keys())
        except Exception:
            keys = set()
    for k in sorted(keys):
        if k.endswith("_to_target_seconds"):
            stem = k[: -len("_seconds")]
            step_key = stem + "_step"
            derived.append({
                "type": "hit_rate", "sec_key": k,
                "step_key": step_key if step_key in keys else None,
                "label": "hit%", "target_value": target_value,
            })

    ctx = {
        "study_prefix": args.study_prefix, "suffix": args.suffix,
        "remote_dir": args.remote_dir,
        "objectives": objectives, "headline_obj_idx": headline,
        "headline_short": head_short, "derived": derived,
        "param_names": param_names, "eval_metrics": eval_metrics,
        "data_mode": data_mode, "built_from": rc,
        "n_cells": len(dbs),
    }
    print("Context for", args.study_prefix, file=sys.stderr)
    print(f"  objectives: " + ", ".join(
        f"{o['short']}({o['direction'][:3]}){'*' if o['idx']==headline else ''}"
        for o in objectives), file=sys.stderr)
    print(f"  derived: {[d['label'] for d in derived] or 'none'}   "
          f"target={target_value}   data={data_mode}", file=sys.stderr)
    print("__JSON__" + json.dumps(ctx))


if __name__ == "__main__":
    main()
