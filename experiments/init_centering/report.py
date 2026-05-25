"""Phase-E reporting layer for the init-centering 18-cell ablation grid.

Three-stage pipeline keeps **plot iteration** decoupled from
**model evaluation**:

1. ``aggregate`` — scan every cell's per-trial ``metrics.json`` and
   join with the matching Optuna study DB; write two flat artifacts:

   * ``summary.csv`` — one row per (cell, trial), scalar columns only.
     Suitable for spreadsheet inspection / pandas.
   * ``records.jsonl`` — full records including per-timestep arrays
     (σ_data²(t) buffer + empirical decomposition, CRPS-per-t,
     GT-latent-JSD-per-t).  This is the canonical source for plots.

2. ``plot`` — *reads* the artifacts (no model touched, no eval re-run)
   and renders the three Phase-E figures + the headline markdown table.

3. ``all`` — convenience: aggregate, then plot.

Per the user's directive ("serialize all computed metrics before
plotting so plots may be iterated on without having to recompute"),
``plot`` only ever reads ``records.jsonl``; agg-side scans are *never*
re-run during plot iteration.

Run end-to-end::

    python -m experiments.init_centering.report all \\
        --sweeps-root runs/sweeps \\
        --optuna-dir runs/optuna \\
        --study-prefix phase_d \\
        --out runs/report/phase_d

Iterate on a plot only::

    python -m experiments.init_centering.report plot \\
        --in runs/report/phase_d \\
        --out runs/report/phase_d/plots

Aggregation discovers cells by walking the Optuna DBs at
``{optuna_dir}/{study_prefix}_*.db``; trial run_dirs are read from the
matching ``{sweeps_root}/{study_prefix}_{cell_name}/{trial_number}/``
layout produced by :mod:`.launch_phase_d`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator

from experiments.init_centering.cells import cell_name as _cell_name
from experiments.init_centering.cells import iter_cells

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    """One Phase-D trial — scalars + per-timestep trajectories.

    The scalar columns are also written to ``summary.csv``; the
    trajectory fields are JSONL-only because CSV can't represent
    variable-length arrays cleanly.
    """

    cell_name: str
    baseline_form: str
    baseline_mode: str
    tracking_mode: str
    trial_number: int
    run_dir: str
    is_control: bool

    # Optuna scalars (None if the cell's DB is missing or the trial
    # was a single-job control run).
    optuna_value: float | None = None
    optuna_state: str | None = None
    duration_sec: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    # Headline metric scalars (from metrics.json).
    stage2_elbo_surrogate: float | None = None
    wallclock_to_target_seconds: float | None = None
    wallclock_to_target_step: int | None = None
    crps_sum_latent_mean: float | None = None
    gt_latent_jsd_mean: float | None = None
    # σ_data²(t) summary scalars (mean of the buffer / decomposition sum).
    sigma_data2_buffer_mean: float | None = None
    sigma_data2_decomposition_sum_mean: float | None = None

    # Per-timestep trajectories (JSONL-only).
    sigma_data2_buffer: list[float] = field(default_factory=list)
    sigma_data2_t_indices: list[int] = field(default_factory=list)
    sigma_data2_component1_per_t: list[float] = field(default_factory=list)
    sigma_data2_component2_per_t: list[float] = field(default_factory=list)
    crps_sum_latent_per_t: list[float] = field(default_factory=list)
    gt_latent_jsd_per_t: list[float] = field(default_factory=list)


_SCALAR_COLUMNS: tuple[str, ...] = (
    "cell_name",
    "baseline_form",
    "baseline_mode",
    "tracking_mode",
    "trial_number",
    "run_dir",
    "is_control",
    "optuna_value",
    "optuna_state",
    "duration_sec",
    "stage2_elbo_surrogate",
    "wallclock_to_target_seconds",
    "wallclock_to_target_step",
    "crps_sum_latent_mean",
    "gt_latent_jsd_mean",
    "sigma_data2_buffer_mean",
    "sigma_data2_decomposition_sum_mean",
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _cell_axes_map() -> dict[str, tuple[str, str, str]]:
    """Reverse-lookup table: cell_name → (form, mode, tracking) triple."""
    return {_cell_name(f, m, t): (f, m, t) for f, m, t in iter_cells()}


# The two control cells reuse the canonical cell's axes.
_CONTROL_CELLS: dict[str, tuple[str, str, str]] = {
    "init_canonical_ctrl_sigma0": ("mlp", "pinned", "per_t"),
    "init_canonical_ctrl_npretrain0": ("mlp", "pinned", "per_t"),
}


def _resolve_cell_axes(cell: str) -> tuple[str, str, str] | None:
    """Return ``(form, mode, tracking)`` for any Phase-D cell name."""
    axes = _cell_axes_map().get(cell)
    if axes is not None:
        return axes
    return _CONTROL_CELLS.get(cell)


def _safe_load_metrics_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def _mean_or_none(xs: Iterable[float]) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _populate_metrics(record: TrialRecord, payload: dict[str, Any]) -> None:
    """Lift the headline-metric fields out of ``metrics.json`` into the record."""
    if "stage2_elbo_surrogate" in payload:
        record.stage2_elbo_surrogate = float(payload["stage2_elbo_surrogate"])

    secs = payload.get("wallclock_to_target_seconds")
    if isinstance(secs, (int, float)):
        record.wallclock_to_target_seconds = float(secs)
    step = payload.get("wallclock_to_target_step")
    if isinstance(step, int):
        record.wallclock_to_target_step = step

    if payload.get("crps_sum_latent_available", False):
        record.crps_sum_latent_mean = float(payload.get("crps_sum_latent_mean", 0.0))
        record.crps_sum_latent_per_t = list(payload.get("crps_sum_latent_per_t", []))

    if payload.get("gt_latent_jsd_available", False):
        record.gt_latent_jsd_mean = float(payload.get("gt_latent_jsd_mean", 0.0))
        record.gt_latent_jsd_per_t = list(payload.get("gt_latent_jsd_per_t", []))

    if payload.get("sigma_data_drift_available", False):
        record.sigma_data2_buffer = list(payload.get("sigma_data2_buffer", []))
        record.sigma_data2_t_indices = list(payload.get("sigma_data2_t_indices", []))
        record.sigma_data2_component1_per_t = list(
            payload.get("sigma_data2_component1_per_t", [])
        )
        record.sigma_data2_component2_per_t = list(
            payload.get("sigma_data2_component2_per_t", [])
        )
        record.sigma_data2_buffer_mean = _mean_or_none(record.sigma_data2_buffer)
        record.sigma_data2_decomposition_sum_mean = _mean_or_none(
            payload.get("sigma_data2_decomposition_sum_per_t", []),
        )


def _populate_optuna(
    record: TrialRecord,
    db_path: str,
    study_name: str,
) -> None:
    """Pull trial value, state, params, duration from the cell's Optuna DB."""
    if not os.path.isfile(db_path):
        log.info("Optuna DB %s missing; skipping trial-level join for %s",
                 db_path, study_name)
        return
    try:
        import optuna
    except ImportError:
        log.warning("optuna not installed; cannot join trial metadata for %s",
                    study_name)
        return
    try:
        study = optuna.load_study(
            study_name=study_name, storage=f"sqlite:///{db_path}",
        )
    except (KeyError, ValueError) as exc:
        log.warning("Could not load Optuna study %s from %s: %s",
                    study_name, db_path, exc)
        return
    by_number = {t.number: t for t in study.trials}
    trial = by_number.get(record.trial_number)
    if trial is None:
        return
    record.optuna_value = (
        float(trial.value) if trial.value is not None else None
    )
    record.optuna_state = trial.state.name if trial.state is not None else None
    record.params = dict(trial.params)
    if trial.datetime_start is not None and trial.datetime_complete is not None:
        delta = trial.datetime_complete - trial.datetime_start
        record.duration_sec = float(delta.total_seconds())


def iter_trial_records(
    sweeps_root: str,
    *,
    optuna_dir: str,
    study_prefix: str,
    include_controls: bool = True,
) -> Iterator[TrialRecord]:
    """Walk every Phase-D cell's sweep dir and yield one record per trial.

    The layout written by :mod:`.launch_phase_d` is::

        {sweeps_root}/{study_prefix}_{cell_name}/{trial_number}/metrics.json
        {optuna_dir}/{study_prefix}_{cell_name}.db

    Missing files / databases degrade gracefully — the trial just
    shows up in the output with ``None`` for the corresponding scalars.
    """
    cells_axes = _cell_axes_map()
    targets: list[str] = sorted(cells_axes.keys())
    if include_controls:
        targets.extend(sorted(_CONTROL_CELLS.keys()))

    for cell in targets:
        sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{cell}")
        if not os.path.isdir(sweep_dir):
            log.info("No sweep dir for %s at %s; skipping.", cell, sweep_dir)
            continue
        axes = _resolve_cell_axes(cell)
        if axes is None:
            log.warning("Unknown cell %s; skipping.", cell)
            continue
        form, mode, tracking = axes
        is_control = cell in _CONTROL_CELLS

        # Trial dirs are numbered subdirs.  Sort numerically.
        trial_dirs: list[tuple[int, str]] = []
        for entry in sorted(os.listdir(sweep_dir)):
            if entry.isdigit():
                trial_dirs.append((int(entry), os.path.join(sweep_dir, entry)))
        if not trial_dirs:
            # Controls run as single jobs (no multirun); the run_dir IS
            # the sweep_dir.
            trial_dirs = [(0, sweep_dir)]

        db_path = os.path.join(optuna_dir, f"{study_prefix}_{cell}.db")
        study_name = f"{study_prefix}_{cell}"

        for trial_number, run_dir in trial_dirs:
            metrics_path = os.path.join(run_dir, "metrics.json")
            payload = _safe_load_metrics_json(metrics_path) if os.path.isfile(metrics_path) else None
            record = TrialRecord(
                cell_name=cell,
                baseline_form=form,
                baseline_mode=mode,
                tracking_mode=tracking,
                trial_number=trial_number,
                run_dir=run_dir,
                is_control=is_control,
            )
            if payload is not None:
                _populate_metrics(record, payload)
            if not is_control:
                _populate_optuna(record, db_path, study_name)
            yield record


def aggregate(
    sweeps_root: str,
    *,
    optuna_dir: str,
    study_prefix: str,
    include_controls: bool = True,
) -> list[TrialRecord]:
    """One-shot aggregation: returns all (cell, trial) records as a list."""
    return list(
        iter_trial_records(
            sweeps_root,
            optuna_dir=optuna_dir,
            study_prefix=study_prefix,
            include_controls=include_controls,
        )
    )


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


SUMMARY_FILENAME = "summary.csv"
RECORDS_FILENAME = "records.jsonl"


def save_artifacts(records: list[TrialRecord], out_dir: str) -> tuple[str, str]:
    """Write ``summary.csv`` (scalars) and ``records.jsonl`` (full records).

    Returns ``(summary_path, records_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, SUMMARY_FILENAME)
    records_path = os.path.join(out_dir, RECORDS_FILENAME)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SCALAR_COLUMNS)
        writer.writeheader()
        for r in records:
            row = {col: getattr(r, col) for col in _SCALAR_COLUMNS}
            writer.writerow(row)

    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), default=float) + "\n")

    return summary_path, records_path


def load_records(records_path: str) -> list[TrialRecord]:
    """Re-hydrate the JSONL records for plot iteration."""
    records: list[TrialRecord] = []
    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(TrialRecord(**payload))
    return records


# ---------------------------------------------------------------------------
# Plot fns (consume the JSONL records — never re-aggregate)
# ---------------------------------------------------------------------------


def _group_by_cell(records: list[TrialRecord]) -> dict[str, list[TrialRecord]]:
    grouped: dict[str, list[TrialRecord]] = {}
    for r in records:
        grouped.setdefault(r.cell_name, []).append(r)
    return grouped


def _import_matplotlib():
    """Import matplotlib lazily so the report module is testable headless."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sigma_data_drift(records: list[TrialRecord], out_path: str) -> str:
    """Per-cell σ_data²(t) trajectories — the Phase-E headline figure.

    One line per cell.  Y-axis: σ_data²(t) (the buffer value averaged
    across trials within the cell).  Cells with no σ_data_drift
    payload are skipped silently.
    """
    plt = _import_matplotlib()
    grouped = _group_by_cell(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for cell, trials in sorted(grouped.items()):
        # Stack each trial's buffer trajectory and average element-wise.
        buffers = [t.sigma_data2_buffer for t in trials if t.sigma_data2_buffer]
        if not buffers:
            continue
        T = min(len(b) for b in buffers)
        mean_buf = [
            sum(b[i] for b in buffers) / len(buffers) for i in range(T)
        ]
        ts = list(range(1, T + 1))  # 1-based per the buffer convention
        ax.plot(ts, mean_buf, label=cell, linewidth=1.0, alpha=0.85)
        plotted += 1

    ax.set_xlabel("latent timestep t")
    ax.set_ylabel(r"$\sigma_{data}^{2}(t)$  (mean across trials)")
    ax.set_title("Phase E — σ_data drift trajectories (per cell)")
    ax.legend(fontsize="x-small", ncols=2, loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Plotted %d cells to %s", plotted, out_path)
    return out_path


def plot_wallclock_to_target(records: list[TrialRecord], out_path: str) -> str:
    """Per-cell wallclock-to-target — bar chart with std error bars."""
    plt = _import_matplotlib()
    grouped = _group_by_cell(records)

    cells: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for cell in sorted(grouped.keys()):
        secs = [
            t.wallclock_to_target_seconds
            for t in grouped[cell]
            if t.wallclock_to_target_seconds is not None
        ]
        if not secs:
            continue
        cells.append(cell)
        means.append(sum(secs) / len(secs))
        if len(secs) > 1:
            mu = means[-1]
            var = sum((x - mu) ** 2 for x in secs) / (len(secs) - 1)
            stds.append(var**0.5)
        else:
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = range(len(cells))
    ax.bar(xs, means, yerr=stds, capsize=3, alpha=0.85)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cells, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("wallclock to target (s, mean ± std)")
    ax.set_title("Phase E — wallclock-to-target by cell")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}g}"


def write_headline_table(records: list[TrialRecord], out_path: str) -> str:
    """Markdown headline table: one row per cell, 5 metrics + best objective.

    For sweep cells (non-control) the per-cell aggregate is taken from
    the **best trial** (min ``optuna_value``).  Controls have a single
    trial and are listed verbatim.
    """
    grouped = _group_by_cell(records)

    lines: list[str] = [
        "# Phase E — headline metrics (init-centering 18-cell grid)",
        "",
        "| cell | form | mode | tracking | best stage2_elbo_surrogate | wallclock_to_target (s) | crps_sum_latent | gt_latent_jsd | σ_data² buffer mean | n_trials |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in sorted(grouped.keys()):
        trials = grouped[cell]
        # Pick the best trial (min objective if available, else first non-null).
        objective_trials = [t for t in trials if t.optuna_value is not None]
        if objective_trials:
            best = min(objective_trials, key=lambda t: t.optuna_value)  # type: ignore[arg-type]
        else:
            best = trials[0]
        lines.append(
            "| {cell} | {form} | {mode} | {tracking} | {elbo} | {wc} | {crps} | {jsd} | {sd} | {n} |".format(
                cell=cell,
                form=best.baseline_form,
                mode=best.baseline_mode,
                tracking=best.tracking_mode,
                elbo=_fmt(best.stage2_elbo_surrogate),
                wc=_fmt(best.wallclock_to_target_seconds, 3),
                crps=_fmt(best.crps_sum_latent_mean),
                jsd=_fmt(best.gt_latent_jsd_mean),
                sd=_fmt(best.sigma_data2_buffer_mean),
                n=len(trials),
            )
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_aggregate(args: argparse.Namespace) -> int:
    records = aggregate(
        args.sweeps_root,
        optuna_dir=args.optuna_dir,
        study_prefix=args.study_prefix,
        include_controls=not args.exclude_controls,
    )
    summary, jsonl = save_artifacts(records, args.out)
    print(f"Aggregated {len(records)} trials")
    print(f"  scalars : {summary}")
    print(f"  records : {jsonl}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    records_path = args.records or os.path.join(args.in_dir, RECORDS_FILENAME)
    records = load_records(records_path)
    out_dir = args.out or os.path.join(args.in_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_sigma_data_drift(records, os.path.join(out_dir, "sigma_data_drift.png"))
    plot_wallclock_to_target(records, os.path.join(out_dir, "wallclock_to_target.png"))
    write_headline_table(records, os.path.join(out_dir, "headline_table.md"))
    print(f"Wrote plots + headline table to {out_dir}")
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    rc = _cmd_aggregate(args)
    if rc != 0:
        return rc
    plot_args = argparse.Namespace(
        in_dir=args.out,
        records=None,
        out=os.path.join(args.out, "plots"),
    )
    return _cmd_plot(plot_args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.report",
        description="Phase-E reporting pipeline: aggregate → save → plot.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    agg_common = argparse.ArgumentParser(add_help=False)
    agg_common.add_argument(
        "--sweeps-root", default="runs/sweeps",
        help="Root dir holding {study_prefix}_{cell}/ subdirs (default runs/sweeps).",
    )
    agg_common.add_argument(
        "--optuna-dir", default="runs/optuna",
        help="Dir holding the per-cell {study_prefix}_{cell}.db files (default runs/optuna).",
    )
    agg_common.add_argument(
        "--study-prefix", default="phase_d",
        help="Study-name + sweep-dir prefix (default phase_d).",
    )
    agg_common.add_argument(
        "--exclude-controls", action="store_true",
        help="Skip the sigma0 / npretrain0 control runs.",
    )
    agg_common.add_argument(
        "--out", required=True,
        help="Where to write summary.csv + records.jsonl.",
    )

    p_agg = sub.add_parser("aggregate", parents=[agg_common],
                            help="Scan disk + Optuna DBs, write summary.csv + records.jsonl.")

    p_plot = sub.add_parser("plot", help="Render plots + headline table from saved artifacts.")
    p_plot.add_argument("--in", dest="in_dir", default=None,
                        help="Dir holding records.jsonl (alternative to --records).")
    p_plot.add_argument("--records", default=None,
                        help="Explicit path to records.jsonl.")
    p_plot.add_argument("--out", default=None,
                        help="Where to write plots (default <in>/plots).")

    p_all = sub.add_parser("all", parents=[agg_common],
                            help="Aggregate then plot in one go.")

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "plot":
        if args.in_dir is None and args.records is None:
            parser.error("plot: one of --in or --records is required")

    cmd = {
        "aggregate": _cmd_aggregate,
        "plot": _cmd_plot,
        "all": _cmd_all,
    }[args.cmd]
    return cmd(args)


if __name__ == "__main__":
    sys.exit(main())
