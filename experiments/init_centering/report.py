"""Phase-E reporting layer for the init-centering ablation grid.

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
layout produced by ``python -m ddssm.launch init_centering`` (see
:mod:`ddssm.launch`).
"""

from __future__ import annotations

import os
import csv
import sys
import json
from typing import Any, Iterable, Iterator
import logging
import argparse
from dataclasses import field, asdict, dataclass

from experiments.init_centering.study import INIT_CENTERING_STUDY

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
    dataset: str
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
    # Self-referential diagnostic (time to first reach 90% of own descent).
    # Always defined for any non-degenerate trial, unlike the fixed-target
    # wallclock above. Used in the report's MOO diagnostic plot.
    wallclock_to_relative_target_seconds: float | None = None
    wallclock_to_relative_target_step: int | None = None
    wallclock_to_relative_target_implied_target: float | None = None
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
    "dataset",
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
    "wallclock_to_relative_target_seconds",
    "wallclock_to_relative_target_step",
    "wallclock_to_relative_target_implied_target",
    "crps_sum_latent_mean",
    "gt_latent_jsd_mean",
    "sigma_data2_buffer_mean",
    "sigma_data2_decomposition_sum_mean",
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


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

    rsecs = payload.get("wallclock_to_relative_target_seconds")
    if isinstance(rsecs, (int, float)):
        record.wallclock_to_relative_target_seconds = float(rsecs)
    rstep = payload.get("wallclock_to_relative_target_step")
    if isinstance(rstep, int):
        record.wallclock_to_relative_target_step = rstep
    impl = payload.get("wallclock_to_relative_target_implied_target")
    if isinstance(impl, (int, float)):
        record.wallclock_to_relative_target_implied_target = float(impl)

    if payload.get("crps_sum_latent_available"):
        record.crps_sum_latent_mean = float(payload.get("crps_sum_latent_mean", 0.0))
        record.crps_sum_latent_per_t = list(payload.get("crps_sum_latent_per_t", []))

    if payload.get("gt_latent_jsd_available"):
        record.gt_latent_jsd_mean = float(payload.get("gt_latent_jsd_mean", 0.0))
        record.gt_latent_jsd_per_t = list(payload.get("gt_latent_jsd_per_t", []))

    if payload.get("sigma_data_drift_available"):
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


def _load_optuna_trials(db_path: str, study_name: str) -> list[Any] | None:
    """Load all trials for a study; ``None`` if the DB is missing/broken."""
    if not os.path.isfile(db_path):
        log.info("Optuna DB %s missing; skipping trial-level join for %s",
                 db_path, study_name)
        return None
    try:
        import optuna
    except ImportError:
        log.warning("optuna not installed; cannot join trial metadata for %s",
                    study_name)
        return None
    try:
        study = optuna.load_study(
            study_name=study_name, storage=f"sqlite:///{db_path}",
        )
    except (KeyError, ValueError) as exc:
        log.warning("Could not load Optuna study %s from %s: %s",
                    study_name, db_path, exc)
        return None
    return list(study.trials)


def _apply_optuna_trial(record: TrialRecord, trial: Any) -> None:
    """Copy value/state/params/duration off an ``optuna.Trial`` into the record.

    Handles both single- and multi-objective trials. For MOO trials,
    ``optuna_value`` holds the *first* objective (the wallclock axis in
    the round-1 MOO setup) so existing "best by optuna_value" callers
    keep working. The full vector is read via ``trial.values`` from the
    DB when ``plot_pareto_front`` needs it.
    """
    try:
        v = trial.values  # multi-objective list
    except AttributeError:
        v = None
    if v is not None and isinstance(v, (list, tuple)) and len(v) > 0:
        record.optuna_value = float(v[0]) if v[0] is not None else None
    else:
        # Single-objective path. ``trial.value`` raises RuntimeError on
        # MOO studies, hence the try/except guard.
        try:
            record.optuna_value = (
                float(trial.value) if trial.value is not None else None
            )
        except RuntimeError:
            record.optuna_value = None
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
    dataset: str | None = None,
) -> Iterator[TrialRecord]:
    """Walk every study point's sweep dir and yield one record per trial.

    The layout written by ``python -m ddssm.launch init_centering`` is::

        {sweeps_root}/{study_prefix}_{cell}__{dataset}/{trial_number}/metrics.json
        {optuna_dir}/{study_prefix}_{cell}__{dataset}.db

    where ``{cell}__{dataset}`` is the registered study-point name. By
    default every point (all datasets) is walked; pass ``dataset`` to
    restrict to one. Missing files / databases degrade gracefully — the
    trial just shows up with ``None`` for the corresponding scalars.
    """
    points = INIT_CENTERING_STUDY.points
    if dataset is not None:
        points = tuple(p for p in points if p.tags["dataset"] == dataset)

    for point in points:
        cell_key = point.name  # e.g. init_mlp_pinned_per_t__1d
        sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{cell_key}")
        if not os.path.isdir(sweep_dir):
            log.info("No sweep dir for %s at %s; skipping.", point.name, sweep_dir)
            continue
        cell = point.tags["cell"]
        form = point.tags["baseline_form"]
        mode = point.tags["baseline_mode"]
        tracking = point.tags["tracking_mode"]
        ds_label = point.tags["dataset"]
        # ``is_control`` retained on the record as an extension point
        # for future ablation-panel records; the original control
        # presets were dropped per ADR-0002.
        is_control = False

        # Trial dirs may be flat integers (``0/``, ``1/``, ...) from a
        # single-worker multirun, or worker-prefixed names like
        # ``w0_0/``, ``w0_1/``, ``w1_0/`` from the multi-worker
        # ablation script. Discover them by listing subdirs that
        # contain a ``metrics.json``; sort by the directory's start
        # time (``.hydra/hydra.yaml`` mtime) so the order matches the
        # Optuna DB's ``datetime_start`` ordering — which is the join
        # key when subdir names don't carry the global trial number.
        candidates: list[tuple[float, str, str]] = []
        for entry in os.listdir(sweep_dir):
            p = os.path.join(sweep_dir, entry)
            if not os.path.isdir(p):
                continue
            if not os.path.isfile(os.path.join(p, "metrics.json")):
                continue
            hydra_yaml = os.path.join(p, ".hydra", "hydra.yaml")
            start_time = (
                os.path.getmtime(hydra_yaml)
                if os.path.isfile(hydra_yaml) else os.path.getmtime(p)
            )
            candidates.append((start_time, entry, p))
        candidates.sort()

        db_path = os.path.join(optuna_dir, f"{study_prefix}_{cell_key}.db")
        study_name = f"{study_prefix}_{cell_key}"
        db_trials_sorted: list = []
        if not is_control:
            loaded = _load_optuna_trials(db_path, study_name)
            if loaded is not None:
                db_trials_sorted = sorted(
                    [t for t in loaded if t.datetime_start is not None],
                    key=lambda t: t.datetime_start,
                )

        if not candidates:
            # No multirun structure (single-job cell run); the run_dir
            # IS the sweep_dir. Preserve the legacy single-record path.
            trial_dirs = [(0, sweep_dir)]
            for trial_number, run_dir in trial_dirs:
                metrics_path = os.path.join(run_dir, "metrics.json")
                payload = _safe_load_metrics_json(metrics_path) if os.path.isfile(metrics_path) else None
                record = TrialRecord(
                    cell_name=cell, dataset=ds_label, baseline_form=form,
                    baseline_mode=mode, tracking_mode=tracking,
                    trial_number=trial_number, run_dir=run_dir,
                    is_control=is_control,
                )
                if payload is not None:
                    _populate_metrics(record, payload)
                yield record
            continue

        for i, (_, entry, run_dir) in enumerate(candidates):
            # Use the matching DB trial's number when zip-by-start-time
            # works; otherwise fall back to the integer in the subdir
            # name (back-compat) or the position index.
            if i < len(db_trials_sorted):
                trial_number = int(db_trials_sorted[i].number)
            elif entry.isdigit():
                trial_number = int(entry)
            else:
                trial_number = i

            metrics_path = os.path.join(run_dir, "metrics.json")
            payload = _safe_load_metrics_json(metrics_path) if os.path.isfile(metrics_path) else None
            record = TrialRecord(
                cell_name=cell, dataset=ds_label, baseline_form=form,
                baseline_mode=mode, tracking_mode=tracking,
                trial_number=trial_number, run_dir=run_dir,
                is_control=is_control,
            )
            if payload is not None:
                _populate_metrics(record, payload)
            if not is_control and i < len(db_trials_sorted):
                _apply_optuna_trial(record, db_trials_sorted[i])
            yield record


def aggregate(
    sweeps_root: str,
    *,
    optuna_dir: str,
    study_prefix: str,
    dataset: str | None = None,
) -> list[TrialRecord]:
    """One-shot aggregation: returns all (cell, trial) records as a list."""
    return list(
        iter_trial_records(
            sweeps_root,
            optuna_dir=optuna_dir,
            study_prefix=study_prefix,
            dataset=dataset,
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
    """Group trials by their study point (``cell__dataset``).

    Keying on cell × dataset keeps the two datasets in separate rows/lines —
    1d and mv aren't comparable, so mixing them per cell would mislead.
    """
    grouped: dict[str, list[TrialRecord]] = {}
    for r in records:
        key = f"{r.cell_name}__{r.dataset}" if r.dataset else r.cell_name
        grouped.setdefault(key, []).append(r)
    return grouped


def _import_matplotlib():
    """Import matplotlib lazily so the report module is testable headless."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _best_trial(trials: list[TrialRecord]) -> TrialRecord:
    """Return the trial with the lowest ``optuna_value`` (or the first if none)."""
    objective_trials = [t for t in trials if t.optuna_value is not None]
    if objective_trials:
        return min(objective_trials, key=lambda t: t.optuna_value)  # type: ignore[arg-type]
    return trials[0]


def plot_sigma_data_drift(records: list[TrialRecord], out_path: str) -> str:
    """Per-cell σ_data²(t) trajectories: best-trial heavy line + per-trial fan.

    For each cell, plots the *best-by-objective* trial's σ_data
    trajectory as a heavy coloured line, with every other trial in the
    cell drawn behind it as a thin, low-alpha line in the same colour
    ("fan plot"). This shows both the headline-cell value and the
    distribution of trial outcomes inside the cell — the previous
    cross-trial mean was misleading because trials have different
    sweep knobs.

    Cells with no σ_data_drift payload are skipped silently.
    """
    plt = _import_matplotlib()
    grouped = _group_by_cell(records)
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for idx, (cell, trials) in enumerate(sorted(grouped.items())):
        trials_with_buf = [t for t in trials if t.sigma_data2_buffer]
        if not trials_with_buf:
            continue
        colour = cmap(idx % 20)
        best = _best_trial(trials_with_buf)
        # Fan of all trials behind the best — thin, low alpha, no label.
        for t in trials_with_buf:
            buf = t.sigma_data2_buffer
            ts = list(range(1, len(buf) + 1))
            ax.plot(ts, buf, color=colour, linewidth=0.5, alpha=0.25)
        # Best-trial trajectory as the heavy headline line.
        best_buf = best.sigma_data2_buffer
        ts = list(range(1, len(best_buf) + 1))
        ax.plot(ts, best_buf, color=colour, linewidth=2.0, alpha=0.95, label=cell)
        plotted += 1

    ax.set_xlabel("latent timestep t")
    ax.set_ylabel(r"$\sigma_{data}^{2}(t)$  (best trial heavy; all trials faint)")
    ax.set_title("Phase E — σ_data drift trajectories (best trial + fan)")
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

    Per-cell row pulled from the **best trial** (min ``optuna_value``).
    See :func:`write_distribution_panel` for the sibling table that
    reports the distribution of trial outcomes (median + IQR) per
    cell — the two panels together let a reader see "best achievable"
    next to "typical across the sweep" per CONTEXT.md § stage2_elbo_surrogate.
    """
    grouped = _group_by_cell(records)

    lines: list[str] = [
        "# Phase E — headline metrics (init-centering ablation grid)",
        "",
        "| cell | form | mode | tracking | best stage2_elbo_surrogate | wallclock_to_target (s) | crps_sum_latent | gt_latent_jsd | σ_data² buffer mean | n_trials |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in sorted(grouped.keys()):
        trials = grouped[cell]
        best = _best_trial(trials)
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
# Distribution panel (median + IQR per cell, per metric)
# ---------------------------------------------------------------------------


_DISTRIBUTION_METRICS: tuple[tuple[str, str, int], ...] = (
    # (field_name, display_label, digits)
    ("stage2_elbo_surrogate", "stage2_elbo_surrogate", 4),
    ("wallclock_to_target_seconds", "wallclock_to_target (s)", 3),
    ("crps_sum_latent_mean", "crps_sum_latent", 4),
    ("gt_latent_jsd_mean", "gt_latent_jsd", 4),
    ("sigma_data2_buffer_mean", "σ_data² buffer mean", 4),
)


def _median_iqr(values: list[float]) -> tuple[float, float, float] | None:
    """Return ``(median, q1, q3)`` of ``values``; ``None`` if empty."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    n = len(values)
    s = sorted(values)

    def _pct(p: float) -> float:
        # Linear interpolation between order statistics (NumPy "linear" rule).
        if n == 1:
            return s[0]
        k = p * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        frac = k - f
        return s[f] * (1 - frac) + s[c] * frac

    return _pct(0.5), _pct(0.25), _pct(0.75)


def _fmt_median_iqr(field: str, trials: list[TrialRecord], digits: int) -> str:
    vals = [getattr(t, field) for t in trials]
    summary = _median_iqr([v for v in vals if v is not None])
    if summary is None:
        return "-"
    med, q1, q3 = summary
    return f"{med:.{digits}g} [{q1:.{digits}g}, {q3:.{digits}g}]"


def write_distribution_panel(records: list[TrialRecord], out_path: str) -> str:
    """Markdown sibling-of-headline: per-cell median + IQR per metric.

    Reports the distribution of trial outcomes within each cell so the
    reader can compare "best achievable" (the headline table) against
    "typical" (this panel) — wide IQR signals an unstable Optuna study,
    a tight IQR signals the cell is converged and the headline number
    is representative.
    """
    grouped = _group_by_cell(records)

    header_cols = ["cell", "form", "mode", "tracking", "n_trials"] + [
        f"{label} (median [IQR])" for _, label, _ in _DISTRIBUTION_METRICS
    ]
    sep = "|".join(["---"] * len(header_cols))
    lines: list[str] = [
        "# Phase E — per-cell distribution panel (median [Q1, Q3] across trials)",
        "",
        "| " + " | ".join(header_cols) + " |",
        f"|{sep}|",
    ]
    for cell in sorted(grouped.keys()):
        trials = grouped[cell]
        row = [
            cell,
            trials[0].baseline_form,
            trials[0].baseline_mode,
            trials[0].tracking_mode,
            str(len(trials)),
        ]
        for field, _label, digits in _DISTRIBUTION_METRICS:
            row.append(_fmt_median_iqr(field, trials, digits))
        lines.append("| " + " | ".join(row) + " |")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Pairwise-comparison panels (from init-experiment.org § Pairwise-comparison stories)
# ---------------------------------------------------------------------------


def _records_by_axes(
    records: list[TrialRecord],
) -> dict[tuple[str, str, str], TrialRecord]:
    """Return ``{(form, mode, tracking): best_trial_for_cell}``."""
    grouped = _group_by_cell(records)
    out: dict[tuple[str, str, str], TrialRecord] = {}
    for trials in grouped.values():
        best = _best_trial(trials)
        out[best.baseline_form, best.baseline_mode, best.tracking_mode] = best
    return out


_BASELINE_FORMS = ("zero", "persistence", "linear", "mlp")
_BASELINE_MODES = ("pinned", "learnable")
_TRACKING_MODES = ("fixed", "per_t")


def plot_baseline_form_ablation(
    records: list[TrialRecord], out_path: str,
) -> str:
    """Baseline-form ablation: ``stage2_elbo_surrogate`` by form, faceted by (mode, tracking).

    One subplot per (mode, tracking) cell; x-axis is the four baseline
    forms (zero/persistence/linear/mlp). Reveals whether residual
    decomposition (persistence vs zero), parametric baseline (linear vs
    persistence), and nonlinear capacity (mlp vs linear) help —
    init-experiment.org § Pairwise-comparison stories, *Baseline form*.
    """
    plt = _import_matplotlib()
    by_axes = _records_by_axes(records)

    cols = _TRACKING_MODES
    rows = _BASELINE_MODES
    fig, axes = plt.subplots(
        len(rows), len(cols),
        figsize=(3.5 * len(cols), 3.0 * len(rows)),
        sharey=True, squeeze=False,
    )
    for ri, mode in enumerate(rows):
        for ci, tracking in enumerate(cols):
            ax = axes[ri][ci]
            ys = []
            forms_plotted = []
            for form in _BASELINE_FORMS:
                # Pinned-only cells: skip the (param-free, learnable) entry
                # since it auto-degenerates and the data lives under pinned.
                lookup_mode = "pinned" if form in ("zero", "persistence") and mode == "learnable" else mode
                rec = by_axes.get((form, lookup_mode, tracking))
                if rec is None or rec.stage2_elbo_surrogate is None:
                    continue
                ys.append(rec.stage2_elbo_surrogate)
                forms_plotted.append(form)
            xs = range(len(forms_plotted))
            ax.bar(xs, ys, alpha=0.85)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(forms_plotted, fontsize=8, rotation=30, ha="right")
            ax.set_title(f"mode={mode}, tracking={tracking}", fontsize=9)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
        axes[ri][0].set_ylabel("best stage2_elbo_surrogate")
    fig.suptitle("Phase E — baseline-form ablation (best trial per cell)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_baseline_mode_ablation(
    records: list[TrialRecord], out_path: str,
) -> str:
    """Baseline-mode ablation: Pinned vs Learnable for parametric forms.

    One subplot per (form, tracking) for ``form ∈ {linear, mlp}`` (the
    two parametric forms that admit Learnable). x-axis is the two
    modes. Tests whether the soft anchor outperforms hard pinning —
    init-experiment.org § Pairwise-comparison stories, *Baseline mode*.
    """
    plt = _import_matplotlib()
    by_axes = _records_by_axes(records)

    cols = _TRACKING_MODES
    rows = ("linear", "mlp")
    fig, axes = plt.subplots(
        len(rows), len(cols),
        figsize=(3.5 * len(cols), 3.0 * len(rows)),
        sharey=True, squeeze=False,
    )
    for ri, form in enumerate(rows):
        for ci, tracking in enumerate(cols):
            ax = axes[ri][ci]
            ys, labels = [], []
            for mode in _BASELINE_MODES:
                rec = by_axes.get((form, mode, tracking))
                if rec is None or rec.stage2_elbo_surrogate is None:
                    continue
                ys.append(rec.stage2_elbo_surrogate)
                labels.append(mode)
            xs = range(len(labels))
            ax.bar(xs, ys, alpha=0.85, color=["#1f77b4", "#ff7f0e"][: len(labels)])
            ax.set_xticks(list(xs))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(f"form={form}, tracking={tracking}", fontsize=9)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
        axes[ri][0].set_ylabel("best stage2_elbo_surrogate")
    fig.suptitle("Phase E — baseline-mode ablation (Pinned vs Learnable)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_tracking_mode_ablation(
    records: list[TrialRecord], out_path: str,
) -> str:
    """Tracking-mode ablation: fixed / per_t at every (form, mode).

    One subplot per (form, mode); x-axis is the two tracking modes.
    Tests whether t-resolved tracking pays off vs. holding σ_data² = 1
    — init-experiment.org § Pairwise-comparison stories, *Tracking mode*.
    """
    plt = _import_matplotlib()
    by_axes = _records_by_axes(records)

    # (form, mode) panels, dropping the (param-free, learnable) pairs
    # since those degenerate to pinned and would duplicate the pinned panel.
    panels: list[tuple[str, str]] = []
    for form in _BASELINE_FORMS:
        for mode in _BASELINE_MODES:
            if form in ("zero", "persistence") and mode == "learnable":
                continue
            panels.append((form, mode))

    n_cols = 3
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        sharey=True, squeeze=False,
    )
    for i, (form, mode) in enumerate(panels):
        ax = axes[i // n_cols][i % n_cols]
        ys, labels = [], []
        for tracking in _TRACKING_MODES:
            rec = by_axes.get((form, mode, tracking))
            if rec is None or rec.stage2_elbo_surrogate is None:
                continue
            ys.append(rec.stage2_elbo_surrogate)
            labels.append(tracking)
        xs = range(len(labels))
        ax.bar(xs, ys, alpha=0.85)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_title(f"form={form}, mode={mode}", fontsize=9)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    # Blank out unused subplots.
    for j in range(len(panels), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    for r in range(n_rows):
        axes[r][0].set_ylabel("best stage2_elbo_surrogate")
    fig.suptitle("Phase E — tracking-mode ablation (fixed / per_t)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _pareto_front_indices(points: list[tuple[float, float]]) -> list[int]:
    """Indices of (x, y) points that are non-dominated under minimise-both.

    Naive O(n²) sweep — fine for the 30-60 trials-per-cell scale.
    """
    n = len(points)
    on_front = [True] * n
    for i in range(n):
        if not on_front[i]:
            continue
        xi, yi = points[i]
        for j in range(n):
            if i == j or not on_front[j]:
                continue
            xj, yj = points[j]
            # j strictly dominates i if j <= i on both axes and < on at least one.
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                on_front[i] = False
                break
    return [i for i, keep in enumerate(on_front) if keep]


def plot_pareto_front(
    records: list[TrialRecord], out_path: str,
) -> str:
    """Per-cell Pareto front: (wallclock_to_target_seconds, stage2_elbo_surrogate).

    Round-1 MOO sweeps optimise these two axes jointly via NSGA-II.
    The front per cell shows the trade-off between *speed* to a fixed
    ELBO threshold and *depth* of final fit. Trials that never hit the
    threshold (penalty = full training wallclock) cluster at the right
    edge of the x-axis; trials with the deepest fits cluster at the
    bottom of the y-axis.
    """
    plt = _import_matplotlib()
    grouped = _group_by_cell(records)
    cells = sorted(grouped.keys())
    if not cells:
        return out_path

    n_cols = min(3, len(cells))
    n_rows = (len(cells) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    for ci, cell in enumerate(cells):
        ax = axes[ci // n_cols][ci % n_cols]
        trials = grouped[cell]
        pts: list[tuple[float, float]] = []
        for t in trials:
            # ``wallclock_to_target_seconds`` is ``None`` for trials that
            # never reached the fixed target. Under MOO the penalty value
            # (full training wallclock) is what Optuna stored as the
            # first objective; surface that via ``optuna_value`` so
            # misses are visible on the front, not silently dropped.
            wc = t.wallclock_to_target_seconds
            if wc is None:
                wc = t.optuna_value
            if wc is None or t.stage2_elbo_surrogate is None:
                continue
            pts.append((float(wc), float(t.stage2_elbo_surrogate)))
        if not pts:
            ax.set_title(f"{cell}  (no data)", fontsize=9)
            ax.axis("off")
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=14, alpha=0.55, color="0.5", label="dominated")
        front_idx = _pareto_front_indices(pts)
        if front_idx:
            fxs = [pts[i][0] for i in front_idx]
            fys = [pts[i][1] for i in front_idx]
            order = sorted(range(len(fxs)), key=lambda k: fxs[k])
            fxs = [fxs[k] for k in order]
            fys = [fys[k] for k in order]
            ax.plot(fxs, fys, "-o", color="C3", markersize=6,
                    linewidth=1.5, label=f"front (n={len(fxs)})")
        ax.set_xlabel("wallclock to target (s)", fontsize=8)
        ax.set_ylabel("stage2_elbo_surrogate", fontsize=8)
        ax.set_title(cell.replace("init_", ""), fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(fontsize=7, loc="best")
    for j in range(len(cells), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(
        "Round-1 MOO Pareto front per cell — "
        "(wallclock_to_target, stage2_elbo_surrogate), both minimise"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_aggregate(args: argparse.Namespace) -> int:
    records = aggregate(
        args.sweeps_root,
        optuna_dir=args.optuna_dir,
        study_prefix=args.study_prefix,
        dataset=args.dataset,
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
    write_distribution_panel(records, os.path.join(out_dir, "distribution_panel.md"))
    plot_baseline_form_ablation(
        records, os.path.join(out_dir, "pairwise_baseline_form.png"),
    )
    plot_baseline_mode_ablation(
        records, os.path.join(out_dir, "pairwise_baseline_mode.png"),
    )
    plot_tracking_mode_ablation(
        records, os.path.join(out_dir, "pairwise_tracking_mode.png"),
    )
    plot_pareto_front(records, os.path.join(out_dir, "pareto_front.png"))
    print(f"Wrote plots + headline + distribution + pairwise panels to {out_dir}")
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
        "--dataset", default=None,
        help=(
            "Dataset label appended as '__{dataset}' to each cell key "
            "by `python -m ddssm.launch init_centering` (e.g. 'mv', '1d'). "
            "Omit to walk every dataset."
        ),
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
    """CLI entry point dispatching to the aggregate / plot / all subcommands.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code from the selected subcommand.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "plot" and args.in_dir is None and args.records is None:
        parser.error("plot: one of --in or --records is required")

    cmd = {
        "aggregate": _cmd_aggregate,
        "plot": _cmd_plot,
        "all": _cmd_all,
    }[args.cmd]
    return cmd(args)


if __name__ == "__main__":
    sys.exit(main())
