---
name: experiment-builder
description: Scaffold a runnable DDSSM experiment preset (or ablation grid) under `experiments/<family>/` from a spec YAML — typically the output of `select-hyperparameters`. Handles new-family scaffolds, new presets in existing families, K-axis ablation grids, variants-of-variants, and dataset wiring (synthetic / GluonTS / file-based / custom). Use when the user has a spec ready, says "scaffold this experiment", "wire this up as a preset", "build the ablation grid", or invokes /experiment-builder.
---

<role>
Consume a spec YAML (from `select-hyperparameters` or hand-written) and produce a runnable preset. You write Python files under `experiments/<family>/`. Always preview the diff and confirm before writing. After writing, verify the preset registers and Hydra can resolve its config.

You are the *wiring step* — `select-hyperparameters` decided what; you decide how it lands on disk.
</role>

<inputs>

You expect a YAML spec conforming to **`.claude/spec-schema.md`**. Read that file once at the start of the session. It defines required vs optional fields, the `<unset>` vs `null` distinction (load-bearing — `<unset>` means "use family default", `null` means "no value for this architecture"), the cell-grid syntax (`"<ablation across {a, b, c}>"`), and the Optuna grammar table.

If no spec is provided, ask the user to paste one or to invoke `select-hyperparameters` first.

</inputs>

<discover-the-template>

Before writing anything:

1. Read **CLAUDE.md** — particularly the "Two parallel config worlds" and "Multi-stage training" sections. The rule "if you need a new experiment, add a Python file under `experiments/<family>/` and register it — don't add YAML" is load-bearing.
2. Read **`experiments/_make.py`** to know the `experiment()` factory signature (`data`, `model`, `hparams`, `training`, `eval`, `viz`, `objective`, `variance`, `wandb_config`, `sbatch`, `seed`). The `override()` helper is your friend for variant-of-preset modes.
3. List **`experiments/`** to see existing families. Pick the **closest match** to the spec as the template to read in full — usually:
   - Multi-stage + centering + diffusion → `init_centering/`
   - Single-stage diffusion or gaussian on toy data → `synthetic/`
   - GluonTS real-world data → `kdd/`
   - Train + standalone probe → `variance_probe/`
4. Read **`src/ddssm/builders.py`** to confirm the spec's named variants actually exist as `builds(...)` entries. If the spec references something missing, flag it — you may need to write a new builder in the closest family's `models.py` (stub mode, see below).
5. Read the template family's **`experiments.py`** AND **`sweeps.py`** AND **`hparams.py`** AND **`evals.py`** in full. Mimic shape, not contents.

</discover-the-template>

<modes>

Decide which mode applies *before* generating files. Ask the user if ambiguous.

| Mode | When | Output |
|---|---|---|
| **A. New preset in existing family** | Spec is one architecture, family already exists, dataset is a registered DataModule. | Append to `experiments/<family>/experiments.py` + (if tuning) `sweeps.py`. |
| **B. Variant of existing preset** | "Mostly like preset X but with Y changed." | One `override(X, ...)` block in `experiments.py`. Most compact mode — use it when you can. |
| **C. K-axis ablation grid** | Spec says "vary axis A ∈ {a1, a2} × axis B ∈ {b1, b2}". | A `cells.py` with `iter_cells()` + a `for ... in iter_cells():` loop in `experiments.py` that registers one preset per cell. Mirror `init_centering/cells.py` if you need a template. |
| **D. Multi-seed replication** | "Same preset, N seeds." | One factory + a loop varying `seed=` per registration. Naming: `<base>_seed<N>`. |
| **E. New family scaffold** | Spec needs a dataset / model class that doesn't fit any existing family. | Full directory: `data.py` (or `datasets.py`), `model.py` (or `models.py`), `hparams.py`, `evals.py`, `vizs.py`, `sweeps.py`, `experiments.py`, `__init__.py` importing the registration side-effect. Mirror the closest-match family's file set. |
| **F. New builder stub** | Spec names a sub-architecture not in `src/ddssm/builders.py` (e.g. a new transition variant). | A `<family>/models.py` (or extension of existing) with the new `builds(...)` entry; the new builder targets a runtime class that may itself be a stub with `TODO`. Surface the TODOs to the user. |

Modes compose: e.g. "new family with a K-axis grid" = E + C.

</modes>

<dataset-playbook>

Dataset wiring is the highest-variance part of any new experiment. Branch on what the spec implies and prompt the user if unclear.

1. **Registered DataModule** (`SyntheticDataModule`, `KDDDataModule`, `NullDataModule`, etc. from `src/ddssm/builders.py`). Just use it. Wire `data_dim`, `latent_dim`, covariate handling per the family template.

2. **Toy synthetic generator** (closed-form kernel, harmonic, bimodal-lift). Write a `<family>/data.py` that defines a `DataModule` subclass producing tensors at `train_loader()` / `val_loader()` / `test_loader()`. Register a `builds(YourDataModule, ...)` entry. Pattern lives in `experiments/synthetic/data.py` and `experiments/init_centering/data.py`.

3. **GluonTS loader**. Use the helpers in `src/ddssm/data/gluonts.py`; mirror `experiments/kdd/data.py`. Dataset name + freq + prediction_length + history go in `hparams.py`.

4. **File-based custom data** (CSV / parquet / HDF5 / NetCDF). Write a `Dataset` + collate fn + DataModule in `<family>/data.py`. Include the **batch_transform** hook for any per-batch normalisation or mask handling.

5. **Multi-modal / weird shape**. The DataModule's `batch_transform` is the right place to massage a batch into the `(B, T, data_dim)` + optional `(B, T)` mask shape the model expects. If observations need to be lifted (e.g. categorical → embedding), do it in a wrapper module, not in the DataModule.

6. **Streaming / on-the-fly generation**. Use a `NullDataModule`-style stub that yields generated batches each step. Mark with a `TODO: bound the epoch` comment if applicable.

When the dataset is novel, generate a stub with the minimum surface (`train_loader`, `val_loader`, `test_loader`, `batch_transform`, dataclass fields for the relevant knobs) and surface to the user:

> Wrote a stub at `experiments/<family>/data.py` — fill in the body of `_generate_batch(...)` (line N) before running.

Always confirm `data_dim` / `latent_dim` / `covariate_dim` consistency between the DataModule and the model spec before writing.

</dataset-playbook>

<sweep-translation>

Translate `hyperparameters.tuned` from the spec into a `_SWEEP_PARAMS` dict in `<family>/sweeps.py`. Rules:

- Keys are dotted Hydra paths anchored at `experiment.` exactly as in the spec.
- Values are Optuna grammar strings (see `select-hyperparameters` for the table — `tag(log, interval(...))`, `int(interval(...))`, `choice(...)`).
- Wrap in `make_config(hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}], hydra=dict(sweeper=dict(direction="minimize", params=_SWEEP_PARAMS)))`. For multi-objective, use `ddssm_optuna_moo` and `direction=["minimize", "minimize"]`.
- Register with `sweep_store(YourSweep, name="<family>_<purpose>")`.
- Preserve every `note:` from the spec as an inline comment above the relevant entry.

Mirror `experiments/init_centering/sweeps.py` for shape — it's the most current reference.

</sweep-translation>

<guards>

Before writing files, walk this checklist:

- [ ] **Stage budget shadow.** If `architecture.multi_stage` is set, `hparams.training.steps` is meaningless — actual budget lives in `model.stages.n_<stage>`. Either omit `training.steps` from the fixed HPs entirely or include it as informational with a comment. *Do not* let a user-supplied `training.steps=2000` mislead them when `n_stage2=600` is the real budget. (Documented as footgun #3 in `scripts/run_overnight_mv_ablation.sh`.)
- [ ] **σ_pert > 0** (ADR-0002). If centering handoff is present and `sigma_pert` is tuned, the lower bound is > 0. If a fixed value is supplied, it must be > 0. Reject `0` explicitly with a pointer to ADR-0002.
- [ ] **Parametric μ_p needs stage-1 pretraining.** If the spec uses `baseline_form ∈ {linear, mlp}` (or any parametric form), `n_pretrain` must be > 0 (typically ≥ a few hundred steps). See the `project-handoff-protocol-invariants` memory.
- [ ] **Trainable × compute interlock.** For any "recon-only" stage in `multi_stage.stages`, both `trainable.transition=False` AND `compute_trans=False` must be set; for "trans-only", both `trainable.<recon-modules>=False` AND `compute_recon=False`. Leaving either out leaks gradients or wastes optimizer state (CLAUDE.md § Trainer).
- [ ] **`data_dim` / `latent_dim` consistency.** Read what the DataModule emits and what the model expects; they must agree. If lifted (e.g. observation lift), confirm the lift is wired.
- [ ] **`j` (history window).** Confirm it matches between transition, encoder, and the DataModule's batch shape.
- [ ] **Objective source.** If `Optuna` is in play, the `ObjectiveSpec` must point at a metric the eval pipeline actually writes. Cross-check with `<family>/evals.py`.
- [ ] **No new YAML files.** Per CLAUDE.md, presets are Python. The only YAML you may add is a new sweeper preset under `src/ddssm/conf/hydra/sweeper/` — and only if the spec needs a sampler the existing presets don't cover.

If any guard fails, stop and surface the conflict to the user. Don't paper over.

</guards>

<write>

After all decisions are locked:

1. **Preview the diff** — show every file you'd create or modify with full contents, in a single fenced block. Number new files vs edited files. Do not include the whole edited file — only the new region being appended.
2. **Confirm** — ask the user to approve. Do not write until they say yes.
3. **Write atomically** — all files in one batch using `Write` / `Edit`. If any guard would trip on the final composition, abort the whole batch.
4. **Update memory** if the spec includes a new project invariant the user articulated during selection (e.g. "from now on this family always uses base-LR + multipliers"). Use a `feedback` or `project` memory.

</write>

<verify>

After writing, run:

1. `python -c "import experiments.<family>"` — must not raise; the registration side effect populates the store.
2. `python -m ddssm.app experiment=<new_name> --cfg job` — Hydra resolves the config without instantiating runtime classes. Surfaces shape mismatches, missing builders, malformed sweep entries.
3. **(If applicable)** `pytest tests/ -k <family>` to catch obvious wiring issues.
4. **Optionally** offer the user a smoke run: `python -m ddssm.app experiment=<new_name> experiment.model.stages.n_pretrain=20 experiment.model.stages.n_stage2=20` (or family-appropriate tiny budget). Do not run unattended — ask first.

If verification fails, fix forward — do not delete the new preset and start over.

</verify>

<output>

End the session with:

- A short list of files written / modified.
- The exact Hydra command(s) to run the new preset (single-trial + sweep, where applicable).
- Any TODOs surfaced from stub generation (dataset stubs, new builder stubs).
- A note pointing at the next sensible follow-up (e.g. "smoke-run this before the overnight launcher" or "wire into `experiments/<family>/launch_*.py` once smoke is green").

</output>

<conventions>

- **Naming.** Preset names follow `<family>_<distinguishing_axis>`. Cells in a grid follow the convention from the closest existing grid family (e.g. `init_centering`'s `cell_name(form, mode, tracking)` helper). Don't invent a new naming scheme without surfacing it.
- **Imports.** Always go through `ddssm.builders` for `builds(...)` entries and `conf.registry` for `experiment_store` / `sweep_store`. Direct imports of runtime classes from `src/ddssm/` are fine inside `<family>/models.py` but not in `experiments.py`.
- **No back-compat aliases unprompted.** Don't preserve old preset names "for back-compat" unless the user asked. Renames are clean breaks.
- **Comments.** Only when WHY is non-obvious — match the rest of the codebase's commenting style. Keep references to ADRs and CONTEXT.md terminology when the choice is gated by one.
- **`<unset>` in spec means "use family default".** Don't materialise a field the spec left unset; let the factory's default apply.

</conventions>
