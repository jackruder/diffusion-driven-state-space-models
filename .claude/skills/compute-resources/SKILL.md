---
name: compute-resources
description: Define and maintain named hardware profiles for this project (workstation, cluster partition, cloud VM, laptop CPU). Profiles persist at `.claude/resources.yaml` and are consumed by `compute-budget` and `plan-campaign` — describing hardware once instead of being re-prompted every session. Use when the user says "register my GPU", "save these hardware specs", "add a SLURM partition", "set my default device", or invokes /compute-resources.
---

<role>
Help the user serialize their available compute as named profiles. The output is a YAML file at `.claude/resources.yaml` (project-local; user-global at `~/.claude/resources.yaml` if the user prefers cross-project reuse). Other skills read this file to skip the resource-intake step.

You are a configuration step, not a benchmarking step. You capture *what's available*, not *how fast it is* — `compute-budget` measures throughput when it runs.
</role>

<inputs>

Either:

- The user describes their hardware in prose ("I have an RTX 4090 and access to a SLURM cluster with H100s") and you turn it into profiles, or
- The user wants to edit an existing profile (rename, change defaults, retire a stale entry), or
- The user has no idea what's available — offer to run `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv`, `lscpu | head -20`, `nproc`, `free -h`, and `sinfo` (if SLURM tools are on PATH) and infer from the output.

</inputs>

<schema>

The file at `.claude/resources.yaml` follows this shape. Read existing contents (if any) and merge — don't overwrite without confirming.

```yaml
profiles:
  <name>:                              # short kebab-case slug, e.g. "workstation", "slurm-h100"
    description: <str>                 # one-liner shown when this profile is active
    devices:                           # one or more device descriptors
      - kind: <gpu | cpu | tpu | other>
        name: <str>                    # e.g. "NVIDIA RTX 4090"
        count: <int>                   # number of devices of this kind in the pool
        count_max: <int>               # optional; for cluster pools where you can request up to N
        memory_gb: <int>               # per-device, not aggregate
        compute_capability: <str>      # optional; gpu only
        notes: [<str>]
    availability: <enum>
      # exclusive          — you own it; no other tenants
      # shared             — other tenants on the same device; leave headroom
      # preemptible        — can be killed; checkpoint often
      # slurm-queued       — request via sbatch, queue time on top of wallclock
    slurm:                             # optional; only if availability == slurm-queued
      partition: <str>
      qos: <str>
      timeout_min_default: <int>
      cpus_per_task: <int>
      mem_gb: <int>
      preempt_window_s: <int or null>  # null if not preemptible
      max_jobs_concurrent: <int>       # account/qos limit
      array_limit: <int>               # max array indices per submission
    storage:
      runs_dir: <path>                 # where the project writes outputs on this profile
      scratch: <path>                  # fast local scratch if any
      optuna_db_dir: <path>            # where SQLite Optuna stores live
      shared_fs: <bool>                # true if multiple compute nodes see the same FS
                                       # (matters for Optuna DB visibility across workers)
    cost:                              # optional
      currency: <str>                  # e.g. "USD"
      per_device_hour: <float>         # cloud / metered usage; null for owned hardware
    notes: [<str>]

defaults:
  active_profile: <name>               # profile that compute-budget loads when no flag is given
  fallback_profile: <name>             # used when the active profile is unreachable
```

Required fields per profile: `devices` (with at least one entry), `availability`. Everything else is optional.

</schema>

<flow>

## Phase 1 — Inventory

Ask the user what they want to register. Three common cases:

1. **One machine, simple.** "RTX 4090, 24 GB, I run things here directly." → one profile, exclusive availability, no SLURM block.
2. **Workstation + cluster.** Two profiles — local workstation + named SLURM partition. Ask which is the default (`defaults.active_profile`).
3. **Heterogeneous cluster.** One profile per partition / device pool. The campaign planner will allocate across them; `compute-budget` for a single experiment will pick one.

If the user is unsure, offer to inspect. Run, with confirmation:

- `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader`
- `lscpu | grep -E 'Model name|^CPU\(s\)|Architecture'`
- `free -h | head -3`
- `sinfo --format='%P %D %G %m' --noheader` (skip silently if `sinfo` isn't installed)
- `df -h <repo>` to confirm `runs_dir` has space

Use the output to fill in `memory_gb`, `compute_capability`, partition names. Show the inferred profile back before writing.

## Phase 2 — Disambiguate

Ask about anything the inspection can't determine:

- **Availability mode.** Is this device exclusive, shared, preemptible, or queued?
- **Headroom expectations.** On shared devices, what fraction is safe to consume? (Affects `workers_per_device` math in `compute-budget`.)
- **Default partition / QOS** for SLURM profiles.
- **Storage paths.** Confirm where outputs land — local `runs/` for workstation, `/scratch/$USER/...` for cluster nodes.
- **Cost**, only if the user volunteers it (cloud users; cluster users with GPU-hour quotas).

## Phase 3 — Write the file

Merge with whatever exists at `.claude/resources.yaml`. Show the diff before writing. After writing:

- Set or confirm `defaults.active_profile`. If the user is registering their first profile, that becomes the default.
- Print the one-line `description` of the active profile so the user can verify it's the right one.

## Phase 4 — Validate

Sanity checks:

- [ ] **At least one profile exists.** A `compute-budget` invocation with no profiles falls back to prompting; warn the user this is fine but defeats the point of serializing.
- [ ] **`active_profile` resolves to a real entry.**
- [ ] **SLURM profiles have a partition set.**
- [ ] **Preemptible profiles have `preempt_window_s` set** (the campaign planner uses it to size checkpoint cadence).
- [ ] **Shared profiles state a headroom assumption** in `notes`, even informally — "leave ≥4 GB for other tenants" or similar.
- [ ] **`runs_dir` is writable on the target machine.** If the profile is the current local machine, check directly (`test -w <path>`); otherwise note as unchecked.

</flow>

<conventions>

- **Project-local takes precedence over user-global.** `compute-budget` reads `.claude/resources.yaml` first, then falls back to `~/.claude/resources.yaml`. State which file you're editing.
- **Slugs are short and kebab-case.** `workstation`, `slurm-h100`, `colab-t4`, `laptop-cpu`. Avoid spaces or version suffixes; the profile is a label, not a build ID.
- **`memory_gb` is per-device.** A node with 4× A100 80 GB has `count: 4, memory_gb: 80`, not `memory_gb: 320`.
- **Don't measure throughput here.** That's `compute-budget`'s job. This skill captures static facts.
- **Gitignore is the user's call.** `.claude/resources.yaml` may contain machine-specific paths or quotas. Mention `.gitignore` once when first creating the file; don't nag.
- **When retiring a profile**, comment it out rather than deleting — the user may want it back.

</conventions>

<output>

End with:

- The path you wrote to (`.claude/resources.yaml` or `~/.claude/resources.yaml`).
- A short table of profiles registered, with the active one marked.
- A one-line pointer: "Now `compute-budget` and `plan-campaign` will load `<active>` by default; pass `--profile=<other>` to override."

</output>
