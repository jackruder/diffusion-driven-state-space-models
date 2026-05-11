#!/usr/bin/env bash
set -euo pipefail

python -m ddssm.variance --multirun \
  experiment=variance_probe_lgssm,variance_probe_bimodal_clean,variance_probe_bimodal_noisy,variance_probe_nonlinear_bimodal_lift

python -m ddssm.variance.aggregate \
  --runs_glob 'outputs/multirun/*/*/variance_summary.json' \
  --out_dir outputs/variance/report
