# Variance probe experiments

Run the 4-dataset variance probe multirun:

```bash
bash scripts/experiments/variance/run_all.sh
```

This invokes `python -m ddssm.variance` over:

- `variance_probe_lgssm`
- `variance_probe_bimodal_clean`
- `variance_probe_bimodal_noisy`
- `variance_probe_nonlinear_bimodal_lift`

and writes an aggregated markdown report.
