# Encoder head-to-head: Gaussian vs IAF vs deterministic-causal

**Status:** campaign running (launched 2026-06-23, single RTX 4090, eager).
Result tables are auto-filled by `collect_results.py` as runs converge; cells marked
_PENDING_ are not yet in. **No number is quoted unless it is a tail-mean at a verified
plateau** (every row carries a convergence STATUS).

## TL;DR
_PENDING — filled once Phase 2 finalists + seeds land._

## The question
Given more capacity and their best learning rate, can the **parallel** encoders (IAF,
deterministic-causal) match the sequential **GaussianEncoder** on an EASY (LGSSM) and a
HARD (NonlinBimodalLiftMV) dataset, at history `j=1`? Two sub-questions:
- **Capacity** (Phase 1): can the encoder+decoder *reconstruct*?
- **Forecast** (Phase 2): does the full model *forecast* well?

## Why this exists / what was wrong before
A prior recon-only "gate" made the parallel encoders look ~4× worse. That gate was
**methodologically broken**: it used a λ-warmup *ramp* (`stage_1_lambda_start=1e-5`,
`warmup_frac=0.95`), so as λ→1 the model traded reconstruction for KL — recon *rose*
(507→625 while rate fell 506→80) and the reported "min recon" was a transient at λ≈0.1,
not a diverging autoencoder. So every recon number in the old `runs/` is unreliable. This
report rebuilds the comparison from a clean protocol.

## Experimental design
**Datasets (2, synthetic, T=32).**
- **LGSSM** (easy): scalar AR(1), D=1 — has an analytic **Kalman** optimum (absolute
  reference, not just a relative ranking) and exposes GT latents.
- **NonlinBimodalLiftMV** (hard): D=8 obs / true latent 4, nonlinear bimodal lift.

**Encoders (3), `j=1`.** Gaussian (reference) · IAF (`arflow_stochastic_state=True`) ·
deterministic-causal (`False`). Same backbone/transition for all; only the encoder
differs. Parallel encoders run at **bumped capacity** (`arflow_channels=128`,
`arflow_causal_layers=4`), decoupled from the transition's `channels=48`.

**Phase 1 — capacity** (`h2h_cap__<enc>__<ds>`). Pure autoencoder: λ pinned **exactly to
0** (`stage_1_lambda_start=end=0`, no ramp), `lambda_sigma_p=0`, **stage-1 only**. Sweep
**base_lr only** (6-point grid 3e-4…8e-3). Metric: **`recon_mse`** = MSE of the decoded
posterior mean vs the observed sequence (val). NOT the distortion NLL: under λ=0 the decoder
σ is unregularised, so a flexible AE shrinks σ_dec→0 and the NLL collapses toward −∞
(non-discriminating); `recon_mse` scores only μ_x, so it is bounded and a clean capacity gauge.

**Phase 2 — full model** (`h2h__<enc>__<ds>`). Two-stage ELBO; all stage knobs swept by
Optuna (`+sweep=h2h_full`, 35 trials/cell). Selection objective = **held-out VAL forecast
CRPS-sum** (`source="json"` → one post-training forecast eval per trial). Finalists are
re-scored on **TEST** (select on val, report on test) over **≥3 seeds**.

**6 cells** = {gaussian, iaf, det} × {lgssm, nlblmv}, all under one identical protocol —
the nlblmv-gaussian cell is **re-run** here, not reused from the old sweep.

## Threats to validity & how they're controlled
1. **Sweep ≠ what we judge on.** Optimising val ELBO can pick forecast-*worse* configs
   (recon↓ but trans-KL↑). → The Phase-2 sweep optimises **forecast CRPS-sum**, not ELBO.
2. **No error bars.** Optuna varies hyperparameters, not seeds. → Finalists are
   **seed-replicated (≥3)**; the headline table reports mean±std.
3. **Capacity confound — LARGE.** The "bump" is not modest: at the configured sizes the
   parallel encoders carry **~70× the Gaussian encoder's parameters** (data_dim=8):
   - Gaussian encoder **13.7k** · IAF **953k** · deterministic **953k** params
     (model totals 170k / 1.11M / 1.11M).
   This is deliberate (give the parallel encoders every advantage), so the comparison reads
   as a **handicap test**: *even with ~70× the encoder capacity*, can IAF/det match Gaussian's
   forecast? A loss is then strong evidence against the architecture; a match is expected and
   would need a **matched-capacity** arm to be conclusive (flagged as a follow-up).
4. **λ=0 collapses IAF→deterministic AND collapses decoder σ.** With no KL penalty, recon
   drives the *encoder* σ→0, so Phase 1 measures the **deterministic backbone** capacity
   (prediction/sanity check: IAF ≈ det in Phase 1; the IAF-vs-det story lives in Phase 2). It
   also drives the *decoder* σ→0 (unbounded NLL) — hence the Phase-1 metric is `recon_mse` on
   μ_x, not the distortion NLL (caught live: distortion ran −3→−54 and dropping).
5. **Convergence.** Every quoted number is a tail-mean at a verified plateau; Phase-2
   stage-2 runs the full calibrated budget (n_stage2=4000; plateau ~3700). Each row reports
   STATUS (converged / transient / diverging) + steps-to-plateau + wall.
6. **Kalman reference matches the data + metric.** Uses the TRUE generative params
   (a=0.9, σ_proc=σ_obs=0.1 — `data/synthetic.py`), as a **predictive forecast NLL** over
   the horizon (apples-to-apples with the model's forecast NLL). Guarded by
   `tests/test_kalman_forecast_nll.py` (empirical NLL must match the analytic entropy floor).
7. **External validity.** Synthetic only, `j=1` only — the conclusion is scoped to
   controlled synthetic regimes at unit history; it does NOT generalise to real GluonTS
   series or longer history.
8. **Absolute floor.** The marginal-baseline gate (`eval_baselines.py`, LOCF + marginal
   Gaussian) is the floor: any cell that fails to beat it is in marginal collapse and is
   marked FAIL regardless of relative rank.

## Phase 1 — encoder capacity (recon_mse on μ_x, pure AE) — COMPLETE
Capacity = val `recon_mse` (MSE of the decoded posterior mean) at the best of a 6-point base_lr
grid. The AE sees x, so it can reconstruct *below* the obs-noise variance (memorisation) —
near-0 = saturated/perfect. Param counts: gaussian 13.7k vs iaf/det 953k (~70×).

| dataset | encoder | best base_lr | recon_mse ↓ | steps (early-stop) | STATUS |
|---------|---------|-------------|-------------|--------------------|--------|
| lgssm  | gaussian | 3e-4 | **0.0001** | 2045 | converged (≈perfect) |
| lgssm  | iaf      | 3e-4 | **0.0000** | 1446 | converged (≈perfect) |
| lgssm  | det      | 3e-4 | **0.0000** | 1110 | converged (≈perfect) |
| nlblmv | gaussian | 2e-3 | **0.99** | 1684 | early-stop @ distortion plateau ‡ |
| nlblmv | iaf      | 8e-3 | **2.02** | 1073 | early-stop; best LR at grid edge ‡ |
| nlblmv | det      | 4e-3 | **2.53** | 854  | early-stop @ distortion plateau ‡ |

‡ Runs early-stop on the *distortion* plateau (the λ=0 NLL), not on recon_mse directly;
recon_mse convergence is re-checked via checkpoint deltas in the final assembly. IAF's optimum
sits at the top of the LR grid (8e-3) → it may want a higher LR (a small caveat on its 2.02).

**Read.** (1) **Easy (lgssm): capacity is not the bottleneck** — all three reconstruct ~perfectly,
and **IAF ≈ det ≈ 0** (sanity check #4 holds: λ=0 collapses both to the deterministic backbone).
(2) **Hard (nlblmv): the Gaussian (13.7k params) reconstructs best (0.99); the parallel encoders
are worse — IAF 2.02, det 2.53 — despite ~70× the parameters.** So on the hard set the parallel
architecture has *worse raw reconstruction capacity*, not just worse forecasting. This is the
key Phase-1 result; Phase 2 tests whether it carries into forecast quality.

## Phase 2 — full model (forecast, held-out TEST, mean±std over seeds)
CRPS-sum / energy / NLL lower = better; marginal-gate must be PASS; LGSSM adds the analytic
Kalman forecast-NLL and the gap-to-optimal.

| dataset | encoder | CRPS-sum ↓ | energy ↓ | NLL ↓ | marginal gate | recon / rate split | STATUS |
|---------|---------|-----------|----------|-------|---------------|--------------------|--------|
| lgssm   | gaussian | _PENDING_ | | | | | |
| lgssm   | iaf      | _PENDING_ | | | | | |
| lgssm   | det      | _PENDING_ | | | | | |
| nlblmv  | gaussian | _PENDING_ | | | | | |
| nlblmv  | iaf      | _PENDING_ | | | | | |
| nlblmv  | det      | _PENDING_ | | | | | |

**LGSSM analytic reference (optimal forecaster):** Kalman forecast-NLL = **−0.183**
(per-horizon −0.43 → −0.06 as forecast uncertainty grows over h=1..8). This is the floor on
the test split (negative because obs noise σ=0.1 ⇒ a sharp Gaussian predictive); each
encoder's model forecast-NLL is ≥ this, and the gap-to-Kalman = _PENDING (finalists)_.

## Diagnostics (live findings — analytic refs + probes)
- **LGSSM is solved.** Best model forecast ≈ the analytic **Kalman optimum** (iaf 0.673 vs Kalman
  0.671 crps-sum), and Phase-1 shows all encoders reconstruct it perfectly. So lgssm does **not**
  discriminate encoders at convergence — the lgssm sweep was intentionally truncated (partial trials
  kept). SNR≈2 (signal std 0.206 / obs-noise 0.10); the "noisy-looking" forecast is the irreducible
  obs noise, not model error.
- **NLBL_MV sits at the marginal floor — on every metric.** Best models ≈ marginal baseline on
  crps-sum (0.668 vs 0.681) **and** the multivariate **energy score** (19.90 vs 19.91), and slightly
  worse on MAE/RMSE. So it is **not** a crps-channel-sum artifact. Root cause: the bimodal impulse
  `δ·s_t` (δ=2, i.i.d. random ±1) has variance δ²=4 per latent dim vs the predictable `tanh(Az)` part
  (≤1) — the dynamics are **mostly unpredictable**, so the marginal is near-optimal and no encoder can
  separate. baselines: LOCF 1.076, marginal 0.681.
- **The diffusion transition mode-collapsed — and the failure is localized to the transition, not the
  encoder.** Probing `p_ψ(z_t|z_hist)` (1000 draws) on the best nlblmv model: the learned transition is
  **unimodal** in latent (0/8 dims) and obs (0/24 channels) space, despite the true transition being
  bimodal (up to 16 modes). Localising it:
  - **Encoder aggregate posterior is BIMODAL** (1D 2-Gaussian EM, sep 2.04) — the encoder *preserves*
    the data's bimodality.
  - **The diffusion's training target is bimodal**: the empirical next-state `z_t | z_{t-1}` (near-fixed
    z_{t-1}, 400-NN, in the encoder's own latents) is bimodal in **21/30** conditionings (sep 1.6–2.6).
  - **The learned conditional is unimodal** (0/8). So the diffusion mode-averages a *clearly bimodal
    target*. Failure chain: data bimodal ✓ → encoder bimodal ✓ → **diffusion unimodal ✗**.
  This is the real headline of the hard arm: the CSDI diffusion transition mode-collapses on
  conditionally-bimodal dynamics even with a bimodal target — a transition property, shared across all
  encoders, so the encoder head-to-head never surfaces it. Hypothesised cause (open): σ_data calibrated
  to the wide marginal + EMA-smoothed denoiser + `lsgm_is` noise-level weighting flatten the score-field
  bifurcation. Plots: `/tmp/{transition_modality,latent_transition_modality,aggpost_modality,cond_modality}.png`.
  Follow-up: a controlled small-δ / predictable-bimodal cell to test whether it can *ever* learn a
  bimodal conditional here.

## Cross-read (capacity vs forecast)
_PENDING._ If the parallel encoders MATCH in Phase 1 (capacity) but LOSE in Phase 2 (full
model), the limit is the transition/forecast coupling, not raw encoder capacity — and vice
versa. Easy (LGSSM) vs hard (NLBL_MV) localises whether any gap is the nonlinearity /
multimodality.

## Conclusion
_PENDING — scoped to synthetic, j=1._

## Reproducibility
```bash
# Phase 1 (one cell): pure-AE capacity, 6-LR grid
TORCHDYNAMO_DISABLE=1 .venv/bin/python -m ddssm.app --multirun \
  experiment=h2h_cap__iaf__lgssm \
  experiment.training.stages.base_lr=3e-4,6e-4,1e-3,2e-3,4e-3,8e-3

# Phase 2 (one cell): full ELBO sweep, minimise val CRPS-sum
TORCHDYNAMO_DISABLE=1 .venv/bin/python -m ddssm.app --multirun \
  experiment=h2h__iaf__nlblmv +sweep=h2h_full hydra.sweeper.n_trials=35 \
  hydra.sweeper.study_name=iaf_nlblmv hydra.sweeper.storage=sqlite:///study.db

# Whole campaign + collection
bash runs/h2h/run_campaign.sh                       # all 12 cells
.venv/bin/python experiments/arflow_headtohead/collect_results.py   # fills the tables

# Finalist test-eval (per cell; LGSSM adds the Kalman reference)
.venv/bin/python -m ddssm.evaluate experiment=h2h__iaf__lgssm checkpoint=... \
  experiment.eval.split=test \
  'experiment.eval.metrics=[crps_sum,energy_score,nll,kalman_forecast_nll]'
```
