# Split-loss training + p_k clip + DDP σ_d sync — TDD module & execution plan

*Revised after review (2026-07-01): line-number references replaced with symbol names; `LossComponents` field-rename shim extended to test-suite ctor sites; M1 `S_k` normalization pinned as symmetric across both accumulators; M2 sufficient-statistic shapes pinned in prose; M6 ψ-scheduler ownership resolved via `_install_scheduler` hook + `stages.py`/`handoff.py` routing; Migration adds Optuna-study caution; Verification adds `use_reentrant=False` grep-fence; wave gate softened to CI (per user preference for local iteration).*

## Context

`training.org` specifies a **one-forward, two-backward split-loss estimator** for the per-step diffusion KL. Encoder / baseline / decoder (`φ, θ`) get an ELBO-weighted `w_ll/p` scalar; the score net (`ψ`) gets a unit-weighted scalar; gradients are routed by two selective `.backward(inputs=[...])` calls sharing one forward.

The current code (`transitions/diffusion.py` at `3bed6bc`) computes only the single ELBO-weighted scalar and applies it to every parameter — ELBO-correct on `(φ,θ)`, forces `ψ` off EDM well-conditioning.

Two adjacent fixes ride along:
1. **`p_k` clip** — bound `w_ll/p` by clamping IS probs post-normalization. Existing floor `1e-12` doesn't cap the weight.
2. **DDP σ_d sync** — `SigmaDataBuffer._update_unchecked` does per-rank EMA with no cross-rank sync. Preventive (no DDP infra today) but must land before any multi-rank run.

**Outcome:** `FullELBO(use_split_loss=True)` flips to the split algorithm; `False` (default) preserves today's exact numerical behavior. `p_k_clip=1e-3` default; `None` disables. σ_d sync is no-op single-rank. Gradient clipping is **removed** repo-wide (it was `None` in every config already) and replaced by an always-on non-finite gradient skip with careful accounting (`optim/grad_norm` + cumulative `optim/grad_skips`, per-skip WARNING, counter persisted in checkpoints). Single-loss mode gains optional per-group ψ betas (`psi_betas=[0.9,0.99]`; default `None` = unchanged).

**Methodology (user-directed):** red-green-refactor per module. Each module's agent writes its failing tests FIRST, confirms red, implements to green, then refactors. Modules have explicit file ownership (disjoint within a wave) and criticality ratings that set the agent model (minimum: sonnet, medium effort).

**Navigation note:** every module below cites *symbols* — `_esm_chunk_loss`, `_update_unchecked`, etc. The earlier draft's line ranges had drifted by 30–100 lines relative to `HEAD`; agents should `grep -n` the symbol and read the current window.

## Locked design decisions

- **Two AdamW optimizers** under split mode: `opt_phith` (encoder + decoder + baseline + aux_posterior + static_embeddings + non-ψ transition params) and `opt_psi` (score-net family).
- **Always compute both scalars** in `_esm_chunk_loss` — the fork happens at `sqerr` before weighting. Single-mode path uses only `_phith`; `_psi` is a free `.sum()` on top.
- **Flag lives on `FullELBO`**: `use_split_loss: bool = False`. Trainer introspects `self._active_loss.use_split_loss` at `fit()` entry to decide optimizer topology.
- **No field rename in the read path** — keep `LossComponents.trans_kl` / `.init_kl` as `@property` aliases of the new `_phith` fields. Alias protects `eval/metrics.py`'s field access and ~10 test files that read the fields. **Property aliases cover reads only, not construction:** the M3 spec (below) explicitly extends the shim to test-suite ctor sites, since `@property` cannot be a ctor kwarg.
- **Two-timescale optimizer setup** (moving-target: ψ regresses onto `F*(φ)`, which moves as the encoder trains):
  - ψ's loss is **not** gated by rate-λ: `loss_psi = trans_kl_psi + init_kl_psi`. ψ trains at full strength through recon-only warmup so it already tracks the posterior when λ turns on. Valid because score matching is invariant to positive rescaling; also avoids time-varying-λ interaction with Adam's β₂ EMA. λ ramp stays on the φθ side (its actual job: protecting φθ from KL through an imperfect ψ).
  - `opt_psi` betas `(0.9, 0.99)` (short second-moment horizon to follow the moving target); `opt_phith` keeps `(0.9, 0.999)`.
  - Per-component LRs preserved: `opt_phith` keeps `enc_lr`/`dec_lr`/`baseline_lr`/aux(enc_lr)/static-emb(enc_lr) groups; `opt_psi` uses `trans_lr`.
  - **LR schedules: same warmup-cosine shape on both sides (locked)** — target motion ∝ φθ's effective LR, so a shared multiplicative shape keeps the timescale-separation ratio constant. Both schedulers restart **together** at stage boundaries. β₂ scheduling: fixed; mechanism documented (per-step `opt_psi.param_groups[i]["betas"]` mutation — Adam reads betas per step; only mutate after ~100 steps due to `beta**step` bias correction) but not implemented.
  - **Per-group ψ betas in single-loss mode (algorithm 1)**: AdamW reads `betas` per param group (constructor default applies where absent). The single-mode builder optionally tags the score-net groups (diffmodel + embed_layer, decay and no-decay) with `betas=(0.9, 0.99)` inside the one optimizer — the two-timescale benefit without two optimizers. Default **off** (`psi_betas=None`) so existing configs stay numerically identical. Survives `_rebuild_optimizer` (re-invokes the builder) and checkpointing (`load_state_dict` restores per-group hyperparams). Caveat (documented): in algo 1, ψ's only gradient is λ·KL, so ψ is frozen during recon-only warmup regardless of betas — fast β₂ helps only after λ turns on.
- **Gradient clipping removed everywhere; gradient skipping (NVAE) in its place**: clipping is already off in practice — `clip_grad_norm` defaults to `None` in `_default_hyperparams` and `DDSSMHyperParamsConf` (both in `dssd.py`), and **no config anywhere overrides it** — so "off everywhere" is enforced by deleting the mechanism: the hparam field, `self.clip_grad_norm` (`train.py` `__init__`), and the clip branch in `_optimizer_step`. In its place, an always-on non-finite grad guard: today a NaN/inf gradient steps unconditionally into the weights and poisons Adam's β₂ EMA (~1/(1−β₂) ≈ 1000 steps). New `_optimizer_step` behavior: compute the global grad norm via `clip_grad_norm_(model.parameters(), float("inf"))` (norm computation only — `inf` max means no rescale ever happens); if non-finite → zero all optimizers' grads (`set_to_none=True`), **skip step + scheduler + EMA**, increment a cumulative skip counter, `log.warning` with step and norm, continue training. **Skip accounting (careful, per user)**: `optim/grad_norm` logged every optimizer step, `optim/grad_skips` cumulative counter logged every step, per-skip WARNING line, and the counter persisted in checkpoint v3 so preempt-resume doesn't zero it. The host-side `isfinite` read syncs once per optimizer step — same cadence as the existing accum-boundary `float(accum_loss_t)` sync, no new per-microstep sync. Under split mode: unscale both optimizers (AMP), one norm over `model.parameters()`, skip gates both. **Non-finite only, no magnitude threshold** — with clipping gone, a large-but-finite spike now enters unclipped (documented trade-off below); the logged norm trace is the evidence base for adding a threshold later if needed. The existing opt-in `abort_on_nonfinite_loss` loss-level guard is unchanged and complementary: it fail-fasts at the loss; the grad guard gracefully skips at the gradient (and catches the NaN-loss case even when the abort flag is off, since NaN loss ⇒ NaN grads).
- **`init_kl` splits too**: the init walk's `loss_init` routes through `_esm_chunk_loss` (via the polymorphic `_score_init_step` hook), so it carries a ψ side. `transition_kl_init`'s decomposition is `loss = entropy + vhp`, `vhp = loss_init + kl_aux` — only `loss_init` splits; `entropy`/`kl_aux` are pure φθ.
- **DDP σ_d fix syncs sufficient statistics**, not the estimator: Bessel-corrected `var(unbiased=True)` inside `_estimator_per_t` is nonlinear in the batch; `all_reduce(AVG)` on the estimator biases σ_d² downward. Requires the per-dim **first moment** (see M2).
- **DDP + split backward**: two selective `.backward(inputs=[...])` fire DDP hooks twice on shared-graph nodes → `find_unused_parameters=True` when wrapping (documented only; no DDP infra in tree).
- **Grad-checkpoint compatibility depends on `use_reentrant=False`** in the existing `torch.utils.checkpoint(...)` call inside `_esm_chunk_loss` (already the case at `HEAD`). The split backward is only safe because this flag is set — a grep-fence in M8 makes that requirement structural.

## Parameter split (φθ vs ψ) — resolved

**ψ (into `opt_psi`):**
- `model.transition.diffmodel.*` (UNet score net)
- `model.transition.embed_layer` — **confirmed an attribute of the transition itself, NOT inside `diffmodel`**. It's the score net's input featurization; it receives gradient only through `diffmodel`'s forward → ψ family.

**φθ (into `opt_phith`):**
- `model.encoder.*`, `model.decoder.*`
- `model.baseline.*` (aliased at `model.transition.baseline` — dedup via the `_claimed_ids` pattern in `param_groups_for_adamw`)
- `model.aux_posterior.*`, `model.static_embeddings.*`
- Any other `model.transition.*` submodule not listed under ψ.

**Guard:** `split_params_phith_psi(model)` asserts every `requires_grad=True` param appears in exactly one list, and hard-errors on any transition submodule not explicitly assigned (so a future module can't land silently mis-routed).

---

## Modules

Each module lists: files owned (no agent touches files outside its module), spec, red tests, criticality → agent model.

### M1 — Diffusion core: split `_esm_chunk_loss` + `p_k` clip
**Criticality: CRITICAL → opus.** Core estimator; a silent numerics error here corrupts every run.
**Files:** `src/ddssm/model/transitions/diffusion.py`, `src/ddssm/model/transitions/transitions.py`, `src/ddssm/model/transitions/baseline_gaussian.py`, `tests/test_transitions/test_diffusion.py`.

Spec:
- `DiffusionScheduleConfig`: add `p_k_clip: float | None = 1e-3` after `time_chunk_size`.
- `_adaptive_is_density_meandom` & `_adaptive_is_density_full`: add `p_k_clip: float | None = None` kwarg, distinct from `floor` (raw-density zero guard). After the existing `raw / raw.sum(-1)`: `p = p.clamp_min(p_k_clip); p = p / p.sum(-1, keepdim=True)`.
- Thread `schedule.p_k_clip` into the density call sites in `transition_kl`.
- `_esm_chunk_loss`: two accumulators; replace the current weighted-sqerr accumulation with
  ```python
  sqerr = (F_pred - F_tgt_flat).pow(2).sum(dim=1)                 # unweighted
  total_sqerr_phith += (sqerr * weights).view(N, kc).sum(dim=1)
  total_sqerr_psi   +=  sqerr.view(N, kc).sum(dim=1)
  ```
  **Apply the existing `/ float(self.S_k)` normalization symmetrically to both accumulators** before returning (i.e. `per_sample_phith = total_sqerr_phith / float(self.S_k)`, same for `_psi`). Return `(per_sample_phith, per_sample_psi, mu_hat_t)` in the per-sample path and `(sum_phith, sum_psi, mu_hat_t)` otherwise. The `grad_checkpoint` branch is unchanged — both losses share `F_pred`; `use_reentrant=False` is already set (required for split backward).
- `transition_kl`: accumulate both sides per chunk; return `{"kl": kl_phith, "kl_phith": kl_phith, "kl_psi": kl_psi}` (+ `"kl_psi_per_sample"` under `return_per_sample`; existing `"L_p"`/`"L_p_per_sample"` wrap the phith side). Same `/(B·S)` denom on both.
- `_score_init_step` **hook contract change to 2-tuple** `(phith, psi)`. **All three implementers must be updated in this module** (grep `_score_init_step` in `transitions/` to enumerate):
  - `DiffusionTransition._score_init_step` (`diffusion.py`): returns both from `_esm_chunk_loss`.
  - Base `TransitionModule._score_init_step` (`transitions.py`): `return loss, loss.new_zeros(())`.
  - `GaussianTransition._score_init_step` (`transitions.py`): `return loss, loss.new_zeros(())`.
  - `BaselineGaussianTransition._score_init_step` (`baseline_gaussian.py`): `return loss, loss.new_zeros(())`.
- Base walk `transition_kl_init` (`transitions.py`): dual accumulators over the `range(j)` loop; returned dict keeps `"loss" = entropy + loss_init_phith + kl_aux` (unchanged composition on the phith side) and adds `"loss_psi" = loss_init_psi`.

Red tests: `test_esm_chunk_loss_returns_phith_and_psi` (differ under non-unit weights); `test_esm_chunk_loss_phith_reproduces_prior_single_loss` (golden bit-level under fixed RNG — capture golden value BEFORE refactor); `test_esm_chunk_loss_sk_division_applied_to_both` (unit-weight case: `sum_phith / S_k == sum_psi / S_k` — sanity that the normalization is symmetric); `test_transition_kl_returns_kl_phith_kl_psi_and_alias`; `test_transition_kl_init_returns_loss_psi`; `test_score_init_step_nondiffusion_returns_zero_psi` (parametrized across Gaussian and BaselineGaussian); `test_pk_clip_bounds_probability` (bins ≥ `p_k_clip/(1+K·p_k_clip)` — post-renormalization floor; rows sum to 1); `test_pk_clip_bounds_max_is_weight` (`max(w_ll/p) ≤ w_ll_max·(1+K·p_k_clip)/p_k_clip`); `test_pk_clip_none_recovers_prior_behavior` (bit-level); `test_pk_clip_threaded_through_full_transition_kl`.

### M2 — σ_d sufficient statistics + DDP sync
**Criticality: HIGH → sonnet** (fully-specified formulas; bit-identity test catches regressions).
**Files:** `src/ddssm/model/centering/sigma_data.py`, `tests/test_centering/test_sigma_data.py`.

Spec:
- Split `_estimator_per_t` into `_suff_stats_per_t` returning, per t:
  - `sum_mu` — first moment, **shape `(n, d)`** (required — Bessel variance is `(Σμ̂² − (Σμ̂)²/count)/(count−1)` per dim).
  - `sum_mu2_total` — **shape `(n,)`**, the scalar Σ over both block index K *and* feature dim d.
  - `sum_s2_total` — **shape `(n,)`**, the scalar Σ over both block index K *and* feature dim d (today's `s2_blocks.mean(dim=1).sum(dim=1)` becomes `sum(dim=1).sum(dim=1)` / (final `/count`) — this is the shape convention the pure estimator relies on).
  - `count` — **shape `(n,)`**, the summed block count K per t across ranks.
- Pure `_estimator_from_suff_stats`:
  ```python
  avg_post_var = sum_s2_total / count                        # (n,) — mean over K, sum over d, matches today
  mu_var = (sum_mu2_total - sum_mu.pow(2).sum(dim=1) / count) / (count - 1)
  mu_var = torch.where(count > 1, mu_var, torch.zeros_like(mu_var))   # post-reduce fallback
  return (avg_post_var + mu_var) / float(d)
  ```
- In `_update_unchecked`: compute suff stats; if `dist.is_available() and dist.is_initialized()`, `all_reduce` a flat concatenated payload with `op=SUM`, unpack; then estimator → existing EMA blend unchanged. `global_ema` `.mean()` needs no special handling. `n_updates` bumps identical across ranks (process-local counter; same batch cadence).
- The `per_t == 1 → mu_var = 0` fallback moves into `_estimator_from_suff_stats` on the **combined** count (two ranks × 1 sample = count 2 = real dispersion).

Red tests: `test_suff_stats_are_linear_in_batch` (halves sum to whole); `test_suff_stats_shapes_match_convention` (asserts the four tensor shapes above); `test_estimator_from_suff_stats_matches_original_estimator` (bit-identical to pre-refactor on same batch — capture golden BEFORE refactor); `test_all_reduce_called_on_suff_stats_when_dist_initialized` (monkeypatch both `dist.is_available`→True and `dist.is_initialized`→True, spy `all_reduce`, assert once/update with `op=SUM`); `test_no_reduce_when_dist_not_available_or_initialized` (parametrized); `test_ranks_agree_after_reduce_two_process_mock` (two buffers + shared summing mock converge to same `bar`); `test_per_t_one_fallback_uses_combined_count`. Repo convention: `monkeypatch`, not MagicMock.

### M3 — Loss objects: `LossComponents` fields, `SplitLoss`, `FullELBO` flag
**Criticality: HIGH → sonnet.**
**Files:** `src/ddssm/model/losses.py`, `src/ddssm/model/dssd.py` (compat shim on the `LossComponents(...)` ctor in `forward` only), `tests/test_losses.py`, `tests/test_loss_per_stage.py`, `tests/test_training/test_split_loss.py` (new; loss-object unit tests).

Spec:
- `LossComponents`: fields become `recon, init_kl_phith, init_kl_psi, trans_kl_phith, trans_kl_psi, r_sigma_p, r_mu_p`; `@property init_kl` / `trans_kl` alias the `_phith` fields; `elbo()` uses the phith fields (unchanged semantics); `elbo_reg()`/`total()` unchanged. Update the class docstring (currently at losses.py:23–28) to state the phith/psi split and the alias's read-only nature.
- **Property aliases cover reads only**: any consumer that does `components.init_kl = x` or constructs with `LossComponents(init_kl=...)` will break. Grep confirms three ctor sites use the legacy kwargs: `dssd.py` inside `forward`, `tests/test_losses.py` (L25/49/68/175), and `tests/test_loss_per_stage.py` (~L108). **All three are updated in M3** to pass the new `_phith` / `_psi` kwargs (test files pass `init_kl_psi=torch.zeros_like(init_kl_phith)` etc. to preserve numerical intent). No consumer sets fields post-construction — verified by grep.
- **Cross-wave compat shim on `dssd.py`**: the field change breaks `dssd.forward`'s keyword construction. M3 edits exactly the `LossComponents(...)` call in `forward` to `init_kl_phith=L_init, init_kl_psi=L_init.detach()*0, trans_kl_phith=L_trans, trans_kl_psi=L_trans.detach()*0` so the suite stays green after M3 alone; M5 replaces the zero placeholders with real values. No other `dssd.py` edits in M3.
- `SplitLoss` dataclass: `phith`, `psi` tensors; shims `.detach() -> SplitLoss`, `.item() -> float((phith+psi).item())`, `__float__ = item` (**required** — fit loop does `float(loss.detach())`), `.total` property, `__truediv__` (grad-accum scaling).
- `Loss.__call__` return type widened to `torch.Tensor | SplitLoss`.
- `FullELBO(use_split_loss: bool = False)`:
  - `False`: today's composition via the aliases — numerical parity.
  - `True`: `loss_phith = recon + lam*(init_kl_phith + trans_kl_phith) + reg`; `loss_psi = trans_kl_psi + init_kl_psi` (**no λ**); return `SplitLoss`.

Red tests: `test_split_loss_forward_shape` (`True`→`SplitLoss`, `False`→`Tensor`); `test_split_loss_dataclass_shim_methods` (`.detach()`, `.item()`, `float()`, `.total`, `/ accum`); `test_full_elbo_single_mode_numerical_parity` (bit-level vs pre-change composition); `test_psi_side_ignores_lambda` (`rate_lambda≡0` → `loss_psi` unchanged, `loss_phith` KL-free); `test_loss_components_alias_still_works` (`components.trans_kl is components.trans_kl_phith`); `test_existing_loss_ctor_sites_updated_to_new_kwargs` (grep-level fence — none of `test_losses.py` / `test_loss_per_stage.py` still uses `init_kl=`/`trans_kl=` as ctor kwargs).

### M4 — Parameter partition + param groups
**Criticality: HIGH → sonnet** (silent mis-assignment corrupts training; exhaustive-disjoint test is the fence).
**Files:** `src/ddssm/training/train_utils.py`, `tests/test_training/test_param_split.py` (new), extend `tests/test_trainer.py` (existing `param_groups_for_adamw` guard needs the new kwarg).

Spec:
- `split_params_phith_psi(model) -> (list, list)` per § Parameter split: ψ = `transition.diffmodel` + `transition.embed_layer`; φθ = everything else; `_claimed_ids`-style dedup for the shared baseline; assert exhaustive-and-disjoint over all `requires_grad` params; hard-error listing any unassigned transition submodule by name.
- `param_groups_phith(model, enc_lr, dec_lr, trans_lr, weight_decay, baseline_lr=None)` / `param_groups_psi(model, trans_lr, weight_decay)`: mirror `param_groups_for_adamw`'s `add_module` closure restricted to each side; preserve decay/no-decay logic (norm/bias/embedding/logvar_raw → no decay; note `embed_layer` is an `nn.Embedding` → lands in ψ's no-decay group).
- **Single-mode per-group ψ betas**: `param_groups_for_adamw` gains `psi_betas: tuple[float, float] | None = None`. When set, the ψ-side groups (diffmodel + embed_layer, decay and no-decay — same assignment as `split_params_phith_psi`) carry an extra `"betas": psi_betas` key; all other groups carry no betas key (inherit the optimizer's constructor default). AdamW resolves hyperparams per group, so this gives the two-timescale β₂ inside one optimizer. `None` (default) emits today's dicts byte-identical.

Red tests: `test_param_split_exhaustive_and_disjoint`; `test_embed_layer_assigned_to_psi`; `test_baseline_dedup_lands_phith_once`; `test_unknown_transition_submodule_raises` (attach a dummy module to transition, expect loud error); `test_param_groups_decay_split_preserved_per_side`; `test_psi_betas_tags_only_score_net_groups` (with `psi_betas=(0.9, 0.99)`: exactly the ψ groups carry the key; with `None`: no group has a betas key — bit-identical group dicts to pre-change); `test_trainer.py::test_param_groups_for_adamw_accepts_psi_betas_none` (backwards-compat: existing guard passes with the new kwarg unset).

### M5 — Model threading (`dssd.py`)
**Criticality: MEDIUM → sonnet (medium).**
**Files:** `src/ddssm/model/dssd.py`, extend `tests/test_training/test_split_loss.py`.
**Depends on:** M1, M3.

Spec:
- `forward`: `L_trans_psi = trans_terms.get("kl_psi", zeros)` (non-diffusion transitions return no ψ key → zero is correct); `L_init_psi = init_terms.get("loss_psi", zeros)`; replace M3's placeholder ctor kwargs on the `LossComponents(...)` call with the real values.
- `_compute_transition_kl` and `_init_kl_loss` are near-pass-throughs (the former dispatches on `stage_selector` before delegating) — verify the new keys propagate; no functional change beyond docstrings.
- Metrics: the `trans_subterms` auto-log loop already emits `loss/rate/trans/<key>` for every non-`kl` key in the returned dict → verify `kl_phith` / `kl_psi` appear; add `loss/rate/init/loss_psi` alongside the existing init sub-component surfacing.
- **Hyperparams (same file, owned here to avoid wave-2 contention with M6's train.py work)**: remove `clip_grad_norm` from `_default_hyperparams` and `DDSSMHyperParamsConf` — no config anywhere sets it, and M6 removes the trainer read sites, so deletion is safe. Add `psi_betas: list[float] | None = None` to both (list, not tuple — OmegaConf-friendly); the trainer converts to tuple at use.

Red tests: `test_forward_components_carry_real_psi_values` (diffusion transition: `trans_kl_psi != trans_kl_phith` under non-unit weights); `test_forward_zero_psi_for_nondiffusion_transition`; `test_metrics_include_kl_phith_and_kl_psi`; `test_metrics_include_loss_init_psi`; `test_hyperparams_have_psi_betas_and_no_clip_grad_norm`.

### M6 — Trainer: dual optimizers, split backward, schedulers, grad-skip
**Criticality: CRITICAL → opus.** Gradient routing, `retain_graph`, AMP, accumulation — the hardest-to-debug failure modes live here.
**Files:** `src/ddssm/training/train.py`, `src/ddssm/training/stages.py`, `src/ddssm/model/centering/handoff.py`, extend `tests/test_training/test_split_loss.py`.
**Depends on:** M3, M4.

Spec:
- Trainer construction (`__init__`) + `fit()` entry (after `_active_loss` install): if `isinstance(self._active_loss, FullELBO) and use_split_loss` → build `opt_phith` (`param_groups_phith`, betas `(0.9, 0.999)`, eps 1e-8) and `opt_psi` (`param_groups_psi`, betas `(0.9, 0.99)`, eps 1e-8); `self._optimizers = [opt_phith, opt_psi]`; `self.optimizer = opt_phith` (alias). Else single-optimizer as today, `self._optimizers = [self.optimizer]`. Cache `self._phith_params, self._psi_params = split_params_phith_psi(self.model)`. Topology decision must happen at `fit()` entry (the orchestrator installs the loss after `__init__`), with `__init__` keeping today's default single optimizer.
- `_rebuild_optimizer`: split-aware — rebuild both with per-side betas; call sites (`stages.py`'s `_rebuild_optimizer(stage.lrs)` under the LR-equality guard, and `handoff.py`'s `_rebuild_optimizer(new_lrs)`) require no signature change.
- **Single-mode ψ betas threading**: optimizer construction and `_rebuild_optimizer` pass `psi_betas=tuple(self.hparams.psi_betas) if self.hparams.psi_betas else None` into `param_groups_for_adamw` (M4's new kwarg). Split mode ignores the knob (per-side betas are structural there). Guard with `getattr(self.hparams, "psi_betas", None)` for hparams objects predating the field.
- `_backward_loss`: dispatch on `SplitLoss` —
  ```python
  lp, lq = loss.phith / accum, loss.psi / accum
  (scaler.scale(lp) if amp else lp).backward(inputs=self._phith_params, retain_graph=True)
  (scaler.scale(lq) if amp else lq).backward(inputs=self._psi_params)
  ```
- `_optimizer_step` — **clip removed, grad skip in**: delete `self.clip_grad_norm` (attr in `__init__` and the clip branch in `_optimizer_step`; M5 deletes the hparam field). New order:
  1. AMP: `scaler.unscale_(opt)` for each of `self._optimizers` (keeps the unscale→inspect→step order; scaler is disabled today but the order stays correct if fp16 is ever enabled — GradScaler's own native skip-on-inf then complements this guard).
  2. `norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))` — norm computation only, `inf` max ⇒ never rescales. One norm over the whole model in both modes.
  3. `if not torch.isfinite(norm):` → `opt.zero_grad(set_to_none=True)` for all `self._optimizers`; `self.grad_skip_count += 1`; `log.warning("[grad-skip] non-finite grad norm %s at step %d (total skips: %d)", ...)`; **return without stepping optimizers, scheduler, or EMA** (a NaN step would poison Adam state and EMA shadows alike). The skipped macro-batch (all accum microsteps) is discarded — correct; count one skip per skipped optimizer step.
  4. Finite path: step all `self._optimizers`; `scaler.update()` once; schedulers (see below); EMA unchanged.
  - Bookkeeping: `self.grad_skip_count: int = 0` in `__init__`; stash `self._last_grad_norm = float(norm)`; `_log_train_step` emits `optim/grad_norm` and `optim/grad_skips` (cumulative) under the existing `optim/*` MetricSpec. The host-side `isfinite`/`float` read syncs once per optimizer step — same cadence as the existing accum-boundary `float(accum_loss_t)` sync, no per-microstep cost.
- `zero_grad` at accum boundaries: loop `self._optimizers`.
- **Schedulers under split mode — ownership resolved**: `self.scheduler` is assigned externally by `stages.py` and `handoff.py`. Under split, the trainer exposes a hook `self._install_scheduler(sched)` that:
  - single-mode: sets `self.scheduler = sched`, `self._schedulers = [sched]` (today's behavior).
  - split mode: sets `self.scheduler = sched` (the phith one, as before), *and* builds `sched_psi` with the same shape (identical `warmup_steps`, `total_steps`, floor) attached to `opt_psi`, and sets `self._schedulers = [sched, sched_psi]`.
  Both `stages.py` and `handoff.py` are edited to route their scheduler assignment through `_install_scheduler` (one-line change per call site). `_optimizer_step` steps `self._schedulers` together on the finite path.
- Validation: `vlog["loss/total"] = vloss.total if isinstance(vloss, SplitLoss) else vloss`.
- Fit-loop scalar sites: `float(loss.detach())` works via `SplitLoss.__float__`; early-stop window consumes the float — no change.

Red tests: `test_trainer_detects_split_loss_flag_and_builds_two_optimizers` (both exist; betas `(0.9,0.999)`/`(0.9,0.99)`); `test_split_backward_routes_grads_correctly` (phith backward populates no ψ grads and vice versa); `test_split_loss_second_backward_requires_retain_graph` (omit flag → RuntimeError; guards future refactors); `test_split_loss_shared_subgraph_no_leak` (grads under split == full backward on `phith+psi` summed per param, on a model whose encoder feeds both branches); `test_psi_trains_during_lambda_zero_warmup` (`rate_lambda≡0`: ψ grads nonzero, φθ sees no KL); `test_split_loss_grad_accum_correct_scaling`; `test_split_loss_amp_scales_both_backwards`; `test_rebuild_optimizer_at_stage_boundary_preserves_split` (both rebuilt, betas preserved); `test_split_schedulers_step_together` (asserts both `self._schedulers[i].last_epoch` increment on a finite step; unchanged on a skip); `test_stages_install_scheduler_routes_split_side` (integration: after `_install_scheduler`, `self._schedulers` has length 2 in split mode, 1 otherwise); `test_validation_logs_scalar_total_under_split`; `test_zero_grad_covers_both_optimizers`.

Grad-skip / betas red tests: `test_nonfinite_grad_skips_step_ema_and_scheduler` (inject a NaN grad; params bit-identical after `_optimizer_step`, EMA shadows unchanged, scheduler `last_epoch` unchanged, grads zeroed, `grad_skip_count == 1`; **disable AMP**); `test_finite_grad_steps_normally_and_never_rescales` (huge finite norm, e.g. 1e12 → step taken with UNCLIPPED grads: resulting param delta matches a hand-computed AdamW step on the raw gradient); `test_grad_skip_covers_both_optimizers_split_mode` (NaN on a ψ param → neither optimizer steps; **disable AMP**); `test_grad_norm_and_skips_logged` (`optim/grad_norm`, `optim/grad_skips` in the log dict); `test_no_clip_grad_norm_attribute_or_branch` (trainer has no `clip_grad_norm` attr; grep-level fence via `not hasattr`); `test_single_mode_psi_betas_threaded` (hparams `psi_betas=[0.9, 0.99]` → single optimizer's ψ groups carry `(0.9, 0.99)`, others `(0.9, 0.999)`; survives `_rebuild_optimizer`); `test_psi_betas_none_is_default_topology` (group dicts identical to pre-change).

### M7 — Checkpoint v3
**Criticality: MEDIUM-HIGH → sonnet.**
**Files:** `src/ddssm/training/checkpoint.py`, `src/ddssm/training/train.py` (restore path only), checkpoint tests (extend existing test file).
**Depends on:** M6.

Spec:
- Bump `_FORMAT` → `ddssm_ckpt_v3`; add to `_SUPPORTED_FORMATS`. `Checkpoint` gains `optimizer_state_psi: dict | None = None`, `scheduler_state_psi: dict | None = None`, `split_loss: bool = False`, `grad_skip_count: int = 0` (careful skip accounting survives preempt-resume; legacy payloads default 0).
- `from_trainer`: snapshot `trainer.opt_psi` / psi scheduler when present; capture `int(getattr(trainer, "grad_skip_count", 0))`.
- `restore_from_checkpoint`: legacy v1/v2 + `use_split_loss=False` → unchanged. Split-mode load requires both optimizer states or errors clearly (mirror the existing scaler/scheduler contract guards). Loading a `split_loss=True` ckpt into a single-loss trainer (or vice versa) → hard error. Restore `trainer.grad_skip_count` from the payload.

Red tests: `test_ckpt_v3_round_trip_split_mode` (save step 5, load fresh, one step, params identical to straight continuation); `test_legacy_v2_ckpt_loads_into_single_mode`; `test_split_ckpt_into_single_trainer_raises`; `test_split_loss_lr_schedulers_survive_round_trip`; `test_grad_skip_count_survives_round_trip` (and legacy v2 → restores as 0).

### M8 — Integration + regression fences
**Criticality: HIGH → opus** (cross-module debugging and tolerance judgment).
**Files:** `tests/test_integration/` (extend; helpers `make_vhp_model`, `run_stage` in `test_integration/conftest.py`), smoke scripts.
**Depends on:** all.

Spec / red tests:
- `test_single_vs_split_agree_at_unit_weight_end_to_end` — force `w_ll≡1`, `p` uniform; 5-step parameter deltas match to tolerance (note: exact equality is not expected — two Adam states vs one differ in step-count bookkeeping; assert tolerance, document why).
- `test_single_loss_off_path_regresses_none` — golden 5-step checkpoint in `use_split_loss=False` mode vs pre-change snapshot.
- `test_split_loss_end_to_end_finite_updates` — 3 steps via `make_vhp_model`; loss decreases; no NaN; both `loss/rate/trans/kl_phith`/`kl_psi` present and finite in the log dict; `optim/grad_skips == 0`.
- `test_grad_skip_recovers_training` — end-to-end: poison one batch to force a non-finite gradient mid-run; training continues, exactly one skip counted, subsequent params finite and moving.
- **Repo fences (grep-level)**:
  - `clip_grad_norm` appears nowhere in `src/` — clipping removed everywhere.
  - `use_reentrant=False` still present at the `torch.utils.checkpoint(...)` call in `_esm_chunk_loss` — the split backward silently corrupts without it.
- Smoke: 100 iters tiny model, single mode (`p_k_clip=1e-3`) vs pre-change baseline within tolerance; same in split mode; `optim/grad_skips` stays 0 in both.
- Run targeted suites (see § Verification) — full fast suite is CI-gated per user preference, not run locally.

---

## Execution plan

Agents run in the shared working tree; each owns its module's files exclusively (no cross-module edits except where a spec explicitly grants one). Every agent follows red-green-refactor: write the module's tests, run to confirm red, implement, run to green, refactor, re-run the module's suite plus any locally-run targeted subset.

| Wave | Modules (parallel) | Agent model | Gate before next wave |
|------|--------------------|-------------|------------------------|
| 0 | Golden capture: record pre-change golden values needed by M1/M2/M8 (`_esm_chunk_loss` scalar under fixed RNG + seed, `_estimator_per_t` outputs on fixed batch, 5-step single-mode checkpoint from `make_vhp_model`) into test fixtures | sonnet (medium) | fixtures committed with reproducer seeds |
| 1 | **M1** (opus) ∥ **M2** (sonnet) ∥ **M3** (sonnet) ∥ **M4** (sonnet) | per module | CI green on touched modules |
| 2 | **M5** (sonnet, medium) ∥ **M6** (opus) | per module | CI green on touched modules |
| 3 | **M7** (sonnet) | sonnet | CI green |
| 4 | **M8** (opus) | opus | CI green + smoke parity |

File-ownership matrix (wave-internal disjointness):
- Wave 1: M1 → `transitions/*.py` + `test_diffusion.py`; M2 → `sigma_data.py` + `test_sigma_data.py`; M3 → `losses.py` + `dssd.py` (`forward` ctor call only) + `test_losses.py` + `test_loss_per_stage.py` + new `test_split_loss.py`; M4 → `train_utils.py` + new `test_param_split.py` + `test_trainer.py` (existing guard). Only `dssd.py` is shared (M3 touches one line, M5 touches the rest in wave 2 — no wave-1 contention).
- Wave 2: M5 → rest of `dssd.py` + new `test_dssd_split.py`; M6 → `train.py` + `stages.py` (`_install_scheduler` routing) + `handoff.py` (same routing) + extension of `test_split_loss.py`. Test-file contention: M5 puts its tests in `test_dssd_split.py`, M6 extends `test_split_loss.py`.
- Wave 3-4: single agent each.

Criticality → model policy (user-directed; minimum sonnet-medium):
- **CRITICAL** (core estimator numerics, gradient routing): opus — M1, M6, M8.
- **HIGH** (statistical/structural correctness with strong test fences): sonnet, high effort — M2, M3, M4, M7.
- **MEDIUM** (mechanical threading, fully specified): sonnet, medium effort — M5, wave-0 capture.

Orchestrator (main session) responsibilities: dispatch wave agents in a single parallel message; after each wave, verify diffs directly (trust-but-verify), CI-gate, resolve any cross-module drift before the next wave.

## Verification (final)

1. Targeted subset (fast, local-friendly): `pytest tests/test_transitions/test_diffusion.py tests/test_training/ tests/test_centering/test_sigma_data.py tests/test_losses.py tests/test_loss_per_stage.py -x`.
2. CI-gated fast suite: `pytest -m "not slow"` (skip locally per user preference; rely on CI).
3. Slow suite in CI: `pytest -m slow` — critically `test_transition_kl_is_invariant_to_num_steps` (IS-normalization guard).
4. Single-mode smoke: 100 iters tiny model, `use_split_loss=False`, `p_k_clip=1e-3` — loss trajectory matches pre-change baseline within tolerance.
5. Split-mode smoke: `use_split_loss=True`; `loss/rate/trans/kl_phith` and `kl_psi` present and finite; `optim/grad_skips == 0`.
6. Checkpoint compat: pre-change v2 checkpoint loads seamlessly with `use_split_loss=False`.
7. `grep -rn clip_grad_norm src/` returns nothing — clipping removed everywhere.
8. `grep -n "use_reentrant=False" src/ddssm/model/transitions/diffusion.py` still matches the `torch.utils.checkpoint(...)` call inside `_esm_chunk_loss` — split-backward compatibility fence.

## Migration & rollback

- Defaults: `use_split_loss=False`, `p_k_clip=1e-3`, `psi_betas=None` — existing configs numerically unaffected on the loss and optimizer paths. Grad clipping is deleted, but it was `None` (off) in every config already, so no run's numerics change; the always-on grad skip only fires on non-finite norms, which previously killed the run.
- Disable p_k clip: `+diffusion_schedule.p_k_clip=null`. Enable split: `+loss.use_split_loss=true`. Enable single-mode ψ betas: `hyperparams.psi_betas=[0.9,0.99]`.
- **Optuna study caution**: `loss/total` in split mode is `phith + psi`, a different magnitude than single-mode. **Do not resume an Optuna study across a `use_split_loss` flip** — start a new `study_name`. Objective values pre/post flip are not comparable.
- Rollback: flip flag off. v3 checkpoint format is backward-readable; `split_loss` marker gates the contract guards.

## Out of scope (NVAE audit outcomes and deferrals)

- Experimental A/B (per user).
- Removing the single-loss path (defer until split validated).
- Further optimizer asymmetry beyond the betas split (`eps`, per-side weight decay, separate ψ schedule shape/warmup/floor).
- Explicit β₂ scheduling for ψ — mechanism documented in M6/design decisions; redundant with φθ LR decay to first order.
- Actual multi-rank DDP wrapping — code DDP-ready; no infra to test against.
- `edm_lognormal` density for the ψ side — training.org explicitly rejects.
- **Threshold-based grad skip** (NVAE's full version: skip finite norms > k× running average) — adds a hyperparameter and an early-training exemption; the `optim/grad_norm` trace this plan adds is the evidence base for bolting it on later (~3 lines on the skip scaffolding).
- **KL balancing (init vs trans γ reweighting)** — NVAE's fix for many-group collapse; with only 2 KL terms the dynamics are mild, and adding nonstationary loss weights during a gradient-routing change would muddy regression attribution. Pure φθ-side composition change; revisit if one KL term flat-lines during the λ ramp.
- **Spectral regularization** — model-side power-iteration machinery, new hyperparameter interacting with weight decay, no observed KL instability. (Note: NOT covered by `r_sigma_p`/`r_mu_p` — those default to 0.0, and the orchestrator's default stage loss hardcodes zeros.)

## Known trade-offs (documented, not fixed)

- **Memory (`retain_graph=True`)**: the first backward must retain the graph so the second can walk the same nodes — `loss_phith` and `loss_psi` both descend from a shared `sqerr = (F_pred − F_tgt)²` and thus a shared forward. Every saved activation (encoder BPTT tape, baseline / aux_posterior / static_embeddings intermediates, reparameterization tensors `z_in = c_in·[μ_t + √(σ_t²+s²)·ε]`, diffmodel UNet intermediates) stays resident until *after* the second backward completes; PyTorch cannot prune conservatively because it doesn't know the second backward's `inputs=` set at first-backward time. The right mental model is **temporal, not additive**: peak activation memory is roughly the same as under single-loss, but held longer — the allocator can't reuse it during the ψ pass. If single-loss is already near the memory ceiling (>~70% of card), split-loss will hit OOM first. Interactions: `grad_checkpoint` (wraps `diffmodel` with `torch.utils.checkpoint(use_reentrant=False)`) still saves memory but pays the recompute cost during *both* backwards → roughly doubled recompute wall-clock for the checkpointed portion; the M8 grep-fence guarantees the `use_reentrant=False` flag stays set, which is what makes the second backward walk the checkpointed section correctly. AMP halves activation size (helps directly); grad_accum retains only within a micro-batch (no compounding). Training.org notes encoder BPTT dominates wall-clock at current scale — expect the same for peak memory here.
- **DDP double-reduce**: with two `.backward(inputs=[...])`, DDP grad-sync hooks fire on every param the graph traversal reaches. Shared-subgraph nodes trigger both a phith-pass and a psi-pass sync. Not a correctness bug (each sync sees fresh grads), but wastes bandwidth. Mitigation: `find_unused_parameters=True` and/or wrap the first backward in `model.no_sync()` when accumulating.
- **No clipping means finite spikes step in raw**: with `clip_grad_norm_` deleted, a large-but-finite gradient (e.g. a near-ceiling `w_ll/p` weight at `w_ll_max·(1+K·c)/c` on a bad batch) now enters the optimizer unclipped — the skip guard only catches NaN/inf. Adam's per-coordinate normalization bounds the *step* to ~lr per coordinate regardless of gradient magnitude, which is the main reason this is tolerable; the residual damage is a transiently inflated β₂ EMA that shrinks effective steps for ~1/(1−β₂) steps afterward (ψ's shorter horizon recovers ~10× faster). The always-logged `optim/grad_norm` trace plus the skip counter is the monitoring contract: if finite spikes show up in practice, the threshold-skip variant (out of scope) is the designated fix — not re-adding clipping.
- **DDP σ_d cadence semantics**: post-reduce, one EMA update to `sigma_data2` represents "one global batch across all ranks," not "one per-rank batch." The EMA horizon `α = 1/(n_updates+1)` therefore weights harder against effective batch size R× (R = world size) than a naïve reading suggests. Intentional — the reduced statistics are what the estimator should be blending — but noted here so future tuning of α is grounded.
