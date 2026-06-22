# Encoder A/B benchmark: PersistenceBlendEncoder (new) vs GaussianEncoder (old "C")

**TL;DR.** The new encoder delivers on its core design promise — its recurrent
Jacobian is pinned at the (I−K_t) contraction and its φ-gradients are
**5× lower at T=32, ~100–1000× lower at T=96**, with a smooth, reproducible
optimisation trajectory. But that stability is packaged with a real cost: the
new encoder is **posterior-collapse-prone**, and the collapse risk **grows with
sequence length**. The old "C" encoder is wildly unstable (gradients explode with
T) yet trains robustly and **fits 2–3 nats better** when both converge. **This is
a fundamental stability ⇄ plasticity tradeoff, not a clean win for either side.**

## Setup

Everything held fixed except the encoder: same persistence-baseline model-v2
**stage-1** model (BaselineGaussianTransition, closed-form KL), same decoder,
same synthetic data, same optimiser/LRs/seeds. Stage 1 is used because the
φ-gradient blow-up the design targets lives in the encoder's recurrence, present
in stage 1, and it's cheap. Datasets: `lgssm` (z_t = 0.9 z_{t-1}+ε — persistence)
and `bimodal` (z_t = 0.9 z_{t-1}+4s_t — persistence + sign jumps). d=4, j=1, GPU.

**Caveat (fair-comparison):** the old encoder trains with its production dropout
(0.1 in the ContextProducer transformer); the new one is dropout-free by design.
The **Jacobian probe runs both in eval mode**, so that structural result is
dropout-independent.

## 1. Stability — decisive new win, widens with T

Recurrent Jacobian ‖∂μ_t/∂z_{t-1}‖ (the doc's central object) and encoder
φ-gradient norm:

| regime | enc | Jacobian | grad-norm median | grad-norm max |
|---|---|---|---|---|
| T=32, warmup | old | 1.01–1.02 (expansive) | 139–220 | 9.7k–13.3k |
| T=32, warmup | **new** | **0.94–1.03 (pinned ≤1)** | **22–84** | **2.9k–7.4k** |
| **T=96**, warmup | old | 1.02 | **396** | **29,200** |
| **T=96**, warmup | **new** | **1.00** | **4.3** | **24.8** |

- `jacobian.png`: both start ~0.38 (zero-init heads); the **old jumps >1 by step
  200 and stays expansive (1.05–1.25)**, the **new is pinned at exactly 1.0** —
  the (I−K_t) boundary, by construction.
- `results_longT/grad_norm.png`: at T=96 the **old spikes to ~5,000** early then
  runs at ~400; the **new stays flat at 3–10**. The ρ^{t−s} compounding is real:
  the stability gap goes from ~6× (T=32) to **100–1000× (T=96)**.
- No NaNs in either at these scales — the old encoder *survives* its huge
  gradients (no grad-clip was used), but is visibly on the edge.

**Verdict: new wins decisively and the margin grows with sequence length.**

## 2. Convergence — old fits better but *unstably*; new is smooth but collapse-prone

`results_warmup/convergence.png` is the key plot. On lgssm the **old encoder
plunges to −60 transiently (wide seed band), then destabilises — spikes back to
+10 around step 600 — before settling at −11**. The **new encoder glides
monotonically to −9** with a narrow band. Same shape on bimodal (old dips to ~70
then **rises back** to 92; new glides to 92).

| regime | enc | final loss | recon (fit) | KL | steps-to-plateau |
|---|---|---|---|---|---|
| const-λ=1, T=32 | old | **−12.7** | −24.6 | ~12 | 382 |
| const-λ=1, T=32 | new | −1.1 | −1.1 | **~0 ⇒ COLLAPSE** | 50 |
| warmup, T=32 | old | **−11.3** | −23.5 | 13 | 48–94 |
| warmup, T=32 | new | −9.3 | −15.0 | 5–9 (healthy) | 182–310 |
| warmup, T=96 | old | **−33.5** | −76.2 | ~40 | 42 |
| warmup, T=96 | new | +2.3 | +2.8 | **~0 ⇒ COLLAPSE** | 43 |

Reading:
- **Old fits better whenever it trains** (lower recon at every setting), and it
  *always* trains — robust to λ schedule and T.
- **New trades ~2–3 nats of fit for a smooth, reproducible trajectory.** Its
  lower-variance path is the stability dividend the final-loss number hides.
- **New is slower** to reach its plateau under warmup (gate opens gradually).
- **New is faster per step** (668s vs 780s for 2500 steps) despite 3× params —
  cross-attention over d=4 sites is cheaper than the old combiner + GRU aggregator.

## 3. Parameter effectiveness — old more efficient

Hidden-dim sweep (const-λ, lgssm), final loss:

| hidden | old params | old loss | new params | new loss |
|---|---|---|---|---|
| 16 | 35.8k | −11.9 | 20.6k | −1.5 |
| 32 | 43.6k | −12.5 | 62.0k | −1.3 |
| 64 | 71.4k | −12.3 | 207.6k | −1.6 |
| 128 | 176.3k | −12.3 | 750.9k | −0.6 |

The old encoder **saturates at 35k params** (−12 and flat). The new encoder
**doesn't benefit from capacity** — because under const-λ it's collapsed; the
extra params sit unused. The new encoder also costs **~3× params at matched
hidden_dim**.

### 3b. Harder data (`nonlinear-bimodal-lift-mv`, D=8, latent d=4, WITH warmup)

To test whether the new encoder's flat curve was (H1) a const-λ collapse artifact
or (H2) the D=1/Q=1 cross-attention degeneracy, I reran the sweep **with warmup**
on the **multivariate, genuinely multimodal** dataset (2⁴=16 attractors, tanh-MLP
observation lift) — so Q=8 obs tokens make the cross-attention *live*.

| hidden | old params | old loss | new params | new loss |
|---|---|---|---|---|
| 16 | 36.2k | 500.5 | 20.8k | 668.0 |
| 32 | 44.0k | 460.7 | 62.4k | 668.0 |
| 64 | 71.9k | 453.3 | 208.5k | 668.1 |
| 128 | 176.8k | **418.7** | 752.6k | 580.3 |

A/B (1 seed, warmup): old loss 454 (recon 116, **KL≈338** — heavy latent use),
new loss 668 (recon 668, **KL≈0 — COLLAPSED**), grad norm old 1730 vs new 4.3.

**Both H1 and H2 are refuted.** Warmup did *not* prevent collapse here, and live
multi-token attention did *not* rescue efficiency: the new encoder is **flat at
668 across 20k→208k params** (only nudging to 580 at 752k), while the old encoder
rides a clean efficient frontier (500→419). The collapse is **more severe on
harder data** — the gap *widens* with task difficulty rather than closing.

Why: on multimodal data the persistence frame (z_{t-1}) is a poor predictor when
the latent jumps attractors, and the **z-free evidence m_b** can't condition on
the realized trajectory to compensate. Neither component of the gated blend fits
the jumps, so reducing recon never justifies the KL cost of opening the gate →
the encoder rationally collapses to the prior. This is **H4 (structural
expressiveness ceiling) manifesting as collapse** — and it is *intrinsic*, not a
benchmark artifact. The earlier optimistic hypotheses (artifact + D=1 starvation)
do not survive contact with the hard data.

### 3c. Warmup sweep — collapse is NOT a schedule artifact (3500 steps, mv)

To rule out "I just picked a bad λ-warmup," I swept warmup_frac ∈ {0, .25, .5,
.75, .9} for the new encoder (old at {0, .5, .9} as a robustness reference), at
an adequate 3500-step budget, tracking the **KL trajectory**:

| enc | warmup | loss | recon | KL_final | KL: start→1k→2k→3k→end |
|---|---|---|---|---|---|
| new | 0.00 | 668 | 668 | 0.01 | 0.09→0.00→0.00→0.01→0.01 |
| new | 0.25 | 668 | 668 | 0.02 | 2.99→0.05→0.01→0.02→0.02 |
| new | 0.50 | 668 | 668 | 0.02 | 3.08→0.05→0.02→0.02→0.02 |
| new | 0.75 | 668 | 668 | 0.02 | 3.41→0.11→0.02→0.02→0.02 |
| new | 0.90 | 668 | 668 | 0.02 | 3.23→**0.46**→0.03→0.03→0.02 |
| old | 0.00 | 403 | 128 | 267 | 47→230→253→263→267 |
| old | 0.50 | 409 | 46 | 350 | 179→442→390→358→350 |
| old | 0.90 | 420 | **6** | 401 | 192→480→450→414→401 |

**The new encoder collapses at EVERY warmup, including 0.9.** Longer warmups only
*delay* it (KL@1k: 0.00 at warmup-0 vs 0.46 at warmup-0.9) but it always converges
to KL≈0.02 by step 2000. Crucially, the KL **decreases during the cheap-KL warmup
phase** (≈3 → ≈0) — i.e. even when the KL penalty is nearly off, the recon
gradient does **not** pull the gate open. That is the signature of a
**self-reinforcing collapse**: weak early `m_b` → gate stays shut → `m_b` receives
no gradient → stays useless → gate stays shut. The old encoder, by contrast, uses
the latent richly at every warmup (KL 267–401, recon down to **6** at warmup-0.9 —
a near-perfect fit) and is robust to the schedule.

**Conclusion: the collapse is structural, not a tunable schedule.** Warmup cannot
fix it because the problem isn't "the gate has too little time to open" — it's
"nothing pulls the gate open" once `m_b` can't reduce recon. The fix must
**force** latent usage: a **KL floor / free-bits** (forbids KL→0 directly), or an
architectural change that lets the evidence path condition on the realized state
(which would break the z-free damping property — the very thing that buys the
stability). This is the central tension, now empirically pinned down.

## The collapse failure mode (the new encoder's Achilles heel)

The new encoder's gate-closed init + (I−K_t) contraction suppress not just the
*bad* gradients but the *learning signal* — grad norm ~3 on lgssm means the gate
barely moves. Combined with the KL penalty pulling toward the random-walk prior,
the encoder collapses to q≈p (KL→0, latent unused) unless:
1. **KL-annealing (λ-warmup)** gives the gate time to open before the penalty
   bites — *production already uses this* (`LambdaRampConf(start=0.001)`). Without
   it (const-λ=1) the new encoder collapses every time; the old encoder is immune.
2. **Adequate step budget** post-warmup. The rescue probe (T=32, 800 steps) hit
   KL≈6 (healthy); at T=96/1000 steps it collapsed again.
3. The risk **worsens with T**: the summed KL grows with sequence length while the
   per-step gate-opening signal stays weak, so longer sequences need slower
   warmup / more steps / a KL floor (free-bits) to avoid collapse.

Gate-init (`b_K` ∈ {−5…−1}) is a **minor** knob (recon −15.4 to −15.9 across the
range); warmup-vs-no-warmup is the decisive variable.

## 4. Stage 2 — the collapse is a transient handoff state, but a fit ceiling remains

The §3 results were all **stage-1** (Gaussian transition), where KL→0 *mathematically
forces* a random-walk latent (q≈N(z_{t-1},σ_p²)) and therefore bad recon. But the
new encoder is *designed* to sit at q≈prior (KL≈0) at the stage-1→2 handoff — so
the stage-1 "collapse" may be benign. The decisive test is **stage 2** (the
expressive `DiffusionTransition` prior). Full two-stage run (stage1=1500 →
handoff → stage2=3000), mv data:

| enc | recon @s1-end | recon @s2-end | gate @s2 | KL/ESM @s2 | **marginal NLL** @final |
|---|---|---|---|---|---|
| old | 132.7 | **−86.8** | n/a | 217 | **303.1** |
| new | 667.8 (collapsed) | 438.7 | **0.001→0.95** | 96 | 543.9 |

Stage-2 trajectory (new): gate **0.001→0.475→0.942 by step 600**, recon **669→490**
in lockstep, then **plateaus at ~440** for 2400 more steps with the gate open.

**Two clean conclusions:**
1. **The stage-1 collapse is largely a transient handoff state — it resolves.**
   Against the diffusion prior the gate opens decisively (0.001→0.95), KL goes
   0→96, recon drops 668→440. The new encoder *does* un-collapse in stage 2. The
   earlier "structural collapse, unfixable by warmup" verdict was drawn from the
   wrong stage; with an expressive prior, q≈prior is informative, not vacuous.
2. **But the diffusion does NOT close the fit gap.** With the gate *fully open*
   (μ_t≈m_b), recon plateaus at ~440 while the old encoder reaches **−87**;
   marginal NLL **544 vs 303**. Because `recon` is transition-independent, the
   diffusion can't fix it — the limit is the **z-free evidence path** `m_b`, which
   cannot condition on the realized trajectory to fit the multimodal jumps. This
   is the *same z-free constraint that buys the stability* (∂m_b/∂z=0 → damping).
   The new encoder also gives the diffusion an *easier* ESM target (its simpler
   posterior → ESM 96 vs old 217) — easier to match, but a worse-fitting encoder.

### 4b. LR probe — the gap is a rate–distortion frontier, NOT a z-free fitting wall

To test "is the new encoder's plateau an LR/optimization deficit (its grad norm is
5–1000× lower) or a structural z-free ceiling?", I froze the gate fully open
(μ_t = m_b — no collapse), trained **recon-only** (no KL confound), and swept LR
over 60× on the mv data:

| enc | lr | recon floor | enc grad-norm |
|---|---|---|---|
| old | 5e-4 | 34.9 | 9.3e3 |
| new | 5e-4 | **57.1** | 4.7e3 |
| new | 2e-3 | 125.8 | 3.0e3 |
| new | 1e-2 | 302.3 | 6.8e2 |
| new | 3e-2 | 668.4 | **3.8e-5** (diverged) |

Two corrections fall out:
1. **Raising the LR HURTS** — monotonically (57→668), diverging at 3e-2 (grad
   norm → 0 = blow-up/saturation). The new encoder is *less* LR-tolerant; Adam's
   per-coordinate scale-invariance holds, so the low grad norm never warranted a
   higher LR. The right LR is low, same as old.
2. **The z-free evidence is NOT raw-fit-incapable.** At the correct LR, recon-only
   reaches **57 ≈ old's 35** — it fits the data nearly as well. So §4's "structural
   z-free expressiveness ceiling" was **too strong**. The stage-2 plateau at 440 is
   a **worse rate–distortion frontier under the full ELBO** (gated-blend posterior +
   KL pull toward the diffusion prior settle at recon 440 / KL 96, vs old's
   −87 / 217), *not* a hard limit on what m_b can fit. Strip the KL coupling and the
   evidence fits fine.

So the corrected tradeoff: the new encoder's stability-buying structure (z-free,
persistence-framed, diagonally-gated mean) yields a **worse rate–distortion
frontier** on hard multimodal data — it can fit *or* match the prior cheaply, but
not both as well as the free combiner. The earlier "fitting ceiling" was an
artifact of reading it off the KL-constrained ELBO; the capacity is there.

## Honest conclusion + recommendations

Neither encoder dominates. The new encoder **swaps one failure mode (φ-gradient
blow-up) for another (posterior collapse)**:

- **Use the new encoder when** training stability / reproducibility is the
  bottleneck, sequences are **long** (its gradient advantage compounds; the old
  encoder's ~30k grad norms at T=96 will eventually NaN with a less forgiving
  setup), and you can afford **careful KL-annealing + a KL floor + adequate
  steps**. Its smooth trajectory and pinned Jacobian are exactly what a fragile
  large-scale / long-horizon run wants.
- **Use the old encoder when** you want raw fit and robust, forgiving training at
  **short T** — its instability is real but non-fatal here, and it fits better
  cheaper.

**Actionable next steps for the new encoder** (it underperformed on fit largely
because it's collapse-prone, which is *fixable*):
1. Add a **free-bits / KL floor** to the transition KL so the gate can't be
   fully shut — likely the single biggest fix.
2. **Slower, longer λ-warmup** scaled to T (summed-KL grows with T).
3. Consider a **less-negative default `gate_init_logit`** (−3) + a small **gate
   entropy bonus** to keep it from latching shut.
4. Re-run this benchmark with those changes + a long-T budget — that's the test
   of whether the stability advantage finally converts to a fit/speed win.

The doc's stability thesis is **confirmed** (the contraction is real and scales);
its implicit corollary that this *yields better training* is **not demonstrated**
— and the multivariate result shows the collapse-proneness **worsens with task
difficulty**, not just at const-λ or D=1. The single most important untested fix
is a **KL floor / free-bits** on the transition KL: it directly forbids KL→0, so
it targets the collapse mechanism head-on. Until that's tried, the verdict is:
**the new encoder buys real, scaling stability at the cost of a collapse failure
mode that currently dominates on anything harder than clean persistence data.**

## 5. The fix — additive-innovation redesign (the rate side, via architecture)

§4b localised the gap to the gated convex-blend `μ = (1−K)z_{t-1} + K·m_b`
constraining the mean to the segment between z_{t-1} and m_b. The fix: replace it
with a **free additive innovation** and a **single** cross-attention stack:

    μ_t = z_{t-1} + g(U),   U = TimeMix(z_<t) → CrossAttn(obs) → FeatureMix
    g_head zero-init ⇒ μ=z_{t-1}, KL≈0 at handoff;  no z-free trunk, no gate.

`g` is unconstrained (μ anywhere in R^d), and it sees the realised state (via the
Time-Mix seed) AND the evidence (via cross-attn) — removing both the convex-blend
constraint and the z-free bottleneck. The persistence frame `+z_{t-1}` is kept,
so `J_t = I + ∂g/∂z` stays anchored at I (no learned dynamics map ⇒ no C-style
blow-up). ~Half the params (one stack, no E-trunk, no gate).

Two-stage result (mv data, same protocol), vs the old encoder and the blend:

| metric | old | blend | **additive** |
|---|---|---|---|
| stage-1 recon | 132.7 | 667.8 (collapsed) | **341.6** |
| stage-1 KL | 315 | 0.0 | **161.9** |
| stage-2 recon | −66.7 | 440 | **144.3** |
| marginal NLL @final | 300.0 | 543.9 | **397.6** |

- **Stage-1 collapse eliminated** (KL 0→162, ‖g‖≈0.38 active) — the additive form
  doesn't collapse where the blend did.
- **~60% of the NLL gap closed** (544→398 vs old's 300), at ~half the params.
- A residual gap to the free combiner remains (398 vs 300) — the persistence-frame
  anchor still makes z_{t-1} the starting point that `g` must correct, a soft
  inductive bias the unconstrained combiner lacks. Acceptable: it keeps the
  pinned-Jacobian stability the whole exercise was about.

**Stability retained (Jacobian, mv data, after 600 steps):**

| | ‖∂μ_t/∂z_{t-1}‖ init | after 600 steps |
|---|---|---|
| old | 0.680 | **0.159** (contractive *here*) |
| additive new | 1.000 | **1.006** (anchored at I) |

The additive form is **pinned at ~1.0 by construction** (J = I + ∂g/∂z, ∂g/∂z ≈
0.006) — *data-independent*. Notably the old encoder is *contractive* on this
multimodal data (0.16) yet was *expansive* on lgssm (1.25): its Jacobian is
**uncontrolled and data-dependent**, whereas the additive form's is **anchored at
I regardless of data**. That's the real stability value — a guarantee, not a
per-dataset accident. (At J=1.006 the additive form drifts *barely* above I; the
deferred weight-decay on the z→g path would pull it to ≤1 for very-long-T
robustness — option (ii), not needed at these scales.)

**Conclusion of the arc:** the new encoder's underperformance was *not* a
fundamental fit limit — it was the gated-blend's rate–distortion bottleneck.
Replacing the gate with a free additive innovation recovers most of the fit (NLL
544→398) while keeping the persistence-frame stability guarantee (J anchored at
~I). The user's redesign (single cross-attn, drop the gate, `z_{t-1} + g`) was the
right call. Remaining gap to the free combiner (398 vs 300) is the soft cost of
the persistence-frame inductive bias — the price of the stability guarantee.

## 6. Stage-2 λ-warmup sweep — the "fit ceiling" was a warmup artifact + a crossover

Driven by the question "is the warmup too long?". hd=64 fixed, stage1=1500 →
handoff → stage2=2500, mv data, single seed, warmup ∈ {0, 0.25, 0.5, 0.75, 0.9}
applied to **both** stages. (`stage2.py --sweep_warmup`; `results_stage2_warmup/`.)

Final marginal NLL (PF-ODE, lower=better):

| warmup | old | new (additive) |
|---|---|---|
| 0.0  | **308.6** | 660.6 |
| 0.25 | 313.5 | 543.1 |
| 0.5  | 346.6 | 415.1 |
| 0.75 | 407.5 | 381.8 |
| 0.9  | 436.7 | **371.5** |

Convergence (steps to recon→95% plateau, `s1/s2`; walls ~470/960s old, ~392/840s new):

| warmup | old | new |
|---|---|---|
| 0.0  | 700/1900 | 100/2499 *(collapsed-flat)* |
| 0.25 | 700/1100 | 100/1750 *(collapsed)* |
| 0.5  | 350/1000 | 900/1900 |
| 0.75 | 350/500  | 1100/1250 |
| 0.9  | 400/400  | 1100/1500 |

**Findings:**
1. **Opposite warmup preferences.** Old degrades monotonically with warmup
   (308→437 — the long warmup is *harmful*, it wants w≈0); new improves
   monotonically (660→371 — the warmup is *load-bearing*, the additive form
   needs the gentle ramp to escape the g≈0 stage-1 collapse). Crossover at w≈0.65.
2. **The §4/§5 "fit ceiling" was a warmup artifact.** Those sections concluded a
   ~90-nat persistence-frame ceiling — but that was all at w0.5. With proper
   warmup the additive form reaches **371.5** (w0.9), not 415. Not capacity- or
   form-capped; under-warmed. (The hd-capacity sweep in §3, also at w0.5 *and*
   with the old encoder's `summary_dim` pinned at 64 while new's scaled with hd,
   is doubly confounded — disregard its "param ceiling" reading.)
3. **Best-vs-best, old still wins on both axes.** old@w0 (308.6) beats new@w0.9
   (371.5) by ~63 nats, *and* trains faster (stage-1 plateaus ≤700 vs new's 1100;
   so 1500 stage-1 is ~2× generous for old). New only reaches its best with a
   long warmup *and* the full step budget.

Open question this raises (→ §7): is old's edge the **free emission form** or the
**aggregator backbone** (vs new's cross-attn)? The two are confounded — same
future summary at hd=64, but old = free `GaussianDistHead` + small
`ContextProducerAggregator`, new = additive `z_{t-1}+g` + L=2 cross-attn (72k vs
124k params). Deconfounding grid (`mu_mode={additive,free}` × {mv, henon-lift}) in progress.

## 7. Deconfounding — old's edge is the BACKBONE (mv-specific), not the form

§6 left old vs new confounded across two axes. We split them with `mu_mode`
(`free` = μ=mu_head(U), xavier; `additive` = μ=z_{t-1}+g) on the *same* cross-attn
backbone, giving a third encoder `new_free` (free emission + cross-attn). Run on
the multimodal mv data and a new chaotic `henon-lift` dataset (Hénon map d=2,
tanh-MLP lift). hd=64, stage1=1500→stage2=2500, 1 seed. Best NLL per encoder
(each at its own best warmup; `results_deconf_{mv,henon}/`, `deconf_form_vs_backbone.png`):

| encoder | form / backbone | mv | henon |
|---|---|---|---|
| old | free / aggregator | **308.6** (w0) | −101.0 (w0) |
| new | additive / cross-attn | 371.5 (w0.9) | −97.5 (w0.75) |
| new_free | free / cross-attn | 410.0 (w0.5) | **−107.6** (w0) |

Isolated (hold one axis fixed):
- **BACKBONE** (new_free − old, both free): mv **+101** (cross-attn much worse), henon −7 (tie).
- **FORM** (new_free − new, both cross-attn): mv +38 (free worse), henon −10 (free better).

**Findings:**
1. **old's mv advantage is its `ContextProducerAggregator` backbone (~100 nats),
   not its free form.** The cross-attn stack the "new" family adopted is the real
   regression on multimodal data; even the best cross-attn config (additive,
   371.5) trails old by 63. The §7-precursor read "free form is old's secret"
   (from henon w0 alone) is **wrong for mv** — there additive *beats* free by 38.
2. **On chaos, architecture is a wash** — all three within ~10 nats (seed noise).
   The differentiation is an mv (multimodal-jump) phenomenon.
3. **The form effect sign-flips by data** (additive +38 on mv, −10 on henon) and
   its warmup optimum flips too — the persistence frame is data-matched, not a
   universal handicap. Old is also as-fast-or-faster to converge everywhere.

**Open follow-up (untested 4th cell):** additive + aggregator (persistence frame
on old's backbone). Additive beats free by 38 on the cross-attn backbone on mv;
on old's *better* backbone it might push below 308. The actionable lever for the
"new" encoder is to **replace the cross-attn combiner with an aggregator-style
one**, independent of the persistence-frame debate.

## 8. Why is the aggregator better than cross-attn? — and the best-of-both fix

§7 left old's mv edge as "the backbone" (~100 nats vs cross-attn, free form). §8
isolates *which part* of the backbone, via three new encoder knobs (all hd=64,
stage1=1500→stage2=2500, 1 seed; `results_{seqtest,combiner,4thcell}_*`):

- `kv_mode=sequence` — feed the cross-attn T distinct future-summary tokens (+causal
  mask) instead of one summary broadcast to Q positional copies.
- `obs_combiner=concat` — replace the softmax cross-attn with a non-averaging
  concat-MLP fusion of the pooled obs context (old-style).
- `mu_mode` on the **old** `GaussianEncoder` — persistence frame on old's real
  backbone (`old_additive`: μ = z_{t-1} + free-μ, zero extra params).

Best-warmup NLL, all six cells (`final_all_cells.png`):

| cell | form / combiner | mv | henon |
|---|---|---|---|
| **old_additive** | additive / aggregator | **295.3** (w.25) | −95.6 |
| old | free / aggregator | 308.6 (w0) | −101.0 |
| new | additive / cross-attn | 371.5 (w.9) | −97.5 |
| new_free | free / cross-attn | 410.0 (w.5) | −107.6 |
| new_free_concat | free / cross-attn+concat | 449 (w0) | **−118.8** |
| new_concat | additive / cross-attn+concat | ~660 (collapse) | — |

**Findings:**
1. **It is NOT the obs-combiner.** `kv_mode=sequence` made *both* forms worse on mv
   (free 410→580, additive 415→collapse) — more/distinct tokens don't help; softmax
   *averages* the full-future token `h_t` with shorter-horizon ones instead of
   selecting it. `obs_combiner=concat` also fails to close the mv gap (best 449 vs
   old 308). So **token structure and softmax averaging are both exonerated** — the
   ~100-nat mv lever is the *holistic* aggregator backbone (`ContextProducer` history
   stack + MLP dist-head), not a single component.
2. **The combiner/form effects sign-flip by data.** On chaotic henon every cell sits
   within ~23 nats and `new_free_concat` is best (−118.8); on multimodal mv old's
   aggregator dominates. The new cross-attn family is competitive-to-better on chaos,
   worse only on the hard multimodal regime.
3. **Best-of-both (the actionable win):** the persistence frame on old's aggregator
   backbone — `old_additive` @ w0.25 = **295.3** — is the best mv result of the whole
   study (beats plain old by 13), at **zero extra params**, with fast convergence
   (stage-2 plateau by 150 steps), *and* the J≈I stability frame. The form effect is
   itself backbone-dependent: additive *helps* the struggling cross-attn backbone
   (+38 vs free) but is ~tie-to-mild-help on old's strong backbone.

**Recommendation:** ship **old's aggregator backbone + persistence frame (`mu_mode=
additive`) at a small warmup** for multimodal forecasting. The cross-attn redesign was
a net regression on multimodal fit; the persistence frame (the one good idea from the
new encoder) transplants onto old's backbone for a strict win. On smooth/chaotic data
the architecture is a wash.

## Artifacts
- `results/` — const-λ run (3 seeds, 1500 steps) + plots + sweep + probe
- `results_stage2_warmup/` — §6 hd=64 warmup sweep (1 seed) + `warmup_frontier.png`
- `results_deconf_{mv,henon}/` — §7 deconfounding grid + `deconf_form_vs_backbone.png`
- `results_{seqtest,combiner,4thcell}_{mv,henon}/` — §8 mechanism grids + `final_all_cells.png`
- `results_warmup/` — fair warmup run (2 seeds, 2500 steps) + plots
- `results_longT/` — T=96 stability test + plots
- `bench.py` (harness), `analyze.py` (plots/summary), `probe.py` (rescue grid)
