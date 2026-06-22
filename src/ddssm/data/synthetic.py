"""Synthetic time-series datasets for controlled DDSSM experiments (IID, LGSSM, etc.)."""

import numpy as np
import torch
from torch.utils.data import Dataset

# Nonlinear-bimodal-lift constants shared with
# ``ddssm.eval.synthetic_kernels`` so the closed-form transition kernel
# matches the data generator exactly.
NLBL_DELTA = 2.0
NLBL_SIGMA_Z = 0.1
NLBL_SIGMA_X = 0.1
NLBL_LIFT_HIDDEN = 8

# Multivariate variant (``nonlinear-bimodal-lift-mv``):
# latent ``d = NLBL_MV_LATENT_D`` driven through a fixed mixing matrix
# ``A`` sampled once from ``NLBL_MV_A_SEED``; observation lifted via
# a tanh-MLP to ``D = NLBL_MV_OBS_D`` channels.
NLBL_MV_LATENT_D = 4
NLBL_MV_OBS_D = 8
NLBL_MV_HIDDEN_DIM = 16
NLBL_MV_A_SEED = 12345

# Chaotic variant (``henon-lift``): a deterministic Hénon-map latent
# (d = 2, a = 1.4, b = 0.3 — the classic chaotic regime) plus small process
# noise, standardised per-dim, lifted via a tanh-MLP to ``D = HENON_OBS_D``.
# Sensitive-dependence / deterministic chaos is a stress test distinct from
# the noise-driven multimodality of the bimodal-lift family. The map state is
# clamped to a bounding box so process noise can't kick a trajectory off the
# attractor into the divergent region.
HENON_A = 1.4
HENON_B = 0.3
HENON_LATENT_D = 2
HENON_OBS_D = 8
HENON_HIDDEN_DIM = 16
HENON_SIGMA_Z = 0.01
HENON_SIGMA_X = 0.1
HENON_BURN_IN = 100
HENON_A_SEED = 23456


class SyntheticDataset(Dataset):
    """Sequence dataset of synthetically generated time series.

    A single generated population of ``3 * N_per_split`` sequences is
    partitioned into deterministic disjoint ``train``/``val``/``test``
    slices. Each item is a ``(D, T)`` sequence with an all-ones
    observation mask; see :meth:`__getitem__` for the emitted dict.
    """

    def __init__(
        self,
        mode: str,
        split: str = "train",
        N_per_split: int = 1024,
        T: int = 100,
        D: int = 1,
        seed: int = 42,
        dataset_seed: int = 1234,
        expose_gt_latents: bool = False,
        expose_clean_data: bool = False,
    ):
        """Generate the population and keep the slice for ``split``.

        Args:
            mode: Generator mode (e.g. ``"iid"``, ``"lgssm"``,
                ``"nonlinear-bimodal-lift"``); see :meth:`_generate_data`.
            split: One of ``"train"``, ``"val"``, ``"test"``.
            N_per_split: Number of sequences per split.
            T: Length of each sequence.
            D: Data (observation) dimension.
            seed: Legacy; ignored (splitting is driven by ``dataset_seed``).
            dataset_seed: Seed for data generation and the split.
            expose_gt_latents: When True, ``__getitem__`` returns an
                additional ``gt_latent`` field containing the
                ground-truth latent ``z`` underlying each sequence.
                Available for modes whose latent dynamics have a
                registered closed-form transition kernel in
                :mod:`ddssm.eval.synthetic_kernels`. Today: ``lgssm``,
                ``nonlinear-bimodal-lift``, ``nonlinear-bimodal-lift-mv``.
            expose_clean_data: When True, ``__getitem__`` returns a
                ``clean_data`` field with the noise-free observation-space
                trajectory (before observation noise is added). Distinct
                from ``gt_latent`` (which is model-latent-space, dim d);
                ``clean_data`` is observation-space (dim D). Supported for
                modes that add explicit observation noise: ``lorenz``.
        """
        self.mode = mode
        self.split = split
        self.N_per_split = N_per_split
        self.T = T
        self.D = D
        self.expose_gt_latents = bool(expose_gt_latents)
        self.expose_clean_data = bool(expose_clean_data)
        # Populated by _generate_data when the mode supports it.
        self._all_gt_latents: torch.Tensor | None = None
        self.gt_latents: torch.Tensor | None = None
        self._all_clean: torch.Tensor | None = None
        self.clean_data: torch.Tensor | None = None

        self.N_total = 3 * N_per_split
        all_data = self._generate_data(dataset_seed)

        if split == "train":
            self.data = all_data[:N_per_split]
            if self._all_gt_latents is not None:
                self.gt_latents = self._all_gt_latents[:N_per_split]
            if self._all_clean is not None:
                self.clean_data = self._all_clean[:N_per_split]
        elif split == "val":
            self.data = all_data[N_per_split : 2 * N_per_split]
            if self._all_gt_latents is not None:
                self.gt_latents = self._all_gt_latents[N_per_split : 2 * N_per_split]
            if self._all_clean is not None:
                self.clean_data = self._all_clean[N_per_split : 2 * N_per_split]
        elif split == "test":
            self.data = all_data[2 * N_per_split : 3 * N_per_split]
            if self._all_gt_latents is not None:
                self.gt_latents = self._all_gt_latents[2 * N_per_split : 3 * N_per_split]
            if self._all_clean is not None:
                self.clean_data = self._all_clean[2 * N_per_split : 3 * N_per_split]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Free the full tensors; the per-split slices are what we keep.
        self._all_gt_latents = None
        self._all_clean = None

        self.N = len(self.data)

    def _generate_data(self, seed):
        """Generate the full ``(N_total, D, T)`` population for ``self.mode``.

        Seeds torch and numpy with ``seed`` for reproducibility and, for
        latent modes with ``expose_gt_latents`` set, records the clean
        latent path in ``self._all_gt_latents``.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = torch.zeros(self.N_total, self.D, self.T)

        if self.mode == "iid":
            # x_t ~ N(0, 1)
            data = torch.randn(self.N_total, self.D, self.T)

        elif self.mode == "lgssm":
            # z_t = 0.9 * z_{t-1} + N(0, 0.1)
            # x_t = z_t + N(0, 0.1)
            z = torch.zeros(self.N_total, self.D, self.T)
            for t in range(1, self.T):
                z[:, :, t] = 0.9 * z[:, :, t - 1] + 0.1 * torch.randn(
                    self.N_total, self.D
                )
            data = z + 0.1 * torch.randn(self.N_total, self.D, self.T)
            # Retain the underlying clean latent for the GT-latent
            # surface (used by ``gt_latent_jsd`` and
            # ``crps_sum_latent`` metrics) only when explicitly
            # requested — otherwise drop the reference so it can be
            # garbage-collected.
            if self.expose_gt_latents:
                self._all_gt_latents = z

        elif self.mode == "nonlinear":
            # z_t = sin(z_{t-1}) + N(0, 0.1)
            z = torch.zeros(self.N_total, self.D, self.T)
            # Random start
            z[:, :, 0] = torch.randn(self.N_total, self.D)
            for t in range(1, self.T):
                z[:, :, t] = torch.sin(z[:, :, t - 1] * 3) + 0.1 * torch.randn(
                    self.N_total, self.D
                )
            data = z  # Direct observation for simplicity

        elif self.mode == "nongaussian":
            # Mixture of Gaussians transition
            # z_t ~ 0.5 * N(z_{t-1} - 1, 0.1) + 0.5 * N(z_{t-1} + 1, 0.1)
            z = torch.zeros(self.N_total, self.D, self.T)
            z[:, :, 0] = torch.randn(self.N_total, self.D)

            for t in range(1, self.T):
                # Choose component for each sample in batch
                comps = torch.randint(0, 2, (self.N_total, self.D)).float()  # 0 or 1

                # Shift: if 0 -> -1.0, if 1 -> +1.0
                shift = (comps * 2.0) - 1.0

                # z_t = z_{t-1} + shift + noise
                z[:, :, t] = (
                    0.9 * z[:, :, t - 1]
                    + shift
                    + 0.2 * torch.randn(self.N_total, self.D)
                )

            data = z

        elif self.mode == "harmonic":
            # Clean sine waves: x_t = sin(omega*t + phi) + epsilon
            # omega ~ U[0.3, 0.6]
            # phi ~ U[0, 2pi]
            # Very low noise to verify dynamics learning
            z = torch.zeros(self.N_total, self.D, self.T)
            t_grid = torch.arange(self.T, dtype=torch.float32)

            for i in range(self.N_total):
                # Random frequency and phase
                omega = 0.3 + 0.3 * torch.rand(1).item()
                phi = 2 * np.pi * torch.rand(1).item()

                sig = torch.sin(omega * t_grid + phi)
                # Minimal observation noise
                z[i, 0, :] = sig + 0.05 * torch.randn(self.T)

            data = z
        elif self.mode == "harmonic-noisy":
            # Noisy sine waves: x_t = sin(omega*t + phi) + epsilon
            # omega ~ U[0.1, 0.4]
            # phi ~ U[0, 2pi]
            # Moderate noise to verify dynamics learning
            z = torch.zeros(self.N_total, self.D, self.T)
            t_grid = torch.arange(self.T, dtype=torch.float32)

            for i in range(self.N_total):
                # # Random frequency and phase
                omega = 0.1 + 0.3 * torch.rand(1).item()
                phi = 2 * np.pi * torch.rand(1).item()

                sig = torch.sin(omega * t_grid + phi)
                # Moderate observation noise
                z[i, 0, :] = sig + 0.2 * torch.randn(self.T)

            data = z

        elif self.mode == "bimodal":
            # Bimodal random walk:
            # z_t = 0.9 z_{t-1} + 4 s_t, s_t in {-1, +1}
            # x_t = z_t + eps_t, eps_t ~ N(0, 0.2)
            z = torch.zeros(self.N_total, self.D, self.T)
            z[:, :, 0] = torch.randn(self.N_total, self.D)

            for t in range(1, self.T):
                s = (torch.randint(0, 2, (self.N_total, self.D)).float() * 2.0) - 1.0
                z[:, :, t] = 0.9 * z[:, :, t - 1] + 4.0 * s

            data = z + 0.2 * torch.randn(self.N_total, self.D, self.T)
        elif self.mode == "bimodal-noisy":
            # Same latent dynamics as ``bimodal`` but with higher
            # observation noise to stress variance-sensitive metrics.
            z = torch.zeros(self.N_total, self.D, self.T)
            z[:, :, 0] = torch.randn(self.N_total, self.D)
            for t in range(1, self.T):
                s = (torch.randint(0, 2, (self.N_total, self.D)).float() * 2.0) - 1.0
                z[:, :, t] = 0.9 * z[:, :, t - 1] + 4.0 * s
            data = z + 0.5 * torch.randn(self.N_total, self.D, self.T)

        elif self.mode == "bimodal-block":
            # Bimodal with changes only every `block` steps.
            z = torch.zeros(self.N_total, self.D, self.T)
            z[:, :, 0] = torch.randn(self.N_total, self.D)

            step_size = 2.5
            noise = 0.05
            block = 5  # increase => harder

            s = (torch.randint(0, 2, (self.N_total, 1)).float() * 2.0) - 1.0
            for t in range(1, self.T):
                if t % block == 0:
                    s = (torch.randint(0, 2, (self.N_total, 1)).float() * 2.0) - 1.0
                z[:, :, t] = (
                    0.9 * z[:, :, t - 1]
                    + s * step_size
                    + noise * torch.randn(self.N_total, self.D)
                )

            data = z
        elif self.mode == "nonlinear-bimodal-lift":
            # Latent (d = 1):
            #   z_t = tanh(z_{t-1}) + delta * s_t + sigma_z * eta_t
            # with s_t in {-1, +1}.
            # Observation:
            #   x_t = W2 @ tanh(W1 @ z_t + b1) + b2 + sigma_x * xi_t
            # where (W1, b1, W2, b2) are sampled once per dataset.
            data = torch.zeros(self.N_total, self.D, self.T)

            z = torch.zeros(self.N_total, 1, self.T)
            z[:, :, 0] = torch.randn(self.N_total, 1)
            for t in range(1, self.T):
                s_t = (torch.randint(0, 2, (self.N_total, 1)).float() * 2.0) - 1.0
                z[:, :, t] = (
                    torch.tanh(z[:, :, t - 1])
                    + NLBL_DELTA * s_t
                    + NLBL_SIGMA_Z * torch.randn(self.N_total, 1)
                )

            W1 = torch.randn(NLBL_LIFT_HIDDEN, 1)
            b1 = torch.randn(NLBL_LIFT_HIDDEN)
            W2 = torch.randn(self.D, NLBL_LIFT_HIDDEN)
            b2 = torch.randn(self.D)

            for t in range(self.T):
                h_t = torch.tanh(z[:, :, t] @ W1.t() + b1)
                x_t = h_t @ W2.t() + b2
                data[:, :, t] = x_t + NLBL_SIGMA_X * torch.randn(self.N_total, self.D)

            if self.expose_gt_latents:
                self._all_gt_latents = z

        elif self.mode == "nonlinear-bimodal-lift-mv":
            # Multivariate variant: latent d = NLBL_MV_LATENT_D, observation
            # lifted to D = NLBL_MV_OBS_D via a tanh-MLP. Per-dim independent
            # bimodal signs (2^d attractors). A is sampled once from a fixed
            # seed so ``synthetic_kernels.nonlinear_bimodal_lift_mv_kernel``
            # can reconstruct it.
            assert self.D == NLBL_MV_OBS_D, (
                f"nonlinear-bimodal-lift-mv expects D={NLBL_MV_OBS_D} "
                f"(matching NLBL_MV_OBS_D); got D={self.D}"
            )
            latent_d = NLBL_MV_LATENT_D
            data = torch.zeros(self.N_total, self.D, self.T)

            # Mixing matrix from a deterministic seed (kernel reads the same).
            A_gen = torch.Generator().manual_seed(NLBL_MV_A_SEED)
            A = torch.randn(latent_d, latent_d, generator=A_gen)

            z = torch.zeros(self.N_total, latent_d, self.T)
            z[:, :, 0] = torch.randn(self.N_total, latent_d)
            for t in range(1, self.T):
                s_t = (
                    torch.randint(0, 2, (self.N_total, latent_d)).float() * 2.0
                ) - 1.0
                Az = z[:, :, t - 1] @ A.t()
                z[:, :, t] = (
                    torch.tanh(Az)
                    + NLBL_DELTA * s_t
                    + NLBL_SIGMA_Z * torch.randn(self.N_total, latent_d)
                )

            W1 = torch.randn(NLBL_MV_HIDDEN_DIM, latent_d)
            b1 = torch.randn(NLBL_MV_HIDDEN_DIM)
            W2 = torch.randn(self.D, NLBL_MV_HIDDEN_DIM)
            b2 = torch.randn(self.D)

            for t in range(self.T):
                h_t = torch.tanh(z[:, :, t] @ W1.t() + b1)
                x_t = h_t @ W2.t() + b2
                data[:, :, t] = x_t + NLBL_SIGMA_X * torch.randn(self.N_total, self.D)

            if self.expose_gt_latents:
                self._all_gt_latents = z

        elif self.mode == "robot-basis-pursuit":
            assert self.D >= 2, "robot-basis-pursuit requires D>=2 (X and Y)"
            z = torch.zeros(self.N_total, self.D, self.T)

            # 1. Intrinsic Parameters
            R = (
                torch.randint(0, 2, (self.N_total,)).float() * 2.0
            ) - 1.0  # +1 (Up) or -1 (Down)

            # Increased clearance so it comfortably clears the massive 0.6 height box
            clearance = 1.0 + 0.4 * torch.rand(self.N_total)

            # Basis rotation (0 to 90 degrees)
            theta = torch.rand(self.N_total) * (torch.pi / 2.0)

            step_sz = 0.18  # Slightly faster to ensure completion in T=48
            tau = 0.04

            c, s = torch.cos(theta), torch.sin(theta)
            b1 = torch.stack([c, s], dim=1) * step_sz
            b2 = torch.stack([-s, c], dim=1) * step_sz

            # The 4 allowed moves: +b1, -b1, +b2, -b2
            actions = torch.stack([b1, -b1, b2, -b2], dim=1)  # (N, 4, 2)

            z[:, 0, 0] = -2.0
            z[:, 1, 0] = 0.0
            batch_indices = torch.arange(self.N_total)

            for t in range(1, self.T):
                prev = z[:, :, t - 1]  # (N, 2)

                # 2. Determine the current Target Waypoint
                # Don't aim for the finish line until safely past the massive box (0.6)
                past_obstacle = prev[:, 0] > 0.65

                # If before obstacle, aim for (0.8, +/- C). If past, aim for end (2.0, 0).
                wx = torch.where(past_obstacle, 2.0, 0.8)
                wy = torch.where(past_obstacle, 0.0, R * clearance)
                waypoint = torch.stack([wx, wy], dim=1)

                # 3. Calculate direction vector to waypoint
                direction = waypoint - prev
                dist_to_wp = torch.norm(direction, dim=1, keepdim=True) + 1e-6
                direction_norm = direction / dist_to_wp  # (N, 2)

                # 4. Score the 4 actions based on alignment with the desired direction
                scores = (actions * direction_norm.unsqueeze(1)).sum(dim=2)  # (N, 4)

                # 5. AHEAD-OF-TIME WALL PENALTY
                # Evaluate where the 4 actions would land
                proposals = prev.unsqueeze(1) + actions  # (N, 4, 2)
                in_box = (
                    (proposals[:, :, 0] > -0.6)
                    & (proposals[:, :, 0] < 0.6)
                    & (proposals[:, :, 1] > -0.6)
                    & (proposals[:, :, 1] < 0.6)
                )
                # If a move hits the box, make its probability exactly ZERO
                scores[in_box] = -1e9

                # 6. STOCHASTIC CHOICE: Convert valid scores to probabilities and sample
                probs = torch.softmax(scores / tau, dim=1)
                distrib = torch.distributions.Categorical(probs)
                chosen_idx = distrib.sample()  # (N,)

                chosen_action = actions[batch_indices, chosen_idx]  # (N, 2)
                next_pos = prev + chosen_action

                # 7. Goal Arrival (if near 2.0 AND past the obstacle phase)
                dist_to_goal = torch.norm(prev - torch.tensor([[2.0, 0.0]]), dim=1)
                at_goal = (dist_to_goal < 0.25) & past_obstacle

                next_pos[at_goal, 0] = 2.0
                next_pos[at_goal, 1] = 0.0

                z[:, :, t] = next_pos

            data = z
        elif self.mode == "student_t":
            # z_t = 0.9 * z_{t-1} + StudentT(df=3)
            # StudentT(df=3) has heavy tails.
            z = torch.zeros(self.N_total, self.D, self.T)
            df = 3.0
            for t in range(1, self.T):
                # Generate Student-t noise: N(0,1) / sqrt(Chi2(df)/df)
                normal = torch.randn(self.N_total, self.D)
                chi2 = torch.distributions.Chi2(df).sample((self.N_total, self.D))
                t_noise = normal / torch.sqrt(chi2 / df)

                z[:, :, t] = 0.9 * z[:, :, t - 1] + 0.1 * t_noise
            data = z

        elif self.mode == "henon-lift":
            # Deterministic Hénon map (chaotic) latent d=2, lifted to D via a
            # tanh-MLP. x_t = 1 - a x_{t-1}^2 + y_{t-1}; y_t = b x_{t-1}.
            assert self.D == HENON_OBS_D, (
                f"henon-lift expects D={HENON_OBS_D}; got D={self.D}"
            )
            latent_d = HENON_LATENT_D
            data = torch.zeros(self.N_total, self.D, self.T)

            # Random ICs near the origin (in the attractor's basin); burn in.
            x = torch.rand(self.N_total) * 0.2 - 0.1
            y = torch.rand(self.N_total) * 0.2 - 0.1
            for _ in range(HENON_BURN_IN):
                x, y = (1.0 - HENON_A * x * x + y).clamp(-2.0, 2.0), HENON_B * x

            z = torch.zeros(self.N_total, latent_d, self.T)
            z[:, 0, 0], z[:, 1, 0] = x, y
            for t in range(1, self.T):
                xp, yp = z[:, 0, t - 1], z[:, 1, t - 1]
                xn = 1.0 - HENON_A * xp * xp + yp + HENON_SIGMA_Z * torch.randn(self.N_total)
                yn = HENON_B * xp + HENON_SIGMA_Z * torch.randn(self.N_total)
                z[:, 0, t] = xn.clamp(-2.0, 2.0)
                z[:, 1, t] = yn.clamp(-1.0, 1.0)

            # Standardise each dim so the lift + unit-variance prior are sane.
            z = (z - z.mean(dim=(0, 2), keepdim=True)) / (
                z.std(dim=(0, 2), keepdim=True) + 1e-6
            )

            # tanh-MLP lift sampled once from a fixed seed.
            g_lift = torch.Generator().manual_seed(HENON_A_SEED)
            W1 = torch.randn(HENON_HIDDEN_DIM, latent_d, generator=g_lift)
            b1 = torch.randn(HENON_HIDDEN_DIM, generator=g_lift)
            W2 = torch.randn(self.D, HENON_HIDDEN_DIM, generator=g_lift)
            b2 = torch.randn(self.D, generator=g_lift)
            for t in range(self.T):
                h_t = torch.tanh(z[:, :, t] @ W1.t() + b1)
                x_t = h_t @ W2.t() + b2
                data[:, :, t] = x_t + HENON_SIGMA_X * torch.randn(self.N_total, self.D)

            if self.expose_gt_latents:
                self._all_gt_latents = z

        elif self.mode == "lorenz":
            # Lorenz 63 system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
            # Standard chaotic parameters; two-lobe attractor with Hausdorff
            # dimension ~2.06. RK4 at dt=0.05 gives ~3 Lyapunov times per T=64
            # sequence, so most trajectories contain 1-3 lobe-switching events —
            # the regime where the Gaussian baseline fails and the diffusion
            # transition is needed. RK4 is used instead of Euler because Lorenz
            # can diverge with Euler from off-attractor starting points.
            assert self.D == 3, (
                f"lorenz mode requires D=3 (x, y, z); got D={self.D}. "
                f"Pass D=3 to the data module."
            )
            sigma_l, rho_l, beta_l = 10.0, 28.0, 8.0 / 3.0
            dt = 0.05
            n_burnin = 200  # ~10 Lyapunov times; ensures all ICs reach the attractor

            def _lorenz_deriv(s: torch.Tensor) -> torch.Tensor:
                xc, yc, zc = s[:, 0], s[:, 1], s[:, 2]
                return torch.stack([
                    sigma_l * (yc - xc),
                    xc * (rho_l - zc) - yc,
                    xc * yc - beta_l * zc,
                ], dim=1)

            def _rk4_step(s: torch.Tensor) -> torch.Tensor:
                k1 = _lorenz_deriv(s)
                k2 = _lorenz_deriv(s + 0.5 * dt * k1)
                k3 = _lorenz_deriv(s + 0.5 * dt * k2)
                k4 = _lorenz_deriv(s + dt * k3)
                return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Initialize inside the attractor bounding box to avoid the large
            # derivatives near z=0 that cause Euler-style divergence.
            state = torch.stack([
                torch.empty(self.N_total).uniform_(-15.0, 15.0),   # x
                torch.empty(self.N_total).uniform_(-20.0, 20.0),   # y
                torch.empty(self.N_total).uniform_(5.0, 45.0),     # z > 0
            ], dim=1)
            for _ in range(n_burnin):
                state = _rk4_step(state)

            # Record T steps after burn-in.
            trajectory = torch.zeros(self.N_total, 3, self.T)
            for t in range(self.T):
                trajectory[:, :, t] = state
                state = _rk4_step(state)

            # Per-channel z-score normalization so all three channels have
            # comparable scale for the encoder/decoder (raw z ranges [0,50],
            # raw x,y range [-20,20] — a 2.5× variance mismatch).
            chan_mean = trajectory.mean(dim=(0, 2), keepdim=True)   # (1, 3, 1)
            chan_std = trajectory.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)
            trajectory = (trajectory - chan_mean) / chan_std

            # Store the clean (noise-free) trajectory before adding noise so
            # the denoising eval can score against it. No extra RNG calls,
            # so generated data remains bit-identical when the flag is off.
            if self.expose_clean_data:
                self._all_clean = trajectory.clone()

            # Small Gaussian observation noise (σ=0.1 in standardized units,
            # SNR≈10) makes posterior inference non-trivial without swamping
            # the lobe-switching signal.
            data = trajectory + 0.1 * torch.randn_like(trajectory)

        return data

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """Return one model-ready item.

        Keys: ``observed_data`` (D, T), ``observation_mask`` (D, T,
        all ones), ``timepoints`` (T,), ``gt_latent`` (d, T) when
        ``expose_gt_latents`` is set and the mode supports it, and
        ``clean_data`` (D, T) when ``expose_clean_data`` is set and the
        mode supports it (currently: ``lorenz``).
        """
        full_seq = self.data[idx]  # (D, T)

        T = full_seq.shape[1]

        item = {
            "observed_data": full_seq,
            "observation_mask": torch.ones_like(full_seq),
            "timepoints": torch.arange(T, dtype=torch.float32),
        }
        if self.expose_gt_latents and self.gt_latents is not None:
            item["gt_latent"] = self.gt_latents[idx]
        if self.expose_clean_data and self.clean_data is not None:
            item["clean_data"] = self.clean_data[idx]
        return item


def generate_lgssm_latents(N: int, T: int, D: int, seed: int = 42):
    """Generate true LGSSM latents and observations.

    z_t = 0.9 * z_{t-1} + 0.1 * eps_z
    x_t = z_t + 0.1 * eps_x
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    z = torch.zeros(N, D, T)
    for t in range(1, T):
        z[:, :, t] = 0.9 * z[:, :, t - 1] + 0.1 * torch.randn(N, D)
    x = z + 0.1 * torch.randn(N, D, T)
    return z, x


class LGSSMLatentDataset(Dataset):
    """Dataset of true LGSSM latents z_t for transition sanity checks."""

    def __init__(self, N: int = 1000, T: int = 100, D: int = 1, seed: int = 42):
        self.N_total = N
        self.T = T
        self.D = D
        self.z, self.x = generate_lgssm_latents(N, T, D, seed)

    def __len__(self):
        return self.N_total

    def __getitem__(self, idx):
        # Return full latent sequence z for this sample
        z_seq = self.z[idx]  # (D, T)
        T = z_seq.shape[1]
        return {
            "z": z_seq,  # (D, T)
            "timepoints": torch.arange(T, dtype=torch.float32),
        }
