"""Synthetic time-series datasets for controlled DDSSM experiments (IID, LGSSM, etc.)."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(
        self,
        mode: str,
        split: str = "train",
        N_per_split: int = 1024,
        T: int = 100,
        D: int = 1,
        seed: int = 42,
        dataset_seed: int = 1234,  # new: controls the data split, not the model seed
    ):
        """
        Args:
            mode: 'iid', 'lgssm', ...
            split: 'train', 'val', 'test'
            N_per_split: Number of sequences per split (default 1024)
            T: Length of each sequence
            D: Data dimension
            seed: (legacy, ignored for splitting)
            dataset_seed: Seed for data generation and split
        """
        self.mode = mode
        self.split = split
        self.N_per_split = N_per_split
        self.T = T
        self.D = D

        self.N_total = 3 * N_per_split
        all_data = self._generate_data(dataset_seed)

        if split == "train":
            self.data = all_data[:N_per_split]
        elif split == "val":
            self.data = all_data[N_per_split : 2 * N_per_split]
        elif split == "test":
            self.data = all_data[2 * N_per_split : 3 * N_per_split]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.N = len(self.data)

    def _generate_data(self, seed):
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
            # Latent:
            #   z_t = tanh(z_{t-1}) + delta * s_t + sigma_z * eta_t
            # with s_t in {-1, +1}.
            # Observation:
            #   x_t = W2 @ tanh(W1 @ z_t + b1) + b2 + sigma_x * xi_t
            # where (W1, b1, W2, b2) are sampled once per dataset.
            delta = 2.0
            sigma_z = 0.1
            sigma_x = 0.1
            hidden_dim = 8
            data = torch.zeros(self.N_total, self.D, self.T)

            z = torch.zeros(self.N_total, 1, self.T)
            z[:, :, 0] = torch.randn(self.N_total, 1)
            for t in range(1, self.T):
                s_t = (torch.randint(0, 2, (self.N_total, 1)).float() * 2.0) - 1.0
                z[:, :, t] = (
                    torch.tanh(z[:, :, t - 1])
                    + delta * s_t
                    + sigma_z * torch.randn(self.N_total, 1)
                )

            W1 = torch.randn(hidden_dim, 1)
            b1 = torch.randn(hidden_dim)
            W2 = torch.randn(self.D, hidden_dim)
            b2 = torch.randn(self.D)

            for t in range(self.T):
                h_t = torch.tanh(z[:, :, t] @ W1.t() + b1)
                x_t = h_t @ W2.t() + b2
                data[:, :, t] = x_t + sigma_x * torch.randn(self.N_total, self.D)

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

        return data

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        full_seq = self.data[idx]  # (D, T)

        # Standard format for DDSSM training
        # observed_data: (D, T)
        # observation_mask: (D, T)
        # timepoints: (T)

        T = full_seq.shape[1]

        return {
            "observed_data": full_seq,
            "observation_mask": torch.ones_like(full_seq),
            "timepoints": torch.arange(T, dtype=torch.float32),
        }


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
