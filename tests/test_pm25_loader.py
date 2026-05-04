import os
import tempfile
import pytest
import numpy as np
import torch

from dssd.data.pm25 import load_tsf_to_series, get_pm25_loaders
from dssd.data.dataload import parse_gluon_batch


@pytest.fixture
def mock_tsf_file():
    """Generates a temporary TSF file with missing data markers."""
    fd, filepath = tempfile.mkstemp(suffix=".tsf")

    # We need enough data to satisfy L1+L2 and the test/val splits.
    # Let's say L1=20, L2=10, val_windows=2, test_windows=2.
    # Total needed = 20 + 10 + (2*10) + (2*10) = 70. We'll make it 100.
    length = 100
    num_series = 3  # D = 3

    with os.fdopen(fd, "w") as f:
        f.write("@relation mock_data\n")
        f.write("@attribute something string\n")
        f.write("@data\n")

        for i in range(num_series):
            vals = ["1.5"] * length
            # Inject varying types of missing data markers
            vals[5] = "?"
            vals[10] = "NaN"
            vals[15] = ""

            val_str = ",".join(vals)
            f.write(f"T{i}:2018-01-01 00:00:00:{val_str}\n")

    yield filepath
    os.remove(filepath)


def test_load_tsf_to_series(mock_tsf_file):
    """Ensure padding and parsing maps missing tokens to np.nan."""
    series_list = load_tsf_to_series(mock_tsf_file, freq="H")

    assert len(series_list) == 3
    assert len(series_list[0]) == 100

    # Check that the missing tokens were successfully parsed into NaNs
    vals = series_list[0].values
    assert np.isnan(vals[5]), "? was not converted to NaN"
    assert np.isnan(vals[10]), "NaN was not converted to NaN"
    assert np.isnan(vals[15]), "Empty string was not converted to NaN"

    # Check a valid number
    assert vals[0] == 1.5


def test_pm25_loaders_and_batch_parsing(mock_tsf_file):
    """Ensure loader constructs properly and parses into NaN-safe (B, D, T) tensors."""
    B = 2
    L1 = 20
    L2 = 10
    T = L1 + L2
    D = 3  # Based on the fixture

    train_loader, val_loader, test_loader, scalers = get_pm25_loaders(
        mock_tsf_file, batch_size=B, L1=L1, L2=L2, test_windows=2, val_windows=2
    )

    # 1. Check scalers
    means_t, stds_t = scalers
    assert means_t.shape == (D,)
    assert stds_t.shape == (D,)
    assert not torch.isnan(means_t).any(), "Means contained NaN, z-scoring failed."
    assert not torch.isnan(stds_t).any(), "Stds contained NaN, z-scoring failed."

    # 2. Get a single raw GluonTS batch
    raw_batch = next(iter(train_loader))

    # 3. Parse batch into PyTorch (B, D, T) tensors
    device = torch.device("cpu")
    parsed = parse_gluon_batch(raw_batch, device)

    obs_data = parsed["observed_data"]
    obs_mask = parsed["observation_mask"]
    timepoints = parsed["timepoints"]

    # 4. Assert Expected Shapes
    assert obs_data.shape == (B, D, T), f"Expected {(B, D, T)}, got {obs_data.shape}"
    assert obs_mask.shape == (B, D, T)
    assert timepoints.shape == (B, T)

    # 5. Assert NaN Safety (The most important part)
    assert not torch.isnan(obs_data).any(), (
        "Found NaNs in observed_data! nan_to_num failed."
    )

    # 6. Check that the observation mask caught the missing values
    # Because we randomly sampled windows, we can't definitively check index 5,
    # but we can check the total count of missing/present data in the mask.
    unique_mask_vals = torch.unique(obs_mask)
    for val in unique_mask_vals:
        assert val.item() in [0.0, 1.0], "Mask should only contain 0s and 1s"
