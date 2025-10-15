"""Tests concerning the disk storage."""

from pathlib import Path

import numpy as np
import zarr

from ls_mcmc import storage


def test_zarr_storage_store_and_read(tmp_path: Path) -> None:
    """Test storing samples in ZarrStorage and reading them back."""
    save_dir = tmp_path / "zarr_test"
    chunk_size = 5
    num_samples = 12
    dim = 3

    zarr_storage = storage.ZarrStorage(save_dir, chunk_size=chunk_size)
    rng = np.random.default_rng()
    samples = [rng.normal(size=dim) for _ in range(num_samples)]
    for sample in samples:
        zarr_storage.store(sample)

    # Ensure all samples are flushed to disk and see that the storage reads them correctly
    arr = zarr_storage.values
    assert arr.shape == (num_samples, dim)
    np.testing.assert_equal(arr[:], np.stack(samples))

    # Read data directly from disk
    zarr_group = zarr.open_group(f"{save_dir}.zarr")
    zarr_array = zarr_group["values"]

    np.testing.assert_equal(zarr_array[:], np.stack(samples))
