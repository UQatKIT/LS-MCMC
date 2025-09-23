"""Custom storage interface for MCMC sampling.

The custom interface is built in a way to enable storage of samples directly to the disk.
"""
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import zarr


# ==================================================================================================
class MCMCStorage(ABC):
    """Abstract base class for MCMC sample storage."""

    def __init__(self) -> None:
        self._samples = []

    @abstractmethod
    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        """Store a single sample."""
        raise NotImplementedError

    @property
    @abstractmethod
    def values(self) -> Iterable:
        """Return all stored samples."""
        raise NotImplementedError


# ==================================================================================================
class NumpyStorage(MCMCStorage):
    """In-memory storage using numpy arrays."""

    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        self._samples.append(sample)

    @property
    def values(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        stacked_samples = np.stack(self._samples, axis=0)
        return stacked_samples


# ==================================================================================================
class ZarrStorage(MCMCStorage):
    """Disk-based storage using Zarr with chunking."""

    def __init__(self, save_directory: pathlib.Path, chunk_size: int) -> None:
        """Initialize ZarrStorage with save directory and chunk size.

        Args:
            save_directory: Path where the Zarr store will be created.
            chunk_size: Number of samples to accumulate before writing to disk.
                Must be greater than zero.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        super().__init__()
        self._save_directory = save_directory
        self._chunk_size = chunk_size
        self._save_directory.parent.mkdir(parents=True, exist_ok=True)
        self._storage_group = zarr.group(store=f"{self._save_directory}.zarr", overwrite=True)
        self._storage = None

    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        self._samples.append(sample)
        if len(self._samples) >= self._chunk_size:
            self._save_to_disk()
            self._samples.clear()

    @property
    def values(self) -> zarr.array:
        self._save_to_disk()
        self._samples.clear()
        return self._storage

    def _save_to_disk(self) -> None:
        """Save accumulated samples to Zarr storage."""
        if len(self._samples) == 0:
            return
        samples_to_store = np.stack(self._samples, axis=0)
        if self._storage is None:
            self._storage = self._storage_group.create_array(
                "values", shape=samples_to_store.shape, dtype=np.float64
            )
            self._storage[:] = samples_to_store
        else:
            self._storage.append(samples_to_store, axis=0)
