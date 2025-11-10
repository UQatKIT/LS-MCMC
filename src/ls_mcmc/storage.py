"""Custom storage interface for MCMC sampling.

The custom interface is built in a way to enable storage of samples directly to the disk.

Classes:
    MCMCStorage: Abstract base class for storages
    NumpyStorage: Simple in memory storage
    ZarrStorage: Storage with a Zarr backend that saves samples to disk automatically
"""

import logging
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import zarr


# ==================================================================================================
class MCMCStorage(ABC):
    """Abstract base class for MCMC sample storage.

    Methods:
        store: Store a new sample
        values: return all stored samples
        flush: flush samples to disk
    """

    def __init__(self) -> None:
        """Initialize storage."""
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

    @abstractmethod
    def flush(self) -> None:
        """Flushes samples to disk."""
        raise NotImplementedError


# ==================================================================================================
class NumpyStorage(MCMCStorage):
    """In-memory storage using numpy arrays.

    Methods:
        store: Store a new sample
        values: return all stored samples
        flush: Not intended to be used
    """

    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        """Stores sample."""
        self._samples.append(sample)

    @property
    def values(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Return all stored samples as numpy array."""
        stacked_samples = np.stack(self._samples, axis=0)
        return stacked_samples

    def flush(self) -> None:
        """Does nothing except log an error, since NumpyStorage does not support storage to disk.

        Use ZarrStorage if you want storage to disk.
        """
        logger = logging.getLogger(__name__)
        logger.error(
            "Flushing to disk is not intended for In-memory only NumpyStorage."
            "Please use ZarrStorage or another Storage class."
        )


# ==================================================================================================
class ZarrStorage(MCMCStorage):
    """Disk-based storage using Zarr with chunking.

    Methods:
        store: Store a new sample
        values: return all stored samples
        flush: flush samples to disk
        initialize_from_disk: Initialize Zarr Storage from existing storage on the disk
    """

    def __init__(
        self,
        save_directory: pathlib.Path,
        chunk_size: int,
        zarr_storage_group: zarr.Group | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize ZarrStorage with save directory and chunk size.

        Args:
            save_directory: Path where the Zarr store will be created.
            chunk_size: Number of samples to accumulate before writing to disk.
                Must be greater than zero.
            zarr_storage_group: If specified this zarr_storage_group is used
                instead of creating a new one.
                In this case save_directory can be set to None.
            overwrite: Whether to overwrite the data at save_directory or not. Defaults to False.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        super().__init__()
        self._save_directory = save_directory
        self._chunk_size = chunk_size
        if not zarr_storage_group:
            self._save_directory.parent.mkdir(parents=True, exist_ok=True)
            self._storage_group = zarr.create_group(
                store=f"{self._save_directory}.zarr", overwrite=overwrite
            )
        else:
            self._storage_group = zarr_storage_group
        self._storage = None

    @classmethod
    def initialize_from_disk(
        cls, save_directory: pathlib.Path, chunk_size: int = 1
    ) -> "ZarrStorage":
        """Initializes the Zarr storage from an already existing Zarr storage on disk.

        This can be used to either read the samples from the disk after a MCMC run
        or to restart a run and append the new samples.

        Args:
            save_directory: Path where the Zarr store will be created.
            chunk_size: Number of samples to accumulate before writing to disk.
                Must be greater than zero.
        """
        try:
            zarr_storage_group = zarr.open_group(store=f"{save_directory}.zarr")
            z_storage = ZarrStorage(
                save_directory=None,
                chunk_size=chunk_size,
                zarr_storage_group=zarr_storage_group,
            )
            z_storage._storage = z_storage._storage_group["values"]
        except BaseException as e:
            raise ValueError(
                f"Something is wrong with the Zarr Storage location: {save_directory}"
            ) from e
        else:
            return z_storage

    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        """Add sample to ZarrStorage."""
        self._samples.append(sample)
        if len(self._samples) >= self._chunk_size:
            self._save_to_disk()
            self._samples.clear()

    @property
    def values(self) -> zarr.Array:
        """Returns ZarrArray containing all the samples."""
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

    def flush(self) -> None:
        """Save all values to disk."""
        self._save_to_disk()
