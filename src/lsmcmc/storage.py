import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import zarr


# ==================================================================================================
class MCMCStorage(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._samples = []

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def store(self, sample: np.ndarray) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def values(self) -> Iterable:
        raise NotImplementedError


# ==================================================================================================
class NumpyStorage(MCMCStorage):
    # ----------------------------------------------------------------------------------------------
    def store(self, sample: np.ndarray) -> None:
        self._samples.append(sample)

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> np.ndarray:
        stacked_samples = np.stack(self._samples, axis=-1)
        return stacked_samples


# ==================================================================================================
class ZarrStorage(MCMCStorage):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, save_directory: pathlib.Path, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        super().__init__()
        self._save_directory = save_directory
        self._chunk_size = chunk_size
        self._save_directory.parent.mkdir(parents=True, exist_ok=True)
        self._storage_group = zarr.group(store=f"{self._save_directory}.zarr", overwrite=True)
        self._storage = None

    # ----------------------------------------------------------------------------------------------
    def store(self, sample: np.ndarray) -> None:
        self._samples.append(sample)
        if len(self._result_list) >= self._chunk_size:
            self._save_to_disk()
            self._samples.clear()

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> zarr.array:
        self._save_to_disk()
        self._samples.clear()
        return self._storage

    # ----------------------------------------------------------------------------------------------
    def _save_to_disk(self) -> None:
        samples_to_store = np.stack(self._samples, axis=-1)
        if self._storage is None:
            self._storage = self._storage_group.create_dataset(
                "values", shape=samples_to_store.shape, dtype=np.float64
            )
            self._storage[:] = samples_to_store
        else:
            self._storage.append(samples_to_store, axis=-1)
