import pathlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import zarr


# ==================================================================================================
class MCMCStorage(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self):
        self._samples = []

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def store(self, sample: np.ndarray) -> None:
        pass

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def values(self) -> Any:
        pass


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
    def __init__(self, save_directory: str | pathlib.Path, chunk_size: int) -> None:
        super().__init__()
        self._save_directory = pathlib.Path(save_directory) if save_directory else None
        self._save_directory.parent.mkdir(parents=True, exist_ok=True)
        self._chunk_size = chunk_size
        self._storage_group = zarr.group(store=f"{self._save_directory}.zarr", overwrite=True)
        self._storage = self._storage_group.create_dataset("values", shape=1, dtype=np.float64)

    # ----------------------------------------------------------------------------------------------
    def store(self, sample: np.ndarray) -> None:
        self._samples.append(sample)
        if len(self._result_list) >= self._chunk_size:
            self._save_to_disk()
            self._samples.clear()

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> np.ndarray:
        self._save_to_disk()
        self._samples.clear()
        return self._storage

    # ----------------------------------------------------------------------------------------------
    def _save_to_disk(self) -> None:
        samples_to_store = np.stack(self._samples, axis=-1)
        if self._storage.shape[-1] == 1:
            self._storage.resize(samples_to_store.shape)
            self._storage[:] = samples_to_store
        else:
            self._storage.append(samples_to_store, axis=-1)
