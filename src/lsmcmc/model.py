from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class MCMCModel(ABC):
    @abstractmethod
    def evaluate_potential(self, state: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_point(self) -> np.ndarray:
        raise NotImplementedError
